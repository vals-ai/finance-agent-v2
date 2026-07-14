import asyncio
import json
import logging
import math
import os
import re
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from bs4 import BeautifulSoup
from model_library.agent import Tool, ToolOutput
from model_library.base import LLM
from tavily import AsyncTavilyClient

from simpleeval import SimpleEval

from .exceptions import (
    RetryExhaustedError,
    SecFilingNotFoundError,
    get_retry_policy,
    retry_http_errors,
    retry_with_policy,
)
from .key_rotator import KeyRotator, get_rotator


MAX_END_DATE = "2026-03-01"
VALID_TOOLS = [
    "web_search",
    "retrieve_information",
    "parse_html_page",
    "edgar_search",
    "calculator",
    "price_history",
]

_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SEC_ARCHIVE_FILE_REGEX = re.compile(r"^(/Archives/edgar/data/\d+/\d{18}/)[^/]+$")


def _canonical_sec_index_url(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc.lower() not in {
        "sec.gov",
        "www.sec.gov",
    }:
        return None

    match = _SEC_ARCHIVE_FILE_REGEX.fullmatch(parsed.path)
    if match is None:
        return None
    return f"https://{parsed.netloc.lower()}{match.group(1)}index.json"


async def _missing_sec_index_url(url: str) -> str | None:
    try:
        canonical_url = _canonical_sec_index_url(url)
        if canonical_url is None:
            return None

        async with aiohttp.ClientSession() as session:
            async with session.get(
                canonical_url,
                timeout=aiohttp.ClientTimeout(total=60),
                headers={"User-Agent": "ValsAI/antoine@vals.ai"},
                allow_redirects=False,
            ) as response:
                return canonical_url if response.status == 404 else None
    except Exception:
        return None


def _validate_date_format(field_name: str, value: str) -> None:
    if not _DATE_REGEX.match(value):
        raise ValueError(
            f"Invalid {field_name} format: '{value}'. Expected YYYY-MM-DD."
        )


class Calculator(Tool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression and return the result. "
        "Use this tool for all arithmetic calculations instead of computing by hand. "
        "Supports: +, -, *, /, ** (exponentiation), % (modulo), "
        "and parentheses for grouping. "
        "Available functions: abs(), min(), max(), sqrt(), log(), log10(). "
        "Examples: '(5000 - 3200) * 0.21', '(2865507 / 1905871) ** 0.5 - 1', '14060 / 2148'."
    )
    parameters: dict[str, Any] = {
        "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate",
        }
    }
    required: list[str] = ["expression"]

    def __init__(self) -> None:
        self._evaluator = SimpleEval(
            functions={
                "abs": abs,
                "min": min,
                "max": max,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
            }
        )

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        expression = args.get("expression", "")
        if not expression:
            return ToolOutput(
                output="Error: expression must not be empty", error="empty expression"
            )
        try:
            result = self._evaluator.eval(expression)
            return ToolOutput(output=str(result))
        except ZeroDivisionError:
            error_msg = f"Error: division by zero in '{expression}'"
            logger.warning(error_msg)
            return ToolOutput(output=error_msg, error=error_msg)
        except OverflowError:
            error_msg = f"Error: numerical overflow in '{expression}'"
            logger.warning(error_msg)
            return ToolOutput(output=error_msg, error=error_msg)
        except Exception as e:
            logger.warning(f"Calculator error for '{expression}': {e}")
            error_msg = f"Error: invalid expression '{expression}'"
            return ToolOutput(output=error_msg, error=error_msg)


class SubmitFinalResult(Tool):
    name = "submit_final_result"
    description = (
        "Submits the final answer to the user. You should include your final answer, as well as any necessary "
        "reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question. "
        "You MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit. "
        "You will not be able to continue working after this tool is called; the conversation will be ended."
    )
    parameters: dict[str, Any] = {
        "final_result": {
            "type": "string",
            "description": "The final result to submit to the user",
        }
    }
    required: list[str] = ["final_result"]

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            final_result = args["final_result"]
            if not final_result:
                raise ValueError("Final result must not be empty")
            return ToolOutput(output=final_result, done=True)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Submission failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg, done=False)


class TavilyWebSearch(Tool):
    name = "web_search"
    description = "Search the public internet for information. Each result will contain a url, a title, and one excerpt taken directly from the page."
    parameters: dict[str, Any] = {
        "search_query": {
            "type": "string",
            "description": "The query to search for",
        },
        "start_date": {
            "type": "string",
            "description": "(optional) The start date for the search range in the format YYYY-MM-DD. Must not be equal to end_date.",
        },
        "end_date": {
            "type": "string",
            "description": f"(optional) The end date for the search range in the format YYYY-MM-DD. If the value is later than {MAX_END_DATE}, it will be set to {MAX_END_DATE}.",
        },
        "number_of_results": {
            "type": "integer",
            "description": "(optional) The number of search results to return.",
            "maximum": 20,
            "minimum": 1,
            "default": 10,
        },
    }
    required = ["search_query"]

    def __init__(self, tavily_api_key: str | None = None):
        if not tavily_api_key:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set")
        self.client = AsyncTavilyClient(api_key=tavily_api_key)

    @retry_http_errors(429, 503, max_tries=8)
    async def _execute_search(
        self,
        search_query: str,
        start_date: str | None = None,
        end_date: str = MAX_END_DATE,
        number_of_results: int = 10,
    ) -> list[dict[str, Any]]:
        kwargs = {}

        if end_date:
            _validate_date_format("end_date", end_date)
            end_date = min(end_date, MAX_END_DATE)

        if start_date:
            _validate_date_format("start_date", start_date)
            start_date = min(start_date, MAX_END_DATE)
            if start_date > end_date:
                raise ValueError(
                    f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'"
                )

            kwargs["start_date"] = start_date

        response = await self.client.search(
            search_depth="fast",
            end_date=end_date,
            max_results=number_of_results,
            chunks_per_source=1,
            query=search_query,
            **kwargs,
        )

        return response.get("results", [])

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            kwargs = {k: v for k, v in args.items() if k in self.parameters}
            results = await self._execute_search(**kwargs)
            return ToolOutput(output=json.dumps(results, default=str))
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Web search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class EDGARSearch(Tool):
    name = "edgar_search"
    description = (
        "Search the EDGAR Database through the SEC API. "
        "You should provide a search query. You can also optionally provide a start date, an end date, a page number, top N results, a list of form types, and/or a list of CIKs. "
        "The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing."
    )
    parameters: dict[str, Any] = {
        "search_query": {
            "type": "string",
            "description": 'The case-insensitive search-term or phrase to search the contents of filings and their attachments. This can be a single word, phrase, or combination of words and phrases. Supported search features include wildcards (*), Boolean operators (OR, NOT), and exact phrase matching by enclosing phrases in quotation marks ("exact phrase"). By default, all terms are joined by an implicit AND operator.',
        },
        "form_types": {
            "type": "array",
            "description": "(optional) Limits search to specific EDGAR form types (e.g., ['8-K', '10-Q']) list of strings. Default: all form types",
            "items": {"type": "string"},
        },
        "ciks": {
            "type": "array",
            "description": '(optional) Filters results to filings from specified CIKs, type list of strings. Leading zeros are optional but may be included. Example: [ "0001811414", "1318605" ]. Default: all CIKs',
            "items": {"type": "string"},
        },
        "start_date": {
            "type": "string",
            "description": f"(optional) Start date for the search range in yyyy-mm-dd format. If the value is a date that is later than {MAX_END_DATE}, it will be set to {MAX_END_DATE}.",
            "default": "1900-01-01",
        },
        "end_date": {
            "type": "string",
            "description": f"(optional) End date for the search range, in the same format as startDate. If the value is a date that is later than {MAX_END_DATE}, it will be set to {MAX_END_DATE}.",
            "default": MAX_END_DATE,
        },
        "page": {
            "type": "integer",
            "description": "(optional) Used for pagination. Each page contains up to 100 matching filings. Increase the page number to retrieve the next set of 100 filings. Example: 3 retrieves the third page. Default: 1",
            "default": 1,
        },
        "top_n_results": {
            "type": "integer",
            "description": "(optional) Return only the first N results out of 100 from the page. If not provided, all 100 results will be returned. E.g. if page is 2, and top_n_results is 10, you will receive results 100 to 110.",
            "maximum": 100,
            "default": 100,
        },
    }
    required = ["search_query"]

    def __init__(
        self,
        sec_api_key: str | None = None,
        key_rotator: KeyRotator | None = None,
    ):
        if key_rotator is not None:
            self._key_rotator = key_rotator
        elif sec_api_key:
            self._key_rotator = KeyRotator([sec_api_key])
        else:
            self._key_rotator = get_rotator("SEC_EDGAR_API_KEY")
        self.sec_api_url: str = "https://api.sec-api.io/full-text-search"

    @retry_with_policy({429: 20, 503: 20})
    async def _execute_search(
        self,
        search_query: str,
        start_date: str = "1900-01-01",
        end_date: str = MAX_END_DATE,
        top_n_results: int = 100,
        page: int = 1,
        form_types: list[str] | str | None = None,
        ciks: list[str] | str | None = None,
    ) -> list[str]:
        if form_types is not None and not isinstance(form_types, list):
            raise ValueError(
                f"The parameter form_types must be a list if provided. Was of type {type(form_types)}"
            )

        if ciks is not None and not isinstance(ciks, list):
            raise ValueError(
                f"The parameter ciks must be a list if provided. Was of type {type(ciks)}"
            )

        _validate_date_format("start_date", start_date)
        _validate_date_format("end_date", end_date)

        start_date = min(start_date, MAX_END_DATE)
        end_date = min(end_date, MAX_END_DATE)

        if start_date > end_date:
            raise ValueError(
                f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'"
            )

        payload: dict[str, str | int | list[str]] = {
            "query": search_query,
            "startDate": start_date,
            "endDate": end_date,
        }

        if page:
            payload["page"] = page

        if form_types:
            payload["formTypes"] = form_types

        if ciks:
            payload["ciks"] = ciks

        async with self._key_rotator.acquire() as key:
            headers = {
                "Content-Type": "application/json",
                "Authorization": key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.sec_api_url, json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

        results = result.get("filings", [])
        results = results[: int(top_n_results)]

        return results

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            kwargs = {k: v for k, v in args.items() if k in self.parameters}
            results = await self._execute_search(**kwargs)
            return ToolOutput(output=json.dumps(results, default=str))
        except aiohttp.ClientResponseError as e:
            if e.status in (429, 503):
                raise RetryExhaustedError(
                    f"SEC API retry attempts exhausted for HTTP {e.status}"
                ) from e
            error_msg = str(e)
            logger.warning(f"EDGAR search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"EDGAR search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class ParseHtmlPage(Tool):
    name = "parse_html_page"
    description = (
        "This tool is used to parse the contents of an HTML page and save it to the agent's data storage system. "
        "The tool will retrieve the HTML page from the URL provided, then parse it from HTML to plain text. "
        "Finally, it will save it to the agent's data storage system under the key provided. "
        "You can use the retrieve_information tool to later retrieve information about the stored page."
    )
    parameters: dict[str, Any] = {
        "url": {"type": "string", "description": "The URL of the HTML page to parse"},
        "key": {
            "type": "string",
            "description": "The key to use when saving the result in the conversation's data storage.",
        },
    }
    required = ["url", "key"]

    async def _parse_html_page(self, url: str) -> str:
        @retry_with_policy(get_retry_policy(url))
        async def _fetch(fetch_url: str) -> str:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        fetch_url,
                        timeout=aiohttp.ClientTimeout(total=60),
                        headers={"User-Agent": "ValsAI/antoine@vals.ai"},
                    ) as response:
                        response.raise_for_status()
                        return await response.text()
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds. The URL might be blocked or the server is taking too long to respond."
                    )
                except Exception:
                    raise

        try:
            html_content = await _fetch(url)
        except aiohttp.ClientResponseError as e:
            if e.status == 503 and (canonical_url := await _missing_sec_index_url(url)):
                raise SecFilingNotFoundError(
                    f"SEC filing not found: {canonical_url}"
                ) from e
            if e.status in (429, 503) and "sec.gov" in url:
                raise RetryExhaustedError(
                    f"sec.gov retry attempts exhausted for HTTP {e.status}"
                ) from e
            raise

        soup = BeautifulSoup(html_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            _ = script_or_style.extract()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    async def _save_tool_output(
        self, output: str, key: str, state: dict[str, Any]
    ) -> str:
        if not output:
            raise ValueError("HTML output was empty")

        tool_result = ""
        if key in state:
            tool_result = "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
        tool_result += (
            f"SUCCESS: The result has been saved to the data storage under the key: {key}."
            + "\n"
        )

        state[key] = output

        keys_list = "\n".join(state.keys())
        tool_result += (
            f"""
        The data_storage currently contains the following keys:
        {keys_list}
        """.strip()
            + "\n"
        )

        return tool_result

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            url = args["url"]
            key = args["key"]
            text_output = await self._parse_html_page(url)
            tool_result = await self._save_tool_output(text_output, key, state)
            return ToolOutput(output=tool_result)
        except RetryExhaustedError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Parse HTML page failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class PriceHistory(Tool):
    name = "price_history"
    description = (
        "Fetch historical daily price data for a specific asset class. "
        "Returns a CSV table with one row per day. "
        "Use asset_class='equity' or 'etf' for US-listed stocks/ETFs (e.g. AAPL, SPY), 'crypto' for pairs like btcusd (lowercase, no dash), "
        "or 'fx' for pairs like audusd. Non-US equities and most indices/futures are not covered by the pricing provider. "
        "Each row includes raw OHLC (open/high/low/close), split- and dividend-adjusted OHLC (adjOpen/adjHigh/adjLow/adjClose), "
        "raw and adjusted volume, divCash (dividend amount on the date), and splitFactor. "
    )
    parameters: dict[str, Any] = {
        "ticker": {
            "type": "string",
            "description": (
                "Ticker symbol. High-level symbology rules: "
                "(1) Equity/ETF/mutual fund: use the US exchange-listed symbol (AAPL, SPY, VFIAX). "
                "Symbols are case-insensitive and contain only letters and dashes. "
                "(2) Share classes use a dash, not a dot: 'BRK-A' / 'BRK-B' (not 'BRK.A'), 'BF-B'. "
                "(3) Securities are identified by the ticker as listed on the primary US exchange; "
                "non-US listings, OTC pink sheets, indices, and futures are generally not covered. "
                "(4) Tickers can be reused after delisting; data is point-in-time for the entity "
                "that held the ticker on each date. "
                "(5) Crypto: concatenated lowercase pair with no separator, e.g. 'btcusd', 'ethusd'. "
                "(6) FX: concatenated lowercase pair with no separator, e.g. 'audusd', 'eurusd'."
            ),
        },
        "start_date": {
            "type": "string",
            "description": "Start date YYYY-MM-DD (inclusive).",
        },
        "end_date": {
            "type": "string",
            "description": "End date YYYY-MM-DD (inclusive).",
        },
        "asset_class": {
            "type": "string",
            "enum": ["equity", "etf", "crypto", "fx"],
            "description": "Asset class used to route to the proper pricing endpoint.",
        },
    }
    required = ["ticker", "start_date", "end_date", "asset_class"]

    EQUITY_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    CRYPTO_URL = "https://api.tiingo.com/tiingo/crypto/prices"
    FX_URL = "https://api.tiingo.com/tiingo/fx/{pair}/prices"

    _COLUMNS = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjOpen",
        "adjHigh",
        "adjLow",
        "adjClose",
        "volume",
        "adjVolume",
        "divCash",
        "splitFactor",
        "volumeNotional",  # crypto only
        "tradesDone",  # crypto only
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("PRICING_DATA_API_KEY")
        if not self._api_key:
            raise ValueError("PRICING_DATA_API_KEY is required for price_history")

    @retry_with_policy({429: 4, 503: 4})
    async def _fetch(
        self, endpoint: str, ticker: str, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "startDate": start_date,
            "endDate": end_date,
            "format": "json",
        }
        if endpoint == "equity":
            url = self.EQUITY_URL.format(ticker=ticker.upper())
        elif endpoint == "crypto":
            url = self.CRYPTO_URL
            params["tickers"] = ticker.lower()
            params["resampleFreq"] = "1day"
        else:  # fx
            url = self.FX_URL.format(pair=ticker.lower())
            params["resampleFreq"] = "1day"

        headers = {"Authorization": f"Token {self._api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                response.raise_for_status()
                payload = await response.json()

        # Crypto responses are nested: [{ticker, priceData: [...]}, ...]
        if endpoint == "crypto":
            if not payload:
                return []
            first = payload[0] if isinstance(payload, list) else payload
            return first.get("priceData", []) if isinstance(first, dict) else []
        return payload

    @classmethod
    def _records_to_csv(cls, records: list[dict[str, Any]]) -> str:
        # Drop columns that are never populated in this response (e.g. adj*/divCash/splitFactor for crypto/fx).
        active_cols = ["date"] + [
            col for col in cls._COLUMNS[1:]
            if any(r.get(col) is not None for r in records)
        ]
        lines = [",".join(active_cols)]
        for record in records:
            date_value = record.get("date", "")
            if isinstance(date_value, str) and "T" in date_value:
                date_value = date_value.split("T", 1)[0]
            row = [str(date_value)]
            for col in active_cols[1:]:
                value = record.get(col)
                row.append("N/A" if value is None else str(value))
            lines.append(",".join(row))
        return "\n".join(lines)

    @staticmethod
    def _route_endpoint(asset_class: str) -> str:
        normalized = asset_class.strip().lower()
        if normalized in {"equity", "etf"}:
            return "equity"
        if normalized in {"crypto", "fx"}:
            return normalized
        raise ValueError(
            f"Unsupported asset_class '{asset_class}'. Expected one of equity, etf, crypto, fx."
        )

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            ticker = str(args["ticker"]).strip()
            start_date = args["start_date"]
            end_date = args["end_date"]
            asset_class = str(args.get("asset_class") or "").strip().lower()

            if not ticker:
                raise ValueError("ticker is required")

            _validate_date_format("start_date", start_date)
            _validate_date_format("end_date", end_date)
            start_date = min(start_date, MAX_END_DATE)
            end_date = min(end_date, MAX_END_DATE)
            if start_date > end_date:
                raise ValueError(
                    f"start_date '{start_date}' is later than end_date '{end_date}'."
                )
            if not asset_class:
                raise ValueError("asset_class is required")

            endpoint = self._route_endpoint(asset_class)
            records = await self._fetch(endpoint, ticker, start_date, end_date)
            if not records:
                return ToolOutput(
                    output=f"No pricing data returned for {ticker} ({asset_class}) {start_date}..{end_date}"
                )
            return ToolOutput(output=self._records_to_csv(records))
        except aiohttp.ClientResponseError as e:
            error_msg = f"Pricing data HTTP {e.status}: {e.message}"
            logger.warning(error_msg)
            return ToolOutput(output=error_msg, error=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Pricing data fetch failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class RetrieveInformation(Tool):
    name = "retrieve_information"
    description = (
        "This tool allows you to retrieve data from previously saved documents from the agent's data storage system, by applying an LLM prompt to the stored document.\n"
        "\n"
        "To use the tool, you will need to provide a prompt. This prompt will include both the query to be sent to the LLM, "
        "as well as the keys of files you have previously saved to the data storage system.\n"
        "\n"
        'For example, if you want to analyze data stored under the key "financial_report", your prompt should look like the following:\n'
        '"Analyze the following financial report and extract the revenue figures: {{financial_report}}"\n'
        "\n"
        "The {{key_name}} will be replaced with the full text of the document stored under that key before the query is sent.\n"
        "\n"
        "IMPORTANT: Your prompt MUST include at least one key from the data storage using this exact format: {{key_name}}. "
        "If you don't use this exact format with double braces, the tool will fail to retrieve the information.\n"
        "\n"
        "You can also optionally only pass *a portion* of each document to the LLM, rather than the entire document. This can be used to avoid token limit errors or improve efficiency. "
        "To do so, use the input_character_ranges parameter to specify which portions of documents to extract. "
        'For example, if "financial_report" contains "Annual Report 2023" and you specify:  [{"key": "financial_report", "start": 1, "end": 6}], '
        'then only "nnual" will be inserted into the prompt (characters 1 through 5, as end is exclusive).'
    )
    parameters: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder.",
        },
        "input_character_ranges": {
            "type": "array",
            "description": "An optional list of character range specifications for extracting only portions of documents. Each object should have 'key' (the document key), 'start' (start character index, inclusive), and 'end' (end character index, exclusive). By default, the full document is used if this parameter is not provided or if a key is not included in the list.",
            "items": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The document key from data storage",
                    },
                    "start": {
                        "type": "integer",
                        "description": "The starting character index (inclusive)",
                    },
                    "end": {
                        "type": "integer",
                        "description": "The ending character index (exclusive)",
                    },
                },
                "required": ["key", "start", "end"],
            },
        },
    }
    required = ["prompt"]

    def __init__(self, llm: LLM):
        self._llm = llm

    def _validate_inputs(
        self, prompt: str, input_character_ranges: list, state: dict[str, Any]
    ) -> dict[str, tuple[int, int]]:
        """Validate prompt placeholders, character ranges, and data storage keys. Returns the parsed ranges dict."""
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. Please try again with the correct format. You can add documents to the data storage with parse_html_page."
            )

        ranges_dict = {}
        for range_spec in input_character_ranges:
            if not isinstance(range_spec, dict):
                raise ValueError(
                    "ERROR: Each item in input_character_ranges must be an object with 'key', 'start', and 'end' fields."
                )
            if (
                "key" not in range_spec
                or "start" not in range_spec
                or "end" not in range_spec
            ):
                raise ValueError(
                    "ERROR: Each range specification must have 'key', 'start', and 'end' fields."
                )
            ranges_dict[range_spec["key"]] = (range_spec["start"], range_spec["end"])

        keys = re.findall(r"{{([^{}]+)}}", prompt)
        keys_set = set(keys)

        for range_key in ranges_dict.keys():
            if range_key not in keys_set:
                raise ValueError(
                    f"ERROR: The key '{range_key}' is specified in input_character_ranges but is not referenced in the prompt. "
                    f"Keys in prompt: {', '.join(keys_set) if keys_set else '(none)'}"
                )

        for key in keys:
            if key not in state:
                raise KeyError(
                    f"ERROR: The key '{key}' was not found in the data storage. Available keys are: {', '.join(state.keys())}. Use the parse_html_page tool to add keys to the data storage."
                )

        return ranges_dict

    def _format_prompt(
        self,
        prompt: str,
        ranges_dict: dict[str, tuple[int, int]],
        state: dict[str, Any],
    ) -> str:
        """Substitute data storage content into prompt placeholders, applying character ranges.

        Uses a single re.sub pass so that document content is never rescanned — this
        preserves literal curly braces (e.g. JSON the LLM included) and prevents
        {{key}}-looking sequences inside one document from triggering substitution of
        another key's placeholder.
        """

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            doc_content = state[key]
            if key in ranges_dict:
                start_idx, end_idx = ranges_dict[key]
                return doc_content[start_idx:end_idx]
            return doc_content

        return re.sub(r"{{([^{}]+)}}", replace, prompt)

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            prompt: str = args["prompt"]
            input_character_ranges = args.get("input_character_ranges", [])
            if input_character_ranges is None:
                input_character_ranges = []

            ranges_dict = self._validate_inputs(prompt, input_character_ranges, state)
            prompt = self._format_prompt(prompt, ranges_dict, state)
            response = await self._llm.query(prompt)
            return ToolOutput(
                output=response.output_text_str,
                metadata=response.metadata,
            )
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Retrieve information failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)
