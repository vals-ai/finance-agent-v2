import logging
import unittest
from contextlib import contextmanager
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import aiohttp

from finance_agent.exceptions import RetryExhaustedError
from finance_agent.tools import ParseHtmlPage


FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/874238/000087423826000005/"
    "0000874238-26-000005-index.htm?output=1#top"
)
CANONICAL_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/874238/000087423826000005/index.json"
)


def _response(url: str, status: int) -> MagicMock:
    response = MagicMock(status=status)
    if status >= 400:
        response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(real_url=url),
            history=(),
            status=status,
        )
    return response


def _response_context(response: MagicMock) -> MagicMock:
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=None)
    return context


@contextmanager
def _mock_requests(*responses: MagicMock | Exception):
    session = MagicMock()
    session.get.side_effect = [
        response if isinstance(response, Exception) else _response_context(response)
        for response in responses
    ]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    with patch("finance_agent.tools.aiohttp.ClientSession", return_value=context):
        yield session


class ParseHtmlPageTest(unittest.IsolatedAsyncioTestCase):
    async def test_downgrades_verified_missing_sec_filing_to_tool_error(self) -> None:
        with (
            patch("finance_agent.tools.get_retry_policy", return_value={503: 1}),
            _mock_requests(
                _response(FILING_URL, 503),
                _response(CANONICAL_INDEX_URL, 404),
            ) as session,
        ):
            result = await ParseHtmlPage().execute(
                {"url": FILING_URL, "key": "filing"}, {}, logging.getLogger()
            )

        self.assertEqual(result.output, f"SEC filing not found: {CANONICAL_INDEX_URL}")
        self.assertEqual(result.error, result.output)
        self.assertEqual(
            session.get.call_args_list,
            [
                call(FILING_URL, timeout=ANY, headers=ANY),
                call(
                    CANONICAL_INDEX_URL,
                    timeout=ANY,
                    headers=ANY,
                    allow_redirects=False,
                ),
            ],
        )

    async def test_preserves_503_when_canonical_index_is_not_404(self) -> None:
        for status in (200, 302, 403, 429, 500, 503):
            with (
                self.subTest(status=status),
                patch("finance_agent.tools.get_retry_policy", return_value={503: 1}),
                _mock_requests(
                    _response(FILING_URL, 503),
                    _response(CANONICAL_INDEX_URL, status),
                ),
                self.assertRaises(RetryExhaustedError),
            ):
                await ParseHtmlPage().execute(
                    {"url": FILING_URL, "key": "filing"},
                    {},
                    logging.getLogger(),
                )

    async def test_preserves_503_when_canonical_index_cannot_be_verified(self) -> None:
        with (
            patch("finance_agent.tools.get_retry_policy", return_value={503: 1}),
            _mock_requests(
                _response(FILING_URL, 503),
                aiohttp.ClientConnectionError("connection failed"),
            ),
            self.assertRaises(RetryExhaustedError),
        ):
            await ParseHtmlPage().execute(
                {"url": FILING_URL, "key": "filing"}, {}, logging.getLogger()
            )

    async def test_preserves_429_without_probing_canonical_index(self) -> None:
        with (
            patch("finance_agent.tools.get_retry_policy", return_value={429: 1}),
            _mock_requests(_response(FILING_URL, 429)) as session,
            self.assertRaises(RetryExhaustedError),
        ):
            await ParseHtmlPage().execute(
                {"url": FILING_URL, "key": "filing"}, {}, logging.getLogger()
            )

        self.assertEqual(session.get.call_count, 1)

    async def test_preserves_503_for_unmatched_sec_url(self) -> None:
        urls = (
            "https://www.sec.gov/files/unavailable.html",
            "http://www.sec.gov/Archives/edgar/data/874238/000087423826000005/file.htm",
            "https://www.sec.gov.example/Archives/edgar/data/874238/000087423826000005/file.htm",
            "https://www.sec.gov/Archives/edgar/data/874238/00008742382600005/file.htm",
            "https://[www.sec.gov/Archives/edgar/data/874238/000087423826000005/file.htm",
        )
        for url in urls:
            with (
                self.subTest(url=url),
                patch("finance_agent.tools.get_retry_policy", return_value={503: 1}),
                _mock_requests(_response(url, 503)) as session,
                self.assertRaises(RetryExhaustedError),
            ):
                await ParseHtmlPage().execute(
                    {"url": url, "key": "filing"}, {}, logging.getLogger()
                )

            self.assertEqual(session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
