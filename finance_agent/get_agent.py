from pathlib import Path

from model_library.agent import Agent, AgentConfig, AgentHooks, TimeLimit, ToolCallRecord, TurnLimit, TurnResult, default_before_query, truncate_oldest
from model_library.base import LLM, LLMConfig, RawResponse, TextInput
from model_library.base.input import InputItem, SystemInput
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel

from .prompt import QUESTION_PROMPT, SYSTEM_PROMPT
from .exceptions import RetryExhaustedError
from .tools import (
    VALID_TOOLS,
    Calculator,
    EDGARSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    Tool,
    PriceHistory,
)


MAX_TIME_SECONDS = 60 * 60  # 1 hour


class Parameters(BaseModel):
    model_name: str
    max_time_seconds: int = MAX_TIME_SECONDS
    max_turns: int | None = None
    tools: list[str] = VALID_TOOLS
    llm_config: LLMConfig


def build_input(question: str) -> list[InputItem]:
    return [SystemInput(text=SYSTEM_PROMPT), TextInput(text=QUESTION_PROMPT.format(question=question))]


def create_llm(parameters: Parameters) -> LLM:
    """Create an LLM instance from parameters using the model registry."""
    return get_registry_model(parameters.model_name, parameters.llm_config)


def get_agent(
    parameters: Parameters,
    llm: LLM | None = None,
    log_dir: Path | None = None,
) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    if llm is None:
        llm = create_llm(parameters)

    available_tools: dict[str, type[Tool]] = {
        "web_search": TavilyWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "edgar_search": EDGARSearch,
        "calculator": Calculator,
        "price_history": PriceHistory,
    }

    selected_tools: list[Tool] = []
    for tool_name in parameters.tools:
        if tool_name not in available_tools:
            raise Exception(f"Tool {tool_name} not found in tools. Available tools: {available_tools.keys()}")
        tool_cls = available_tools[tool_name]
        if tool_name == "retrieve_information":
            selected_tools.append(tool_cls(llm=llm))  # type: ignore[call-arg]
        else:
            selected_tools.append(tool_cls())  # type: ignore[call-arg]

    selected_tools.append(SubmitFinalResult())

    # Agent loop stop behavior:
    #
    # The loop exits when:
    # - submit_final_result tool returns done=True -> break, no final_error
    # - max_time exceeded -> time limit triggers, final_error = MaxTimeExceeded
    # - query error re-raised by before_query -> caught by outer except, final_error set
    # - context window exceeded -> before_query truncates history, continues (not a stop)
    # - text-only response (no tool calls) -> continues (overridden below, default would stop)
    #
    # Answer extraction (default_determine_answer):
    # - On final_error (max_time, query error, etc): returns ""
    # - On clean exit: returns done tool output (submit_final_result)
    # - Fallback to LLM text: exists in default but is dead code here, because
    #   _should_stop=False means the only clean exit (no final_error) is the done tool break,
    #   and default_determine_answer finds the done record before reaching the text fallback.

    def _before_query(history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
        """Truncate on context window overflow, re-raise all other errors (stops the loop).

        Also injects a nudge to call a tool when the previous turn had no tool calls
        (last item in history is a RawResponse, meaning no ToolResult was appended).
        """
        if isinstance(last_error, MaxContextWindowExceededError):
            return truncate_oldest(history)
        if history and isinstance(history[-1], RawResponse):
            history.append(TextInput(text=(
                "Your last response produced no tool call. "
                "Call `submit_final_result` if you have a final result, "
                "otherwise continue with the next tool call."
            )))
        return default_before_query(history, last_error)

    def _on_tool_result(record: ToolCallRecord, state: dict) -> None:
        if record.error and record.error.type == "RetryExhaustedError":
            raise RetryExhaustedError(record.error.message)

    def _should_stop(turn_result: TurnResult) -> bool:
        """Never stop on text-only responses.

        The model library default stops on text-only responses (no tool calls), but the finance agent
        should keep looping until the model calls submit_final_result or a configured limit is hit.
        """
        return False

    return Agent(
        llm=llm,
        tools=selected_tools,
        name="finance",
        log_dir=log_dir or Path("logs"),
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=parameters.max_turns) if parameters.max_turns else None,
            time_limit=TimeLimit(max_seconds=parameters.max_time_seconds),
        ),
        hooks=AgentHooks(
            before_query=_before_query,
            should_stop=_should_stop,
            on_tool_result=_on_tool_result,
        ),
    )
