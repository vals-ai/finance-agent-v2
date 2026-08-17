import asyncio
import logging

from model_library.agent import Tool, ToolOutput
from model_library.base import LLMConfig
from model_library.base.input import SystemInput

from finance_agent.get_agent import Parameters, build_input
from finance_agent.tool_budget import BudgetedTool, ToolBudget, budget_instructions


class FakeTool(Tool):
    name = "web_search"
    description = "fake"
    parameters = {}
    required = []

    def __init__(self):
        self.calls = 0

    async def execute(self, args, state, logger):
        self.calls += 1
        return ToolOutput(output="result")


def test_budget_blocks_calls_before_execution():
    async def run():
        inner = FakeTool()
        tool = BudgetedTool(inner, ToolBudget(2, {"web_search": 1}))
        logger = logging.getLogger("test")

        first = await tool.execute({}, {}, logger)
        second = await tool.execute({}, {}, logger)
        blocked = await tool.execute({}, {}, logger)

        assert inner.calls == 2
        assert "$1.00 remaining" in first.output
        assert "$0.00 remaining" in second.output
        assert blocked.error is not None
        assert "blocked and not charged" in blocked.output

    asyncio.run(run())


def test_budget_supports_fractional_costs():
    async def run():
        inner = FakeTool()
        tool = BudgetedTool(inner, ToolBudget(1, {"web_search": 0.75}))

        allowed = await tool.execute({}, {}, logging.getLogger("test"))
        blocked = await tool.execute({}, {}, logging.getLogger("test"))

        assert "$0.25 remaining" in allowed.output
        assert "$0.25 remains" in blocked.output
        assert inner.calls == 1

    asyncio.run(run())


def test_concurrent_calls_cannot_overspend():
    async def run():
        inner = FakeTool()
        tool = BudgetedTool(inner, ToolBudget(1, {"web_search": 1}))

        results = await asyncio.gather(
            tool.execute({}, {}, logging.getLogger("test")),
            tool.execute({}, {}, logging.getLogger("test")),
        )

        assert inner.calls == 1
        assert sum(result.error is None for result in results) == 1

    asyncio.run(run())


def test_budget_prompt_is_opt_in():
    assert budget_instructions(None, {"web_search": 1}) == ""
    instructions = budget_instructions(5, {"web_search": 1})
    assert "hard $5.00 budget" in instructions
    assert "`web_search` costs $1.00" in instructions


def test_build_input_preserves_default_and_adds_budget_when_enabled():
    default_system_input = build_input("question")[0]
    parameters = Parameters(
        model_name="test/model",
        tools=["calculator"],
        tool_budget_usd=5,
        llm_config=LLMConfig(),
    )
    budgeted_system_input = build_input("question", parameters)[0]

    assert isinstance(default_system_input, SystemInput)
    assert isinstance(budgeted_system_input, SystemInput)
    assert "hard $5.00 budget" not in default_system_input.text
    assert "hard $5.00 budget" in budgeted_system_input.text
    assert "`calculator` costs $1.00" in budgeted_system_input.text
