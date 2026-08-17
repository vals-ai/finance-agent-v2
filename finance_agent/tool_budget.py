import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from model_library.agent import Tool, ToolOutput


def _money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.01'))}"


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    cost: Decimal
    spent: Decimal
    remaining: Decimal


class ToolBudget:
    """Concurrency-safe, per-agent budget for tool calls."""

    def __init__(self, total_usd: float, costs_usd: dict[str, float]):
        self.total = Decimal(str(total_usd))
        self.costs = {name: Decimal(str(cost)) for name, cost in costs_usd.items()}
        self.spent = Decimal("0")
        self._lock = asyncio.Lock()

        if not self.total.is_finite() or self.total < 0:
            raise ValueError("Tool budget must be a finite non-negative number")
        invalid_costs = {
            name: cost
            for name, cost in self.costs.items()
            if not cost.is_finite() or cost < 0
        }
        if invalid_costs:
            raise ValueError(f"Tool costs must be non-negative: {invalid_costs}")

    async def authorize(self, tool_name: str) -> BudgetDecision:
        cost = self.costs[tool_name]
        async with self._lock:
            if self.spent + cost > self.total:
                return BudgetDecision(False, cost, self.spent, self.total - self.spent)
            self.spent += cost
            return BudgetDecision(True, cost, self.spent, self.total - self.spent)


class BudgetedTool(Tool):
    """Wrap a tool and enforce a hard budget before executing it."""

    # Tool validates these attributes when a subclass is defined. Each wrapper
    # replaces them with the wrapped tool's definition during initialization.
    name = "budgeted_tool"
    description = "A tool with a hard call budget."
    parameters: dict[str, Any] = {}
    required: list[str] = []

    def __init__(self, tool: Tool, budget: ToolBudget):
        self._tool = tool
        self._budget = budget
        self.name = tool.name
        self.description = tool.description
        self.parameters = tool.parameters
        self.required = tool.required

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        decision = await self._budget.authorize(self.name)
        if not decision.allowed:
            message = (
                f"TOOL BUDGET EXHAUSTED: {self.name} costs {_money(decision.cost)}, but only "
                f"{_money(decision.remaining)} remains. This call was blocked and not charged. "
                "Use the information already gathered and call submit_final_result."
            )
            logger.warning(message)
            return ToolOutput(output=message, error=message)

        logger.info(
            "Tool budget: charged %s for %s; %s spent; %s remaining",
            _money(decision.cost),
            self.name,
            _money(decision.spent),
            _money(decision.remaining),
        )
        result = await self._tool.execute(args, state, logger)
        result.output = (
            f"[Tool budget: charged {_money(decision.cost)}; "
            f"{_money(decision.remaining)} remaining.]\n{result.output}"
        )
        return result


def budget_instructions(total_usd: float | None, costs_usd: dict[str, float]) -> str:
    if total_usd is None:
        return ""

    costs = ", ".join(
        f"`{name}` costs ${cost:.2f}" for name, cost in sorted(costs_usd.items())
    )
    return f"""
You have a hard ${total_usd:.2f} budget for tool calls. {costs}.
Plan your research and prioritize high-value calls. The harness enforces this budget: an accepted call is charged
even if the tool later fails, and a call that would exceed the remaining budget is blocked without charge.
The `submit_final_result` tool is free and always available. Each tool result reports your remaining budget.
""".strip()
