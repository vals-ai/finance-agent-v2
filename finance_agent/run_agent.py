import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from model_library.agent import AgentResult
from model_library.base import LLMConfig
from tqdm.asyncio import tqdm

from .get_agent import (
    MAX_TIME_SECONDS,
    Parameters,
    build_input,
    get_agent,
    resolved_tool_costs,
)
from .tools import VALID_TOOLS


def parse_tool_costs(values: list[str]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for value in values:
        try:
            name, raw_cost = value.split("=", 1)
            cost = float(raw_cost)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid tool cost '{value}'; expected TOOL=USD"
            ) from exc
        if name not in VALID_TOOLS:
            raise argparse.ArgumentTypeError(
                f"Unknown tool '{name}'; choose from {VALID_TOOLS}"
            )
        if not math.isfinite(cost) or cost < 0:
            raise argparse.ArgumentTypeError(
                f"Tool cost must be a finite non-negative number: {value}"
            )
        costs[name] = cost
    return costs


async def run_tests_parallel(
    questions: list[str],
    max_concurrent: int,
    parameters: Parameters,
    log_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run multiple questions in parallel using the agent"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question: str, question_index: int):
        async with semaphore:
            agent = get_agent(parameters, log_dir=log_dir)
            result = await agent.run(
                build_input(question, parameters),
                question_id=f"q{question_index:03d}",
                atif_export=True,
            )
            return result

    tasks = [process_question(question, i + 1) for i, question in enumerate(questions)]

    results: list[AgentResult] = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            formatted_results.append(
                {"question": question, "success": False, "error": str(result)}
            )
            print(f"\nFAIL Question failed: {question}\n   Error: {result}\n")
        else:
            formatted_results.append(
                {
                    "question": question,
                    "success": result.success,
                    "result": result.model_dump(mode="json"),
                }
            )
            if not result.success and result.final_error:
                print(
                    f"\nFAIL Question failed: {question}\n   Turns: {result.total_turns}\n   Error: [{result.final_error.type}] {result.final_error.message}\n"
                )
            else:
                print(
                    f"\nOK Question succeeded: {question}\n   Turns: {result.total_turns}\n   Result: {result.final_answer}\n"
                )

    # Write results next to agent logs (use first result's output_dir parent)
    non_error_results = [r for r in results if not isinstance(r, Exception)]
    if non_error_results:
        results_dir = non_error_results[0].output_dir.parent
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(formatted_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        if parameters.tool_budget_usd is not None:
            budget_file = results_dir / "tool_budget.json"
            with open(budget_file, "w") as f:
                json.dump(
                    {
                        "budget_usd_per_question": parameters.tool_budget_usd,
                        "costs_usd": resolved_tool_costs(parameters),
                        "charging_policy": "accepted calls are charged; blocked calls are not; submit is free",
                    },
                    f,
                    indent=2,
                )
            print(f"Tool-budget configuration saved to: {budget_file}")

    return formatted_results


async def main():
    parser = argparse.ArgumentParser(
        description="Run the harness for the finance agent benchmark"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum number of tokens for completion generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model generation",
    )
    parser.add_argument(
        "--questions", type=str, nargs="+", help="List of questions to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
        help="Model to use to generate completions",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="Path to file containing questions (one per line)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=VALID_TOOLS,
        choices=VALID_TOOLS,
        help="List of tools to make available to the agent",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=MAX_TIME_SECONDS,
        help="Maximum time in seconds for the agent to run before stopping (default: 2 hours)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of agent turns (default: unlimited, time limit only)",
    )
    parser.add_argument(
        "--tool-budget-usd",
        type=float,
        default=None,
        help="Hard per-question budget for tools (disabled by default)",
    )
    parser.add_argument(
        "--tool-cost",
        action="append",
        default=[],
        metavar="TOOL=USD",
        help="Override a tool's cost; repeat as needed (default: $1 per enabled tool)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel requests to make to the model",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory where per-question agent logs are written",
    )
    args = parser.parse_args()

    env_file = Path(".env")
    load_dotenv(override=True, dotenv_path=env_file)

    if args.question_file:
        with open(args.question_file) as f:
            questions = [line.strip() for line in f if line.strip()]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception(
            "No questions provided. One of --question-file or --questions must be used."
        )

    try:
        tool_costs_usd = parse_tool_costs(args.tool_cost)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    parameters = Parameters(
        model_name=args.model,
        max_time_seconds=args.max_time,
        max_turns=args.max_turns,
        tools=args.tools,
        tool_budget_usd=args.tool_budget_usd,
        tool_costs_usd=tool_costs_usd,
        llm_config=LLMConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )

    await run_tests_parallel(
        questions=questions,
        max_concurrent=args.parallelism,
        parameters=parameters,
        log_dir=args.log_dir,
    )


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
