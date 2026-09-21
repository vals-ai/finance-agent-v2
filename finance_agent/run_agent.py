import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from model_library.agent import AgentResult
from model_library.base import LLMConfig
from tqdm.asyncio import tqdm

from .get_agent import Parameters, build_input, get_agent, MAX_TIME_SECONDS
from .tools import VALID_TOOLS

# A time or turn limit is reported through the stop reason, not by raising. Anything
# unrecognised is an error, so an incomplete run is never read as a finished one.
_STOP_REASON_STATUS = {
    "done_tool": "success",
    "max_time": "max_time",
    "max_turns": "max_turns",
}


async def run_single_task(
    question: str,
    task_id: str,
    parameters: Parameters,
    output_dir: Path,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one question and write `<output_dir>/<task_id>/generation.json`.

    For a harness driving one question per process. Failures propagate: the process
    exits nonzero and writes no file, which is how the harness detects them.
    """
    # task_id names a directory under output_dir. An absolute value or one containing
    # a separator would resolve outside it and write the record somewhere unexpected.
    if not task_id or task_id in {".", ".."} or "/" in task_id or "\\" in task_id:
        raise ValueError(f"task_id must be a single path segment, got {task_id!r}")

    agent = get_agent(parameters, log_dir=log_dir)
    result = await agent.run(build_input(question), question_id=task_id, atif_export=True)

    generation = {
        "task_id": task_id,
        "status": _STOP_REASON_STATUS.get(result.stop_reason.value, "error"),
        "data": result.final_answer or "",
        "model": parameters.model_name,
    }

    generation_path = output_dir / task_id / "generation.json"
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    generation_path.write_text(json.dumps(generation, indent=2))
    print(f"{task_id}: {generation['status']} -> {generation_path}")
    return generation


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
                build_input(question),
                question_id=f"q{question_index:03d}",
                atif_export=True,
            )
            return result

    tasks = [process_question(question, i + 1) for i, question in enumerate(questions)]

    results: list[AgentResult] = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            formatted_results.append({"question": question, "success": False, "error": str(result)})
            print(f"\nFAIL Question failed: {question}\n   Error: {result}\n")
        else:
            formatted_results.append(
                {"question": question, "success": result.success, "result": result.model_dump(mode="json")}
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

    return formatted_results


async def main():
    parser = argparse.ArgumentParser(description="Run the harness for the finance agent benchmark")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens for completion generation (default: model-library's)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for model generation (default: model-library's)",
    )
    parser.add_argument("--questions", type=str, nargs="+", help="List of questions to process")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run a single task and write <output-dir>/<task-id>/generation.json. "
        "Requires --task-id, and reads --question-file as one whole question.",
    )
    parser.add_argument("--task-id", type=str, help="Task id naming the generation.json subdirectory")
    args = parser.parse_args()

    env_file = Path(".env")
    load_dotenv(override=True, dotenv_path=env_file)

    single_task = args.output_dir is not None
    if single_task and not args.task_id:
        parser.error("--task-id is required when --output-dir is set")
    if single_task and not args.question_file:
        parser.error("--question-file is required when --output-dir is set")

    if single_task:
        # Read the file whole: a single question may span multiple lines.
        questions = [Path(args.question_file).read_text().strip()]
    elif args.question_file:
        with open(args.question_file) as f:
            questions = [line.strip() for line in f if line.strip()]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception("No questions provided. One of --question-file or --questions must be used.")

    # Omitted values keep the model-library default.
    llm_kwargs = {
        name: value
        for name, value in (("max_tokens", args.max_tokens), ("temperature", args.temperature))
        if value is not None
    }

    parameters = Parameters(
        model_name=args.model,
        max_time_seconds=args.max_time,
        max_turns=args.max_turns,
        tools=args.tools,
        llm_config=LLMConfig(**llm_kwargs),
    )

    if single_task:
        await run_single_task(
            question=questions[0],
            task_id=args.task_id,
            parameters=parameters,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
        )
        return

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
