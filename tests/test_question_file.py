"""How `finance-agent --question-file` turns the file into model inputs.

problem_file: the file is one whole problem, so each entry point (the batch run
without --output-dir and the single task with it) sends it as one input, as written.
bundled (the default): the batch run sends each nonblank line stripped and the
single task sends the whole file stripped, as before.
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from model_library.base import TextInput
from model_library.base.input import SystemInput

from finance_agent import AGENT_PROMPT, QUESTION_PROMPT, SYSTEM_PROMPT
from finance_agent import run_agent

FIXTURES = Path(__file__).parent / "fixtures"

# Leading and trailing whitespace and a final newline, so stripping would show.
PROBLEM = (FIXTURES / "fab_rules.txt").read_text() + "\n\nQuestion:\n  What was X?\n"


@dataclass
class _StopReason:
    value: str = "done_tool"


@dataclass
class _Result:
    output_dir: Path
    final_answer: str = "answer"
    final_error: Any = None
    success: bool = True
    total_turns: int = 1
    stop_reason: _StopReason = field(default_factory=_StopReason)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"final_answer": self.final_answer}


class _FakeAgent:
    def __init__(self, calls: list[tuple[str, list]], log_dir: Path) -> None:
        self.calls = calls
        self.log_dir = log_dir

    async def run(self, items: list, question_id: str, **_: Any) -> _Result:
        self.calls.append((question_id, items))
        output_dir = self.log_dir / question_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return _Result(output_dir=output_dir)


def _pairs(items: list) -> list[tuple[str, str]]:
    roles = {SystemInput: "system", TextInput: "user"}
    return [(roles[type(item)], item.text) for item in items]


class QuestionFileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.calls: list[tuple[str, list]] = []

    def _run(self, text: str, mode: str | None, *extra: str) -> list[list[tuple[str, str]]]:
        question_file = self.tmp / "problem.txt"
        question_file.write_text(text)
        argv = ["finance-agent", "--question-file", str(question_file), "--log-dir", str(self.tmp / "logs"), *extra]
        with (
            patch.dict(os.environ),
            patch.object(sys, "argv", argv),
            patch.object(run_agent, "load_dotenv"),
            patch.object(
                run_agent, "get_agent", lambda *_, **__: _FakeAgent(self.calls, self.tmp / "logs" / "run")
            ),
        ):
            os.environ.pop("FINANCE_AGENT_INPUT", None)
            if mode is not None:
                os.environ["FINANCE_AGENT_INPUT"] = mode
            run_agent.main_sync()
        return [_pairs(items) for _, items in sorted(self.calls, key=lambda call: call[0])]

    def _single_task(self, text: str, mode: str | None) -> list[list[tuple[str, str]]]:
        inputs = self._run(text, mode, "--output-dir", str(self.tmp / "out"), "--task-id", "t1")
        generation = json.loads((self.tmp / "out" / "t1" / "generation.json").read_text())
        self.assertEqual(generation["status"], "success")
        return inputs


class ProblemFileModeTests(QuestionFileTestCase):
    def test_batch_run_sends_whole_file_as_one_input(self) -> None:
        self.assertEqual(self._run(PROBLEM, "problem_file"), [[("system", AGENT_PROMPT), ("user", PROBLEM)]])

    def test_single_task_sends_whole_file_as_written(self) -> None:
        self.assertEqual(self._single_task(PROBLEM, "problem_file"), [[("system", AGENT_PROMPT), ("user", PROBLEM)]])

    def test_questions_argument_is_unchanged(self) -> None:
        argv = ["finance-agent", "--questions", "a\nb", "c", "--log-dir", str(self.tmp / "logs")]
        with (
            patch.dict(os.environ, {"FINANCE_AGENT_INPUT": "problem_file"}),
            patch.object(sys, "argv", argv),
            patch.object(run_agent, "load_dotenv"),
            patch.object(
                run_agent, "get_agent", lambda *_, **__: _FakeAgent(self.calls, self.tmp / "logs" / "run")
            ),
        ):
            run_agent.main_sync()
        inputs = [_pairs(items) for _, items in sorted(self.calls, key=lambda call: call[0])]
        self.assertEqual(inputs, [[("system", AGENT_PROMPT), ("user", "a\nb")], [("system", AGENT_PROMPT), ("user", "c")]])


class BundledModeTests(QuestionFileTestCase):
    LINES = "  What was X?  \n\n   \nWhat was Y?\n"

    def _bundled(self, question: str) -> list[tuple[str, str]]:
        return [("system", SYSTEM_PROMPT), ("user", QUESTION_PROMPT.format(question=question))]

    def test_batch_run_splits_nonblank_lines(self) -> None:
        for mode in (None, "bundled"):
            with self.subTest(mode=mode):
                self.calls.clear()
                self.assertEqual(
                    self._run(self.LINES, mode), [self._bundled("What was X?"), self._bundled("What was Y?")]
                )

    def test_single_task_sends_whole_file_stripped(self) -> None:
        for mode in (None, "bundled"):
            with self.subTest(mode=mode):
                self.calls.clear()
                self.assertEqual(
                    self._single_task(self.LINES, mode), [self._bundled("What was X?  \n\n   \nWhat was Y?")]
                )


if __name__ == "__main__":
    unittest.main()
