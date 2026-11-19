"""Model input in the two FINANCE_AGENT_INPUT modes.

bundled (the default) must stay byte-identical to the pinned prompts. problem_file
sends AGENT_PROMPT and the problem file as written. The FAB service writes that file
as the benchmark rules, a blank line and "Question:\\n<question>"; the rules fixture
is that service text, and model_input.diff pins the exact difference between the
two modes for one question.
"""

import difflib
import hashlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from model_library.base import TextInput
from model_library.base.input import SystemInput

from finance_agent import AGENT_PROMPT, QUESTION_PROMPT, SYSTEM_PROMPT, build_input

FIXTURES = Path(__file__).parent / "fixtures"
QUESTION = "<question>"

# sha256 of the prompts as pinned in the agent-registry FAB agents today.
PINNED_SYSTEM_PROMPT_SHA256 = (
    "905496c683567fc2fc646a261d201397f2c6414e8a8343e3247e284196a621d5"
)
PINNED_QUESTION_PROMPT_SHA256 = (
    "21edc2f9414805f11e631888884031d11c9c8947d824003410231e31144f7f81"
)

TOOL_NAMES = (
    "price_history",
    "submit_final_result",
    "retrieve_information",
    "web_search",
    "edgar_search",
    "parse_html_page",
    "calculator",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pairs(items: list) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, SystemInput):
            pairs.append(("system", item.text))
        elif isinstance(item, TextInput):
            pairs.append(("user", item.text))
        else:
            raise AssertionError(f"unexpected input item {type(item).__name__}")
    return pairs


def _render(items: list) -> str:
    return "".join(f"[{role}]\n{text}\n" for role, text in _pairs(items))


def _problem_file(question: str) -> str:
    return (FIXTURES / "fab_rules.txt").read_text() + "\n\nQuestion:\n" + question


class BundledModeTests(unittest.TestCase):
    def test_pinned_prompts_unchanged(self) -> None:
        self.assertEqual(_sha(SYSTEM_PROMPT), PINNED_SYSTEM_PROMPT_SHA256)
        self.assertEqual(_sha(QUESTION_PROMPT), PINNED_QUESTION_PROMPT_SHA256)

    def test_unset_env_is_todays_input(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FINANCE_AGENT_INPUT", None)
            items = build_input("What was X?")
        self.assertEqual(
            _pairs(items),
            [("system", SYSTEM_PROMPT), ("user", "Question:\nWhat was X?")],
        )

    def test_explicit_bundled_equals_unset(self) -> None:
        os.environ.pop("FINANCE_AGENT_INPUT", None)
        unset = build_input("What was X?")
        with patch.dict(os.environ, {"FINANCE_AGENT_INPUT": "bundled"}):
            explicit = build_input("What was X?")
        self.assertEqual(unset, explicit)

    def test_unknown_mode_fails(self) -> None:
        with patch.dict(os.environ, {"FINANCE_AGENT_INPUT": "problem-file"}):
            with self.assertRaises(ValueError):
                build_input("What was X?")


class ProblemFileModeTests(unittest.TestCase):
    def test_sends_agent_prompt_and_file_as_written(self) -> None:
        text = _problem_file("What was {X}?")
        with patch.dict(os.environ, {"FINANCE_AGENT_INPUT": "problem_file"}):
            items = build_input(text)
        self.assertEqual(_pairs(items), [("system", AGENT_PROMPT), ("user", text)])

    def test_agent_prompt_holds_no_benchmark_rules(self) -> None:
        for line in (FIXTURES / "fab_rules.txt").read_text().splitlines():
            if line.strip():
                self.assertNotIn(line, AGENT_PROMPT)

    def test_rules_name_no_tool(self) -> None:
        rules = (FIXTURES / "fab_rules.txt").read_text()
        for name in TOOL_NAMES:
            self.assertNotIn(name, rules)

    def test_model_input_diff_is_pinned(self) -> None:
        os.environ.pop("FINANCE_AGENT_INPUT", None)
        today = _render(build_input(QUESTION))
        with patch.dict(os.environ, {"FINANCE_AGENT_INPUT": "problem_file"}):
            new = _render(build_input(_problem_file(QUESTION)))
        diff = "".join(
            difflib.unified_diff(
                today.splitlines(True),
                new.splitlines(True),
                "bundled",
                "problem_file",
                n=0,
            )
        )
        self.assertEqual(diff, (FIXTURES / "model_input.diff").read_text())


if __name__ == "__main__":
    unittest.main()
