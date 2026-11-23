import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from model_library.base import LLM, LLMConfig

from finance_agent.get_agent import Parameters
from finance_agent.run_agent import run_single_task


class RunSingleTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_during_query_saves_error_and_generation(self) -> None:
        async def wait_for_query(**_kwargs: object) -> None:
            await asyncio.Future()

        parameters = Parameters(
            model_name="synthetic-finance",
            max_time_seconds=1,
            tools=[],
            llm_config=LLMConfig(),
        )
        llm = MagicMock(
            spec=LLM,
            model_name=parameters.model_name,
            query=AsyncMock(side_effect=wait_for_query),
            **parameters.llm_config.model_dump(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("finance_agent.get_agent.create_llm", return_value=llm):
                await run_single_task(
                    "Find revenue.", "test-task", parameters, root / "output", root / "logs"
                )

            llm.query.assert_awaited_once()
            generation = json.loads((root / "output/test-task/generation.json").read_text())
            self.assertEqual(generation["status"], "max_time")
            result = json.loads(next((root / "logs").rglob("test-task/result.json")).read_text())
            self.assertEqual(result["final_error"]["type"], "CancelledError")
