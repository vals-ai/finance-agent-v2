"""Run with the paired native-capture SDK, not the legacy image dependency pin."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from model_library.agent import Agent, AgentConfig, AgentHooks
from model_library.base import LLM, RawResponse, TextInput
from model_library.base.input import ToolCall, ToolResult
from model_library.base.output import QueryResult, QueryResultMetadata

from finance_agent.tools import RetrieveInformation


class RetrieveInformationCaptureTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_tool_preserves_helper_evidence_before_sdk_hooks(self) -> None:
        call = ToolCall(
            id="retrieve-1",
            name="retrieve_information",
            args={"prompt": "Extract revenue: {{filing}}"},
        )
        helper = QueryResult(
            output_text="Revenue was $42 million.",
            reasoning="Read the revenue line.",
            metadata=QueryResultMetadata(in_tokens=7, out_tokens=3),
            history=[
                TextInput(text="Extract revenue: Revenue: $42 million."),
                RawResponse(response={"provider_payload": "opaque helper response"}),
            ],
        )
        main = QueryResult(
            output_text="Read the filing.",
            metadata=QueryResultMetadata(in_tokens=11, out_tokens=5),
            tool_calls=[call],
            history=[TextInput(text="Find revenue."), RawResponse(response={"id": "main"})],
        )
        final = QueryResult(
            output_text="The revenue is $42 million.",
            metadata=QueryResultMetadata(in_tokens=2, out_tokens=1),
            history=[TextInput(text="Find revenue.")],
        )
        # Only the external provider boundary is synthetic. Tool and SDK are real.
        llm = MagicMock(spec=LLM)
        llm.model_name = "synthetic-finance"
        llm.max_tokens = None
        llm.query = AsyncMock(side_effect=[main, helper, final])
        hook_evidence = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def on_tool_result(record, state):
                helper_dir = next(root.rglob("helper_queries/tool_000"))
                evidence = json.loads((helper_dir / "result.json").read_text())
                history = (helper_dir / "history.json").read_bytes()
                self.assertEqual(evidence["query_result"]["output_text"], helper.output_text)
                self.assertEqual(record.tool_output.output, helper.output_text)
                self.assertEqual(record.tool_output.metadata, helper.metadata)
                self.assertIsNone(record.tool_output.native_query_result)
                self.assertNotIn("history", evidence["query_result"])
                hook_evidence.append((helper_dir, history))
                record.tool_output.output = "hook changed the live record"
                record.tool_call.id = "hook-changed-id"
                helper.output_text = "hook changed the response"
                helper.history.clear()

            agent = Agent(
                llm=llm,
                tools=[RetrieveInformation(llm)],
                name="finance",
                log_dir=root,
                config=AgentConfig(turn_limit=None, time_limit=None),
                hooks=AgentHooks(on_tool_result=on_tool_result),
            )
            result = await agent.run(
                [TextInput(text="Find revenue.")],
                question_id="capture-test",
                state={"filing": "Revenue: $42 million."},
            )

            self.assertIsNone(result.final_error)
            self.assertEqual(len(hook_evidence), 1)
            helper_dir, original_history = hook_evidence[0]
            self.assertEqual((helper_dir / "history.json").read_bytes(), original_history)
            history = json.loads(original_history)
            self.assertEqual(history[0]["text"], "Extract revenue: Revenue: $42 million.")
            self.assertEqual(history[1]["kind"], "raw_response")
            self.assertIn("response", history[1])  # Retain opaque data; never deserialize it.
            native = json.loads((helper_dir.parent.parent / "result.json").read_text())
            self.assertEqual(native["query_result"]["output_text"], "Read the filing.")
            self.assertEqual(native["tool_call_records"][0]["tool_call"]["id"], "retrieve-1")
            self.assertEqual(
                native["tool_call_records"][0]["tool_output"]["output"],
                "Revenue was $42 million.",
            )
            saved_helper = json.loads((helper_dir / "result.json").read_text())
            self.assertEqual(saved_helper["query_result"]["output_text"], "Revenue was $42 million.")
            self.assertEqual(saved_helper["query_result"]["reasoning"], "Read the revenue line.")
            self.assertEqual(llm.query.call_args_list[1].args, ("Extract revenue: Revenue: $42 million.",))
            next_history = llm.query.call_args_list[2].kwargs["input"]
            tool_results = [item.result for item in next_history if isinstance(item, ToolResult)]
            self.assertEqual(tool_results, ["Revenue was $42 million."])
            # The SDK's existing aggregate covers main turns only; helper
            # accounting remains on the tool output, not added to that total.
            self.assertEqual(result.final_aggregated_metadata.in_tokens, 13)
            self.assertEqual(result.final_aggregated_metadata.out_tokens, 6)

    async def test_provider_failure_keeps_existing_model_visible_error(self) -> None:
        llm = MagicMock(spec=LLM)
        llm.query = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        output = await RetrieveInformation(llm).execute(
            {"prompt": "Extract: {{filing}}"},
            {"filing": "Revenue: $42 million."},
            logging.getLogger(__name__),
        )
        self.assertEqual(output.output, "provider unavailable")
        self.assertEqual(output.error, "provider unavailable")
        self.assertIsNone(output.metadata)
        self.assertIsNone(output.native_query_result)


if __name__ == "__main__":
    unittest.main()
