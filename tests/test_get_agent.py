import unittest
from unittest.mock import MagicMock, patch

from model_library.base import LLM, LLMConfig

from finance_agent.get_agent import Parameters, get_agent
from finance_agent.tools import RetrieveInformation


def registry_model(model_name: str, config: LLMConfig) -> LLM:
    llm = MagicMock(spec=LLM)
    llm.model_name = model_name
    llm.prompt_caching = config.prompt_caching
    return llm


class GetAgentTest(unittest.TestCase):
    def test_retrieve_information_llm_skips_prompt_caching(self) -> None:
        parameters = Parameters(
            model_name="anthropic/model",
            tools=["retrieve_information"],
            llm_config=LLMConfig(),
        )
        with patch("finance_agent.get_agent.get_registry_model", registry_model):
            agent = get_agent(parameters)

        tool = agent._tools["retrieve_information"]
        assert isinstance(tool, RetrieveInformation)
        self.assertEqual(agent._llm.prompt_caching, "5m")
        self.assertEqual(tool._llm.prompt_caching, "off")
