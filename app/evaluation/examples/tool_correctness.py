import asyncio

from pprint import pprint
from app.core.agents.cli_agent import CLIAgent


from deepeval.metrics import ToolCorrectnessMetric
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
from deepeval.models import OllamaModel

ollama_model = OllamaModel(model="qwen3:4b", temperature=0.0)

tool_correctness_metric = ToolCorrectnessMetric(
    threshold=0.5,
    include_reason=True
)

async def test_agent():

    agent = CLIAgent(agent_id="cli_agent")
    await agent.initialize()
    agent.provider.config.track_tool_calls = True
        
    response = await agent.chat("What is the current time in London?")
    tools_called = [ToolCall(name=tool["tool_name"], arguments=tool["arguments"]) for tool in agent.provider.get_tool_calls_made()]

    print(f"Tools called: {tools_called}")

    test_case = LLMTestCase(
        input="What is the current time in London?",
        actual_output=response,
        tools_called=tools_called,
        expected_tools=[ToolCall(name="get_current_datetime")]
    )

    evaluate(
        test_cases=[test_case],
        metrics=[tool_correctness_metric]
    )

if __name__ == "__main__":
    asyncio.run(test_agent())
