import asyncio

from app.core.agents.cli_agent import CLIAgent

from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import OllamaModel

ollama_model = OllamaModel(model="qwen3:4b", temperature=0.0)
task_completion = TaskCompletionMetric(
    model = ollama_model,
    threshold=0.5,
    include_reason=True
)

@observe(metrics=[task_completion])
async def test_agent(query: str) -> str:
    agent = CLIAgent(agent_id="cli_agent")    

    await agent.initialize()

    response = await agent.chat(query)

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response
        )
    )
    return response

async def test_with_observe():
    query = "What is the current time in London?"
    expected_output = f"used the tool get_current_datetime to get the current time in London."
    
    result = evaluate(
        observed_callback=test_agent,
        goldens=[Golden(input=query, expected_output=expected_output)]
    )


if __name__ == "__main__":
    print("Starting test")
    # Run the async test
    asyncio.run(test_with_observe())