import asyncio

from app.core.agents.cli_agent import CLIAgent

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden
from deepeval import evaluate
from deepeval.models import OllamaModel

correctness_metric = GEval(
    name="Correctness",
    # criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    # if no evaluation steps are provided then deepeval will generate its own using the criteria
    evaluation_steps=[
        "Check whether the facts in 'actual output' is correct",
        "You should also heavily penalize omission of detail",
        "penalize for vague language, or contradicting OPINIONS"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=OllamaModel(model="qwen3:4b", temperature=0.0)

)

@observe(metrics=[correctness_metric])
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
    query = "Why do we have seasons on Earth?"    
    result = evaluate(
        observed_callback=test_agent,
        goldens=[Golden(input=query)]
    )

if __name__ == "__main__":  
    print("Starting test")
    asyncio.run(test_with_observe())
