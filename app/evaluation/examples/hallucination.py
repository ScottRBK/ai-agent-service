import asyncio

from app.core.agents.cli_agent import CLIAgent

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import OllamaModel

hallucination_metric = HallucinationMetric(
    threshold=0.5,
    model=OllamaModel(model="qwen3:4b", temperature=0.0)
)

async def test_agent() -> str:

    context=["A man with blond-hair, and a brown shirt drinking out of a public water fountain."]

    agent = CLIAgent(agent_id="cli_agent")
    await agent.initialize()
    await agent.chat(str(context[0]))
    
    query = """what was the blonde doing?"""
    response = await agent.chat(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        context=context
    )
    
    evaluate(
        test_cases=[test_case],
        metrics=[hallucination_metric]
    )

if __name__ == "__main__":  
    print("Starting test")
    asyncio.run(test_agent())
