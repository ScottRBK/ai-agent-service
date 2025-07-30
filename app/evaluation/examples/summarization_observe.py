import asyncio

from app.core.agents.cli_agent import CLIAgent

from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden
from deepeval import evaluate
from deepeval.models import OllamaModel

summarization_metric = SummarizationMetric(
    assessment_questions=[
        "Is the summary accurate?",
        "Is the summary concise?",
        "Is the summary complete?",
        "Is the summary more than 300 words?",
        "Document events in chronological order"

    ],
    threshold=0.5,
    model=OllamaModel(model="qwen3:8b", temperature=0.0)

)

@observe(metrics=[summarization_metric])
async def test_agent(query: str) -> str:
    agent = CLIAgent(agent_id="summary_agent")
    await agent.initialize()
    response = await agent.chat(query)

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response
        )
    )
    print(f"Response: {response}")
    return response

async def test_with_observe():
    query = """Please summarize this conversation:
    user: What does MCP stand for?
    Assistant: MCP stands for Model Context Protocol. It's a protocol for passing context between agents.
    user: Okay and when was it created?
    Assistant: MCP was created in 2024 around November.
    user: Interesting, i am going to a conference on it in London next week. What is the current time in London?"
    Assistant: The current time in London is 10:00 AM.
    user: Do you know what the weather will be like in London next week?
    Assistant: The weather in London next week is expected to be sunny with a temperature of 20 degrees Celsius.
    user: That's great, i'm going to wear a light jacket.
    Assistant: Good idea, it will be a bit chilly in the mornings and evenings.
    """    
    result = evaluate(
        observed_callback=test_agent,
        goldens=[Golden(input=query)]
    )

if __name__ == "__main__":  
    print("Starting test")
    asyncio.run(test_with_observe())
