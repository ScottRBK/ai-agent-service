import asyncio
import uuid
from deepeval.test_case import Turn
from deepeval.simulator import ConversationSimulator 
from deepeval.dataset import ConversationalGolden
from deepeval.models import OllamaModel 
from deepeval.test_case.conversational_test_case import ConversationalTestCase

from app.core.agents.cli_agent import CLIAgent
from app.config.settings import settings

simulator_model = OllamaModel(model="mistral-small3.2:24b", temperature=0.0, base_url=settings.OLLAMA_BASE_URL)

conversation_golden = ConversationalGolden(
    scenario="""User starts by asking about Sunderlands Transfers
    After the agent responsed the user then asks about a specific player
    After the agent responsed the user then asks about rival teams transfers
    and how they compare to their teams
    After the agents response the user then asks what the agent thinks their teams chances
    are to avoid relegation in the upcoming season
    YOU MUST NOT ASK ALL THIS IN ONE TURN IT IS A MULTI-TURN CONVERSATION""",
    expected_outcome="""The conversation must include multiple turns:
    1. Users asks about Sunderland Transfers -> Agent provides transfer news
    2. Users asks follow up about specific players -> Agent provides details
    3. User asks about rival teams transfers and how they comare -> Agent provides details
    4. User asks what the agent thinks about relegation chances -> Agent provides its view""",
    user_description="""Working class individual from the north east of england 
    with a passion for football"""
)

agent = CLIAgent(agent_id="cli_agent")
agent.user_id = f"test_user_{uuid.uuid4()}"
agent.session_id = f"test_session_{uuid.uuid4()}"

async def agent_callback(input):
    return Turn(role="assistant", content=await agent.chat(input))

async def main():
    await agent.initialize()
    simulator = ConversationSimulator(
        model_callback=agent_callback,
        simulator_model=simulator_model
    )

    conversational_test_cases = simulator.simulate(conversational_goldens=[conversation_golden],
                                                   max_turns=10)
    for test_case in conversational_test_cases:
        for turn in test_case.turns:
            print(f"\n{turn.role}: {turn.content}")

if __name__ == "__main__":
    asyncio.run(main())
