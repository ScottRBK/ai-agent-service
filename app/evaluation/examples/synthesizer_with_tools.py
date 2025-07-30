from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.dataset import EvaluationDataset, Golden
from app.core.agents.cli_agent import CLIAgent
from app.utils.chat_utils import clean_response_for_memory

import asyncio
import argparse
from typing import Dict, Any, List
import pandas as pd

dataset_file = "synthesizer_goldens_with_tools.pkl"

model = OllamaModel(model="mistral:7b", temperature=0.0)
# 1. Create contexts that represent tool responses
contexts_with_metadata = [
    {
        "context": ["Search results show that OpenAI released GPT-4 in March 2023, with significant improvements in reasoning and reduced hallucinations."],
        "tools": ["searxng__searxng_web_search"],
    },
    {
        "context": ["GitHub search found 15 repositories related to MCP implementation, with fastmcp being the most popular Python library."],
        "tools": ["github__search_repositories"],
    },
    {
        "context": ["The DeepWiki page for ScottRBK/ai-agent-service shows it's a production-ready AI agent framework with MCP integration."],
        "tools": ["deepwiki__read_wiki_contents"]
    }
]

# 2. Configure synthesizer to generate tool-specific queries
styling_config = StylingConfig(
    scenario="User asking questions that require specific tools",
    task="Generate queries that clearly indicate which tool to use",
    input_format="""
    - Search queries: "What's the latest news about X?", "Find information about Y", "Search for Z"
    - GitHub queries: "Show me repositories about X", "Find GitHub projects for Y", "List repos related to Z"
    - Deepwiki queries: "What does the wiki say about X?", "Tell me about the Y project on DeepWiki", "Read the DeepWiki page for Z"
    """,
    expected_output_format="A helpful response using information from the appropriate tool"
)

async def generate_tool_goldens() -> list[Golden]:
    synthesizer = Synthesizer(model=model, styling_config=styling_config)
    goldens_to_return = []
    for context in contexts_with_metadata:
        print(f"\nGenerating goldens for tool: {context['tools']}")

        goldens = await synthesizer.a_generate_goldens_from_contexts(
            contexts=[context["context"]],
            include_expected_output=True,
            max_goldens_per_context=2
        )

        for i, base_golden in enumerate(goldens):
            print(f"  Golden {i+1}: {base_golden.input[:50]}...")
            
            exp_tools = []

            for tool in context["tools"]:
                exp_tools.append(ToolCall(name=tool))
                
            base_golden.expected_tools = exp_tools

        goldens_to_return.extend(goldens)
        print(f"\nTotal goldens generated: {len(goldens_to_return)}")
    return goldens_to_return

async def get_tool_goldens(generate_goldens: bool) -> EvaluationDataset:
    dataset = EvaluationDataset()
    if generate_goldens:
        goldens = await generate_tool_goldens()
        dataset.goldens = goldens
        await save_dataset_with_tools(dataset, dataset_file)
    else:
        goldens = await load_dataset_with_tools(dataset_file)
        dataset.goldens = goldens
    return dataset

async def save_dataset_with_tools(dataset: EvaluationDataset, filename: str):
    """Save dataset using pandas pickle - preserves everything perfectly"""
    data = [{
        'input': g.input,
        'expected_output': g.expected_output,
        'expected_tools': g.expected_tools if hasattr(g, 'expected_tools') else []
    } for g in dataset.goldens]
    
    df = pd.DataFrame(data)
    df.to_pickle(filename)
    print(f"Saved {len(df)} goldens to {filename}")

async def load_dataset_with_tools(filename: str) -> List[Golden]:
    """Load dataset using pandas pickle"""
    df = pd.read_pickle(filename)
    
    goldens = []
    for _, row in df.iterrows():
        golden = Golden(
            input=row['input'],
            expected_output=row['expected_output']
        )
        golden.expected_tools = row['expected_tools']
        goldens.append(golden)
    print(f"Loaded {len(goldens)} goldens from file {filename}")
    return goldens

async def main(generate_goldens=False):

    dataset = await get_tool_goldens(generate_goldens)
    print(f"Loaded {len(dataset.goldens)} goldens")

    test_cases = []
    
    for i, golden in enumerate(dataset.goldens):
        print(f"Processing golden {i+1} of {len(dataset.goldens)}")

        agent = CLIAgent("cli_agent")
        await agent.initialize()
        agent.provider.config.track_tool_calls = True
        response = await agent.chat(golden.input)

        tools_called = [ToolCall(name=tool["tool_name"]) for tool in agent.provider.get_tool_calls_made()]
        
        test_case = LLMTestCase(
            input=golden.input,
            expected_tools=golden.expected_tools,
            actual_output=clean_response_for_memory(response),
            tools_called=tools_called
        )
        test_cases.append(test_case)


    metric = ToolCorrectnessMetric()
    results = evaluate(
        test_cases=test_cases,
        metrics=[metric]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthesizer with tools evaluation")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate golden test cases")
    args = parser.parse_args()
    
    asyncio.run(main(generate_goldens=args.generate))