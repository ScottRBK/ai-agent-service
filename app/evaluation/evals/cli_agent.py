from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToolCorrectnessMetric, GEval, HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import ToolCall, LLMTestCaseParams
from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.dataset import EvaluationDataset, Golden

from app.core.agents.cli_agent import CLIAgent
from app.utils.chat_utils import clean_response_for_memory
from app.evaluation.evaluation_utils import EvaluationUtils as eutils
from app.evaluation.dataset import GoldenDataset
from app.evaluation.runner import EvaluationRunner

from typing import Dict, Any, List
from contextlib import redirect_stdout, redirect_stderr

import asyncio
import argparse
import pandas as pd
import json
import io


dataset_file = "synthesizer_goldens_with_multiple_evals.pkl"

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
        "context": ["The DeepWiki page for ScottRBK/ai-agent-service shows it's a production-ready AI agent framework with MCP (Model Context Protocol) integration."],
        "tools": ["deepwiki__read_wiki_contents"]
    }
]

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

def create_metrics():
    """Create all metrics in one place"""
    return [
        ToolCorrectnessMetric(),
        GEval(
            name="coherence",
            criteria="Is the response coherent and well-structured?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model
        ),
        HallucinationMetric(threshold=0.5, model=model),
        AnswerRelevancyMetric(threshold=0.7, model=model)
    ]


async def main(generate_goldens=False, print_verbose=False):

    dataset = GoldenDataset("cli_agent")

    synthesizer_config = {
        "model": model,
        "styling_config": styling_config
    }

    if generate_goldens:
        await dataset.generate_from_contexts(contexts_with_metadata, synthesizer_config, 2)
        dataset.save(dataset_file)
        print(f"Saved {len(dataset.goldens)} goldens to file {dataset_file}")
    else:
        dataset.load(dataset_file)
        print(f"Loaded {len(dataset.goldens)} goldens from file {dataset_file}")

    metrics = create_metrics()

    runner = EvaluationRunner()

    results = await runner.run_evaluation(
        agent_id="cli_agent",
        dataset=dataset,
        metrics=metrics
    )

    df = results["dataframe"]
    if print_verbose:
        eutils.print_evaluation_summary_verbose(df)
    else:
        eutils.print_evaluation_summary(df, results["raw_results"])

    df.to_pickle("synthesizer_with_multiple_evals_results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthesizer with tools evaluation")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate golden test cases")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print out each test case and its results")
    
    args = parser.parse_args()
    
    asyncio.run(main(generate_goldens=args.generate, print_verbose=args.verbose))