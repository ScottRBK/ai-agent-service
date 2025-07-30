from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import ToolCorrectnessMetric, GEval, HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import OllamaModel

from app.evaluation.config import EvaluationConfig, SynthesizerConfig, ContextWithMetadata
from app.evaluation.runner import EvaluationRunner

import asyncio
import argparse


def create_evaluation_config() -> EvaluationConfig:
    """Create evaluation configuration for CLI agent"""
    
    # Model for evaluation
    model = OllamaModel(model="mistral:7b", temperature=0.0)
    
    # Styling configuration for synthesizer
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
    
    # Contexts with expected tools
    contexts = [
        ContextWithMetadata(
            context=["Search results show that OpenAI released GPT-4 in March 2023, with significant improvements in reasoning and reduced hallucinations."],
            tools=["searxng__searxng_web_search"]
        ),
        ContextWithMetadata(
            context=["GitHub search found 15 repositories related to MCP implementation, with fastmcp being the most popular Python library."],
            tools=["github__search_repositories"]
        ),
        ContextWithMetadata(
            context=["The DeepWiki page for ScottRBK/ai-agent-service shows it's a production-ready AI agent framework with MCP (Model Context Protocol) integration."],
            tools=["deepwiki__read_wiki_contents"]
        )
    ]
    
    # Metrics for evaluation
    metrics = [
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
    
    # Create complete configuration
    return EvaluationConfig(
        agent_id="cli_agent",
        synthesizer_config=SynthesizerConfig(
            model=model,
            styling_config=styling_config,
            max_goldens_per_context=2
        ),
        metrics=metrics,
        contexts=contexts,
        dataset_name="cli_agent",
        dataset_file="cli_agent_goldens.pkl",
        results_file="cli_agent_results"
    )


async def main(generate_goldens: bool = False, print_verbose: bool = False):
    """Run CLI agent evaluation"""
    
    # Create evaluation configuration
    config = create_evaluation_config()
    
    # Create and run evaluation
    runner = EvaluationRunner(config)
    await runner.run(generate=generate_goldens, verbose=print_verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthesizer with tools evaluation")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate golden test cases")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print out each test case and its results")
    
    args = parser.parse_args()
    
    asyncio.run(main(generate_goldens=args.generate, print_verbose=args.verbose))