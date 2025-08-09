from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import ToolCorrectnessMetric, GEval, HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import OllamaModel

from app.evaluation.config import EvaluationConfig, SynthesizerConfig, ContextWithMetadata
from app.evaluation.runner import EvaluationRunner
from app.evaluation.custom_ollama import CustomOllamaModel
from app.config.settings import settings

import asyncio
import argparse


def create_evaluation_config() -> EvaluationConfig:
    """Create evaluation configuration for CLI agent"""
    
    # Model for evaluation
    synthesis_model = OllamaModel(model="qwen3:14b", temperature=0.7, base_url=settings.OLLAMA_BASE_URL)
    evaluation_model = CustomOllamaModel(model="qwen3:30b", temperature=0.0, base_url=settings.OLLAMA_BASE_URL)
    
    # Styling configuration for synthesizer
    styling_config = StylingConfig(
        scenario="User asking questions that require up to date and relevant information",
        task="Generate queries that require up to date information, however do neccessarily request tool usage or that they are up to date",
        input_format="""
        - Search queries: "Can you tell me about x ? Who are the current champions of y, Who is the CEO of x ?"
        """,
        expected_output_format="A helpful response using information from the appropriate tool"
    )
    
    # Contexts with expected tools
    contexts = [
        ContextWithMetadata(
            context=["Response indicating that the recent premier league winners in were Liverpool FC."],
            tools=["searxng__searxng_web_search", "searxng__web_url_read"]
        ),
        ContextWithMetadata(
            context=["Response indicating that the of Kraft Heinz is Carlos Abrams-Reivera"],
            tools=["searxng__searxng_web_search", "searxng__web_url_read"]
        ),
        ContextWithMetadata(
            context=["Response indicating that the LIGO-Virgo-Kagra Collaboration has detected the merger of the most massive black holes ever observed in 2023"],
            tools=["searxng__searxng_web_search", "searxng__web_url_read"]
        )
    ]
    
    # Metrics for evaluation
    metrics = [
        ToolCorrectnessMetric(),
        GEval(
            name="coherence",
            criteria="Is the response coherent and well-structured?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=evaluation_model
        ),
        GEval(
            name="correctness",
            criteria="Determine if the output is factually correct based on the expected output",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=evaluation_model
        ),
        HallucinationMetric(threshold=0.5, model=evaluation_model),
        AnswerRelevancyMetric(threshold=0.7, model=evaluation_model)
    ]
    
    # Create complete configuration
    return EvaluationConfig(
        agent_id="cli_agent",
        synthesizer_config=SynthesizerConfig(
            model=synthesis_model,
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