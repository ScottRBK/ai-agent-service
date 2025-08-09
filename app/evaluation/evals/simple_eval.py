from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import ToolCorrectnessMetric, GEval, HallucinationMetric, AnswerRelevancyMetric, ContextualRelevancyMetric 
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import OllamaModel

from app.evaluation.config import EvaluationConfig, SynthesizerConfig, ContextWithMetadata
from app.evaluation.runner import EvaluationRunner
from app.evaluation.custom_ollama import CustomOllamaModel
from app.config.settings import settings
from datetime import datetime 

import asyncio
import argparse


def create_evaluation_config() -> EvaluationConfig:
    """Create a simple evaluation configuration for an agent"""
    
    # Model for evaluation
    synthesis_model = OllamaModel(model="mistral-small3.2:24b", temperature=0.0, base_url=settings.OLLAMA_BASE_URL)
    evaluation_model = CustomOllamaModel(model="qwen3:30b", temperature=0.0, base_url=settings.OLLAMA_BASE_URL)

    styling_config = StylingConfig(
        scenario="User asking questions that require up to date and relevant information",
        task="""Generate an input query that enquires about who won the premier league in 2025."""
    )

        # Contexts with expected tools
    contexts = [
        ContextWithMetadata(
            context=[f"""User asking about who won the recent premier league title""",],
            retrieval_context=[f"""Liverpool won the premier league title in 2025 - 
                               todays date is {datetime.now().strftime('%Y-%m-%d')}"""],
            tools=["searxng__searxng_web_search", "searxng__web_url_read"],
            expected_output="Liverpool won the premier league title in 2025."
        )
    ]
    
    metrics = [
        ToolCorrectnessMetric(),
        # GEval(
        #     name="recent-information-correctness",
        #     criteria=f"""Does the actual output contain the same factual information as the expected output?""",
        #     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        #     model=evaluation_model
            
        # )
        ContextualRelevancyMetric(
            threshold=0.7,
            model=evaluation_model,
            include_reason=True
        )

    ]

    return EvaluationConfig(
        agent_id="cli_agent",
        synthesizer_config=SynthesizerConfig(
            model=synthesis_model,
            styling_config=styling_config,
            max_goldens_per_context=5
        ),
        metrics=metrics,
        contexts=contexts,
        dataset_name="cli_from_scratch_agent_eval",
        dataset_file="cli_from_scratch_agent_eval.json",
        results_file="cli_from_scratch_agent_eval_results.json"  
    )

async def main(generate_goldens: bool = False, print_verbose: bool = False):
    """Run CLI agent evaluation"""
    
    # Create evaluation configuration
    config = create_evaluation_config()
    
    # Create and run evaluation
    
    runner = EvaluationRunner(config)
    await runner.run(generate=generate_goldens, verbose=print_verbose)

    # await runner.generate_goldens()
    # goldens = runner.dataset.goldens
    # print(goldens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthesizer with tools evaluation")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate golden test cases")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print out each test case and its results")
    
    args = parser.parse_args()
    
    asyncio.run(main(generate_goldens=args.generate, print_verbose=args.verbose))

