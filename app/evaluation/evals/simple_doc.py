from deepeval.synthesizer.config import StylingConfig
from deepeval.models import OllamaModel
from uuid import uuid4
from deepeval.metrics import (
    ToolCorrectnessMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    AnswerRelevancyMetric,
    HallucinationMetric
)

from app.evaluation.custom_ollama import CustomOllamaModel
from app.evaluation.config import EvaluationConfig, SynthesizerConfig
from app.evaluation.runner import EvaluationRunner
from app.config.settings import settings
from app.utils.logging import logger
from app.core.agents.cli_agent import CLIAgent

import asyncio
import argparse

agent_id = "knowledge_agent"
synthesis_model = OllamaModel(model="mistral:7b", temperature=0.0, base_url=settings.OLLAMA_BASE_URL)
evaluation_model = CustomOllamaModel(model="qwen3:4b", temperature=0.0, base_url=settings.OLLAMA_BASE_URL)
test = ("this is a super test string i want to edit")

def create_evaluation_config() -> EvaluationConfig:

    """Create evaluation configuration for a rag test being created from documents
    uses the document itself and front matter metadata to create the contexts"""

    styling_config = StylingConfig(
        scenario="User asking queries about information that is in a knowledge base but is not publically available",
        task="answer the users queries based on information contained within the knowledge base",
        input_format="Natural language questions asking for specific information")
    
    synthesizer_config = SynthesizerConfig(
        model=synthesis_model,
        styling_config=styling_config,
        max_goldens_per_context=1
    )

    metrics = [
            ToolCorrectnessMetric(threshold=0.9),        
            FaithfulnessMetric(
                threshold=0.7, 
                model=evaluation_model,
                include_reason=True
            ),
            ContextualRelevancyMetric(
                threshold=0.7, 
                model=evaluation_model,
                include_reason=True
            ),
            ContextualRecallMetric(
                threshold=0.6, 
                model=evaluation_model,
                include_reason=True
            ),
            ContextualPrecisionMetric(
                threshold=0.7,
                model=evaluation_model,
                include_reason=True
            ),
            AnswerRelevancyMetric(
                threshold=0.7, 
                model=evaluation_model
            ),
            HallucinationMetric(
                threshold=0.5, 
                model=evaluation_model
            )
        ]

    
    return EvaluationConfig(
        agent_id=agent_id,
        synthesizer_config=synthesizer_config,
        dataset_name=f"{agent_id}_simple_doc_eval",
        dataset_file=f"{agent_id}_simple_doc_eval.pkl",
        results_file=f"{agent_id}_simple_doc_eval_results.pkl",
        metrics=metrics,
        parse_frontmatter=True,
        document_paths=["knowledge_docs/file_user_guide.md"],
        golden_generation_type="document",
        persist_to_kb=True
    )

async def main(generate_goldens: bool = False, 
               print_verbose: bool =False,
               cleanup: bool = False):
    """Run the document knowledge base simple evaluation"""

    config = create_evaluation_config()

    runner = EvaluationRunner(config)
    await runner.run(generate=generate_goldens, verbose=print_verbose)
    
    if cleanup:
        cleanup_agent = CLIAgent(agent_id)  
        await cleanup_agent.initialize()
        
        if cleanup_agent and cleanup_agent.knowledge_base:
            await cleanup_agent.knowledge_base.delete_all_documents_for_user(runner.generated_user_id)

        if cleanup_agent and cleanup_agent.memory:
            await cleanup_agent.memory.clear_all_sessions_for_user(runner.generated_user_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run knowledge base agent evaluation with Simple RAG metrics. "
                    "When running evaluation (not generating goldens), test documents "
                    "will be automatically created and cleaned up."
    )
    parser.add_argument("--generate", action="store_true", 
                       help="Generate golden test cases")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print out each test case and its results")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Clean up generated documents after evaluation")
    
    args = parser.parse_args()
    
    asyncio.run(main(generate_goldens=args.generate, 
                     print_verbose=args.verbose,
                     cleanup=args.cleanup))