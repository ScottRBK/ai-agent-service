from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import (
    ToolCorrectnessMetric, 
    GEval, 
    HallucinationMetric, 
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import OllamaModel

from app.evaluation.config import EvaluationConfig, SynthesizerConfig, ContextWithMetadata
from app.evaluation.runner import EvaluationRunner
from app.evaluation.custom_ollama import CustomOllamaModel
from app.core.agents.cli_agent import CLIAgent
from app.models.resources.knowledge_base import DocumentType
from app.utils.logging import logger
from app.config.settings import settings

import asyncio
import argparse
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any


@asynccontextmanager
async def knowledge_agent_test_context(test_users: List[str] = None):
    """
    Context manager for setting up and tearing down test data for knowledge agent evaluation.
    
    Sets up test documents in the knowledge base that match the evaluation contexts,
    then cleans them up after tests complete.
    
    Args:
        test_users: List of user IDs to set up test data for. If None, creates a default user.
    """
    # Track created documents for cleanup
    created_document_ids: List[str] = []
    test_agents = {}
    
    # Use provided user IDs or create a default one
    if not test_users:
        test_users = [f"test_user_{uuid.uuid4()}"]
    
    try:
        # Create test agents for each user
        for test_user_id in test_users:
            test_session_id = f"test_session_{uuid.uuid4()}"
            test_agent = CLIAgent(
                agent_id="knowledge_agent",
                user_id=test_user_id,
                session_id=test_session_id
            )
            await test_agent.initialize()
            test_agents[test_user_id] = test_agent
        
        # Ensure knowledge base is available for all agents
        for user_id, agent in test_agents.items():
            if not agent.knowledge_base:
                raise RuntimeError(f"Knowledge agent for user {user_id} does not have knowledge base configured")
        
        logger.info(f"Setting up test data for knowledge agent evaluation (users: {test_users})")
        
        # Get the embedding model for logging purposes (should be same for all)
        embedding_model = list(test_agents.values())[0].knowledge_base.embedding_model
        
        # Set up test data for each user
        for test_user_id, test_agent in test_agents.items():
            # Use base namespaces without embedding model - the knowledge base will append it
            base_conversations_ns = f"conversations:{test_user_id}"
            base_documents_ns = f"documents:{test_user_id}"
            
            logger.info(f"Creating test data for user {test_user_id}:")
            logger.info(f"  Conversations base: {base_conversations_ns}")
            logger.info(f"  Documents base: {base_documents_ns}")
            logger.info(f"  Embedding model (to be appended): {embedding_model}")
        
            # Define test documents with completely fictional content to avoid training data contamination
            test_documents = [
                {
                    "title": "Quixel Token Protocol Implementation Meeting",
                    "content": (
                        "Team Phoenix meeting from 2024-11-15: Discussed Quixel Token Protocol (QTP-3) implementation for Project Stellaris. "
                        "Key decisions: Use QTP tokens with 45-minute expiration, implement cascade refresh mechanism, "
                        "store sessions in MemoryVault for scalability. Security lead Zara Chen proposed token scrambling algorithm with entropy seeds."
                    ),
                    "namespace": base_conversations_ns,  # Base namespace - KB will append embedding model
                    "source": "qtp_implementation_meeting_2024-11-15",
                    "metadata": {"topic": "authentication", "date": "2024-11-15", "team": "phoenix", "project": "stellaris"}
                },
                {
                    "title": "NebulaSoft Architecture - Service Discovery",
                    "content": "Team Aurora conversation: Discussed service discovery using MeshLink Protocol and NodeTracker v2.1 for Project Stellaris",
                    "namespace": base_conversations_ns,  # Base namespace - KB will append embedding model
                    "source": "nebula_service_discovery",
                    "metadata": {"topic": "nebula_architecture", "subtopic": "service_discovery", "project": "stellaris"}
                },
                {
                    "title": "NebulaSoft Architecture - Gateway Routing",
                    "content": "Team meeting notes: Gateway routing patterns with GatewayPrime and RouteForge for handling 50K requests per second",
                    "namespace": base_conversations_ns,  # Base namespace - KB will append embedding model
                    "source": "nebula_gateway_routing",
                    "metadata": {"topic": "nebula_architecture", "subtopic": "gateway_routing", "project": "stellaris"}
                },
                {
                    "title": "NebulaSoft Architecture - ContainerFlux Deployment",
                    "content": "Technical discussion: ContainerFlux deployment using Orbital Charts and FluxOps methodology with auto-scaling to 200 pods",
                    "namespace": base_conversations_ns,  # Base namespace - KB will append embedding model
                    "source": "nebula_containerflux",
                    "metadata": {"topic": "nebula_architecture", "subtopic": "containerflux", "project": "stellaris"}
                },
                {
                    "title": "DataHaven Optimization Discussion",
                    "content": (
                        "DataHaven Optimization Meeting with Team Phoenix: Created quantum indexes on (nebula_id, flux_timestamp), "
                        "optimized slow queries using QUANTUM ANALYZE, configured StreamPool for connection handling with 150 parallel streams"
                    ),
                    "namespace": base_conversations_ns,  # Base namespace - KB will append embedding model
                    "source": "datahaven_optimization_discussion",
                    "metadata": {"topic": "database", "subtopic": "optimization", "system": "datahaven"}
                },
                # Additional documents for list functionality testing
                {
                    "title": "Stellaris API Design",
                    "content": "Discussion about HyperLink API design patterns and quantum response formatting for Project Stellaris endpoints",
                    "namespace": base_documents_ns,  # Base namespace - KB will append embedding model
                    "source": "stellaris_api_design",
                    "metadata": {"topic": "api_design", "project": "stellaris"}
                },
                {
                    "title": "FluxContainer Deployment",
                    "content": "FluxContainer orchestration and deployment strategies using NebulaSoft's proprietary container system",
                    "namespace": base_documents_ns,  # Base namespace - KB will append embedding model
                    "source": "fluxcontainer_deployment",
                    "metadata": {"topic": "deployment", "technology": "fluxcontainer"}
                }
            ]
        
            # Ingest all test documents for this user
            for doc in test_documents:
                try:
                    doc_id = await test_agent.knowledge_base.ingest_document(
                        content=doc["content"],
                        namespace=doc["namespace"],
                        doc_type=DocumentType.CONVERSATION,
                        source=doc["source"],
                        title=doc["title"],
                        metadata=doc["metadata"]
                    )
                    created_document_ids.append(doc_id)
                    logger.info(f"Created test document: {doc['title']} (ID: {doc_id})")
                except Exception as e:
                    logger.error(f"Failed to create test document '{doc['title']}': {e}")
                    raise
            
            logger.info(f"Successfully created {len(test_documents)} test documents for user {test_user_id}")
        
        # Verify test data is searchable for each user
        await asyncio.sleep(1)  # Wait briefly for indexing
        
        for test_user_id, test_agent in test_agents.items():
            try:
                # Try a simple search to verify data is accessible
                # Use the namespace with embedding model appended for verification
                verify_namespace = f"conversations:{test_user_id}:{embedding_model}"
                results = await test_agent.knowledge_base.search(
                    query="Quixel Token Protocol MemoryVault",
                    namespaces=[verify_namespace],
                    limit=1
                )
                
                if results:
                    logger.info(f"✓ Test data verification passed for user {test_user_id} - found {len(results)} results")
                else:
                    logger.warning(f"✗ Test data verification failed for user {test_user_id} - no results found")
                    logger.warning(f"  Search namespace: {verify_namespace}")
                    logger.warning("  This may indicate the namespace mismatch issue persists")
            except Exception as e:
                logger.error(f"Test data verification error for user {test_user_id}: {e}")
        
        logger.info(f"Total documents created: {len(created_document_ids)}")
        
        # Yield control to run tests - return the test agents dictionary
        yield test_agents
        
    finally:
        # Cleanup: Delete all created documents
        if test_agents and created_document_ids:
            logger.info(f"Cleaning up {len(created_document_ids)} test documents")
            cleanup_failures = 0
            
            # Use the first available agent for cleanup (any agent can delete documents)
            cleanup_agent = list(test_agents.values())[0] if test_agents else None
            
            if cleanup_agent and cleanup_agent.knowledge_base:
                for doc_id in created_document_ids:
                    try:
                        success = await cleanup_agent.knowledge_base.delete_document(doc_id)
                        if not success:
                            logger.warning(f"Document {doc_id} may not have been deleted (returned False)")
                            cleanup_failures += 1
                    except Exception as e:
                        logger.error(f"Failed to delete test document {doc_id}: {e}")
                        cleanup_failures += 1
                
                if cleanup_failures > 0:
                    logger.warning(f"Failed to clean up {cleanup_failures} documents")
                else:
                    logger.info("Successfully cleaned up all test documents")
        
        # Note: Agents don't have an explicit cleanup method
        # Resources are cleaned up when the agent instance is garbage collected


def create_evaluation_config(fixed_user_id: str = None) -> EvaluationConfig:
    """Create evaluation configuration for knowledge base agent
    
    Args:
        fixed_user_id: Optional fixed user ID to use for all contexts (useful for consistent testing)
    """
    
    # Generate consistent user IDs for contexts
    test_user_1 = fixed_user_id or f"test_user_{uuid.uuid4()}"
    # Could add test_user_2 for isolation testing if needed
    
    # Separate models for synthesis and evaluation
    # Synthesis model - smaller/faster for generating test cases (keep OllamaModel for synthesis)
    synthesis_model = OllamaModel(model="qwen3:4b", temperature=0.7, base_url=settings.OLLAMA_BASE_URL)
    
    # Evaluation model - use CustomOllamaModel with JSON enforcement for metrics
    evaluation_model = CustomOllamaModel(model="qwen3:4b", temperature=0.0, base_url=settings.OLLAMA_BASE_URL)
    
    # Styling configuration for synthesizer
    styling_config = StylingConfig(
        scenario="User asking questions about Project Stellaris and NebulaSoft technical discussions that require knowledge base searches",
        task="Generate queries that clearly indicate knowledge base operations about fictional technologies",
        input_format="""
        - Search queries: "What did Team Phoenix discuss about Quixel Token Protocol?", "Find information about DataHaven optimization", "Search for NebulaSoft architecture discussions"
        - Cross-session queries: "Remind me what Team Aurora said about MeshLink Protocol", "What was our decision on ContainerFlux?", "Earlier we mentioned StreamPool configuration"
        - Document listing: "Show my Project Stellaris conversations", "List all documents about NebulaSoft", "What topics have we covered for Team Phoenix?"
        """,
        expected_output_format="A helpful response using information retrieved from the knowledge base about Project Stellaris and NebulaSoft technologies"
    )
    
    # Contexts with expected tools and retrieval context for RAG metrics - using fictional content
    contexts = [
        ContextWithMetadata(
            context=[
                "Team Phoenix previously discussed implementing Quixel Token Protocol with 45-minute token expiration and MemoryVault for session storage. "
                "The conversation covered security measures including token scrambling with entropy seeds."
            ],
            tools=["search_knowledge_base"],
            expected_output="Based on Team Phoenix's discussion, you decided to implement Quixel Token Protocol (QTP-3) with 45-minute token expiration and use MemoryVault for session storage. Zara Chen proposed token scrambling algorithm with entropy seeds.",
            retrieval_context=[
                "Team Phoenix meeting from 2024-11-15: Discussed Quixel Token Protocol (QTP-3) implementation for Project Stellaris. "
                "Key decisions: Use QTP tokens with 45-minute expiration, implement cascade refresh mechanism, "
                "store sessions in MemoryVault for scalability. Security lead Zara Chen proposed token scrambling algorithm with entropy seeds."
            ],
            user_id=test_user_1,
            session_id=f"session_{uuid.uuid4()}"
        ),
        ContextWithMetadata(
            context=[
                "Multiple conversations exist about NebulaSoft architecture for Project Stellaris, including discussions on "
                "service discovery with MeshLink Protocol, gateway routing with GatewayPrime, and container orchestration with ContainerFlux."
            ],
            tools=["search_knowledge_base"],
            expected_output="Found several discussions about NebulaSoft architecture: service discovery using MeshLink Protocol and NodeTracker v2.1, gateway routing with GatewayPrime and RouteForge for 50K requests per second, and ContainerFlux deployment with Orbital Charts.",
            retrieval_context=[
                "Team Aurora conversation: Discussed service discovery using MeshLink Protocol and NodeTracker v2.1 for Project Stellaris",
                "Team meeting notes: Gateway routing patterns with GatewayPrime and RouteForge for handling 50K requests per second",
                "Technical discussion: ContainerFlux deployment using Orbital Charts and FluxOps methodology with auto-scaling to 200 pods"
            ],
            user_id=test_user_1,
            session_id=f"session_{uuid.uuid4()}"
        ),
        ContextWithMetadata(
            context=[
                "User has multiple conversation archives spanning various Project Stellaris topics and NebulaSoft technical discussions."
            ],
            tools=["list_documents"],
            expected_output="Here are your recent conversations covering topics like Quixel Token Protocol, NebulaSoft architecture, DataHaven optimization, Stellaris API design, and FluxContainer deployment.",
            retrieval_context=[
                "Document list: 7 conversations found",
                "Topics: Quixel Token Protocol Implementation, NebulaSoft Architecture components, DataHaven Optimization, Stellaris API Design, FluxContainer Deployment"
            ],
            user_id=test_user_1,
            session_id=f"session_{uuid.uuid4()}"
        ),
        ContextWithMetadata(
            context=[
                "Previous conversation with Team Phoenix about DataHaven optimization included discussion of quantum indexing strategies, "
                "query optimization with QUANTUM ANALYZE, and connection handling with StreamPool."
            ],
            tools=["search_knowledge_base"],
            expected_output="In the DataHaven optimization meeting with Team Phoenix, we covered creating quantum indexes on (nebula_id, flux_timestamp), using QUANTUM ANALYZE for query optimization, and configuring StreamPool for 150 parallel streams.",
            retrieval_context=[
                "DataHaven Optimization Meeting with Team Phoenix: Created quantum indexes on (nebula_id, flux_timestamp), "
                "optimized slow queries using QUANTUM ANALYZE, configured StreamPool for connection handling with 150 parallel streams"
            ],
            user_id=test_user_1,
            session_id=f"session_{uuid.uuid4()}"
        )
    ]
    
    # Comprehensive metrics for knowledge base evaluation
    metrics = [
        # Tool usage validation
        ToolCorrectnessMetric(threshold=0.9),
        
        # RAG-specific metrics
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
        
        # General quality metrics
        AnswerRelevancyMetric(
            threshold=0.7, 
            model=evaluation_model
        ),
        HallucinationMetric(
            threshold=0.5, 
            model=evaluation_model
        ),
        
        # Custom metrics for knowledge base operations
        GEval(
            name="information_completeness",
            criteria="Does the response include all relevant information from the retrieved context without omitting key details?",
            evaluation_params=[
                LLMTestCaseParams.INPUT, 
                LLMTestCaseParams.ACTUAL_OUTPUT, 
                LLMTestCaseParams.RETRIEVAL_CONTEXT
            ],
            model=evaluation_model,
            threshold=0.7
        ),
        GEval(
            name="cross_session_awareness",
            criteria="Does the response correctly identify and reference information from previous conversations or sessions when relevant?",
            evaluation_params=[
                LLMTestCaseParams.INPUT, 
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=evaluation_model,
            threshold=0.7
        )
        ,
        GEval(
            name="search_effectiveness",
            criteria="Did the agent use the appropriate search parameters and retrieve relevant information for the user's query?",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.TOOLS_CALLED
            ],
            model=evaluation_model,
            threshold=0.8
        )
    ]
    
    # Create complete configuration
    return EvaluationConfig(
        agent_id="knowledge_agent",
        synthesizer_config=SynthesizerConfig(
            model=synthesis_model,  # Use smaller/faster model for synthesis
            styling_config=styling_config,
            max_goldens_per_context=3  # Generate more test cases for comprehensive evaluation
        ),
        metrics=metrics,
        contexts=contexts,
        dataset_name="knowledge_agent",
        dataset_file="knowledge_agent_goldens.pkl",
        results_file="knowledge_agent_results"
    )


async def main(generate_goldens: bool = False, print_verbose: bool = False):
    """Run knowledge agent evaluation with test data setup/teardown"""
    
    config = create_evaluation_config()
    
    # Extract unique user IDs from contexts
    user_ids = list(set(
        context.user_id for context in config.contexts 
        if hasattr(context, 'user_id') and context.user_id
    ))
    
    # If no user IDs found in contexts, create a default one
    if not user_ids:
        user_ids = [f"test_user_{uuid.uuid4()}"]
    
    async with knowledge_agent_test_context(test_users=user_ids) as test_agents:
        if generate_goldens:
            logger.info("Test data setup complete, generating golden test cases...")
        else:
            logger.info("Test data setup complete, running evaluation...")
        
        runner = EvaluationRunner(config)
        await runner.run(generate=generate_goldens, verbose=print_verbose)
        
        logger.info("Process complete, cleaning up test data...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run knowledge base agent evaluation with RAG metrics. "
                    "When running evaluation (not generating goldens), test documents "
                    "will be automatically created and cleaned up."
    )
    parser.add_argument("--generate", action="store_true", 
                       help="Generate golden test cases")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print out each test case and its results")
    
    args = parser.parse_args()
    
    asyncio.run(main(generate_goldens=args.generate, print_verbose=args.verbose))