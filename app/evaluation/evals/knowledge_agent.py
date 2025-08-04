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
from app.core.agents.cli_agent import CLIAgent
from app.models.resources.knowledge_base import DocumentType
from app.utils.logging import logger

import asyncio
import argparse
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any


@asynccontextmanager
async def knowledge_agent_test_context():
    """
    Context manager for setting up and tearing down test data for knowledge agent evaluation.
    
    Sets up test documents in the knowledge base that match the evaluation contexts,
    then cleans them up after tests complete.
    """
    # Track created documents for cleanup
    created_document_ids: List[str] = []
    test_agent = None
    
    try:
        # Create a test agent instance
        test_user_id = f"test_user_{uuid.uuid4()}"
        test_session_id = f"test_session_{uuid.uuid4()}"
        test_agent = CLIAgent(
            agent_id="knowledge_agent",
            user_id=test_user_id,
            session_id=test_session_id
        )
        await test_agent.initialize()
        
        # Ensure knowledge base is available
        if not test_agent.knowledge_base:
            raise RuntimeError("Knowledge agent does not have knowledge base configured")
        
        logger.info(f"Setting up test data for knowledge agent evaluation (user: {test_user_id})")
        
        # Define test documents matching the evaluation contexts
        test_documents = [
            {
                "title": "JWT Authentication Discussion",
                "content": (
                    "Conversation from 2024-01-15: Discussed JWT authentication implementation. "
                    "Key decisions: Use JWT tokens with 1-hour expiration, implement refresh token mechanism, "
                    "store sessions in Redis for scalability. Security considerations included token rotation and secure storage."
                ),
                "namespace": f"conversations:{test_user_id}",
                "source": "jwt_auth_discussion_2024-01-15",
                "metadata": {"topic": "authentication", "date": "2024-01-15"}
            },
            {
                "title": "Microservices - Service Discovery",
                "content": "Conversation 1: Discussed service discovery using Consul and etcd",
                "namespace": f"conversations:{test_user_id}",
                "source": "microservices_service_discovery",
                "metadata": {"topic": "microservices", "subtopic": "service_discovery"}
            },
            {
                "title": "Microservices - API Gateway",
                "content": "Conversation 2: API gateway patterns with Kong and Traefik",
                "namespace": f"conversations:{test_user_id}",
                "source": "microservices_api_gateway",
                "metadata": {"topic": "microservices", "subtopic": "api_gateway"}
            },
            {
                "title": "Microservices - Kubernetes Deployment",
                "content": "Conversation 3: Kubernetes deployment with Helm charts and GitOps",
                "namespace": f"conversations:{test_user_id}",
                "source": "microservices_kubernetes",
                "metadata": {"topic": "microservices", "subtopic": "kubernetes"}
            },
            {
                "title": "Database Optimization Discussion",
                "content": (
                    "Database Optimization Discussion: Created composite indexes on (user_id, created_at), "
                    "optimized slow queries using EXPLAIN ANALYZE, configured PgBouncer for connection pooling with 100 max connections"
                ),
                "namespace": f"conversations:{test_user_id}",
                "source": "database_optimization_discussion",
                "metadata": {"topic": "database", "subtopic": "optimization"}
            },
            # Additional documents for list functionality testing
            {
                "title": "REST API Design",
                "content": "Discussion about RESTful API design patterns and best practices",
                "namespace": f"documents:{test_user_id}",
                "source": "rest_api_design",
                "metadata": {"topic": "api_design"}
            },
            {
                "title": "Docker Deployment",
                "content": "Docker containerization and deployment strategies",
                "namespace": f"documents:{test_user_id}",
                "source": "docker_deployment",
                "metadata": {"topic": "deployment"}
            }
        ]
        
        # Ingest all test documents
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
        
        logger.info(f"Successfully created {len(created_document_ids)} test documents")
        
        # Yield control to run tests
        yield test_agent
        
    finally:
        # Cleanup: Delete all created documents
        if test_agent and test_agent.knowledge_base and created_document_ids:
            logger.info(f"Cleaning up {len(created_document_ids)} test documents")
            cleanup_failures = 0
            
            for doc_id in created_document_ids:
                try:
                    success = await test_agent.knowledge_base.delete_document(doc_id)
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


def create_evaluation_config() -> EvaluationConfig:
    """Create evaluation configuration for knowledge base agent"""
    
    # Model for evaluation
    model = OllamaModel(model="mistral:7b", temperature=0.0)
    
    # Styling configuration for synthesizer
    styling_config = StylingConfig(
        scenario="User asking questions that require knowledge base searches or document management",
        task="Generate queries that clearly indicate knowledge base operations",
        input_format="""
        - Search queries: "What did we discuss about X?", "Find information about Y in our conversations", "Search for previous discussions on Z"
        - Cross-session queries: "Remind me what we talked about last time", "What was our decision on X?", "Earlier you mentioned Y"
        - Document listing: "Show my recent conversations", "List all documents about X", "What topics have we covered?"
        """,
        expected_output_format="A helpful response using information retrieved from the knowledge base"
    )
    
    # Contexts with expected tools and retrieval context for RAG metrics
    contexts = [
        ContextWithMetadata(
            context=[
                "User previously discussed implementing JWT authentication with 1-hour token expiration and Redis for session storage. "
                "The conversation covered security best practices and implementation details."
            ],
            tools=["search_knowledge_base"],
            expected_output="Based on our previous discussion, you decided to implement JWT authentication with 1-hour token expiration and use Redis for session storage.",
            retrieval_context=[
                "Conversation from 2024-01-15: Discussed JWT authentication implementation. "
                "Key decisions: Use JWT tokens with 1-hour expiration, implement refresh token mechanism, "
                "store sessions in Redis for scalability. Security considerations included token rotation and secure storage."
            ]
        ),
        ContextWithMetadata(
            context=[
                "Multiple conversations exist about microservices architecture, including discussions on "
                "service discovery, API gateways, and container orchestration with Kubernetes."
            ],
            tools=["search_knowledge_base"],
            expected_output="Found several discussions about microservices: service discovery patterns, API gateway implementation, and Kubernetes deployment strategies.",
            retrieval_context=[
                "Conversation 1: Discussed service discovery using Consul and etcd",
                "Conversation 2: API gateway patterns with Kong and Traefik",
                "Conversation 3: Kubernetes deployment with Helm charts and GitOps"
            ]
        ),
        ContextWithMetadata(
            context=[
                "User has multiple conversation archives spanning various technical topics and project discussions."
            ],
            tools=["list_documents"],
            expected_output="Here are your recent conversations covering topics like authentication, microservices, database design, and API development.",
            retrieval_context=[
                "Document list: 10 conversations found",
                "Topics: JWT Authentication, Microservices Architecture, PostgreSQL Optimization, REST API Design, Docker Deployment"
            ]
        ),
        ContextWithMetadata(
            context=[
                "Previous conversation about database optimization included discussion of indexing strategies, "
                "query optimization, and connection pooling for PostgreSQL."
            ],
            tools=["search_knowledge_base"],
            expected_output="In our database optimization discussion, we covered creating composite indexes, using EXPLAIN ANALYZE for query optimization, and configuring connection pooling.",
            retrieval_context=[
                "Database Optimization Discussion: Created composite indexes on (user_id, created_at), "
                "optimized slow queries using EXPLAIN ANALYZE, configured PgBouncer for connection pooling with 100 max connections"
            ]
        )
    ]
    
    # Comprehensive metrics for knowledge base evaluation
    metrics = [
        # Tool usage validation
        ToolCorrectnessMetric(threshold=0.9),
        
        # RAG-specific metrics
        FaithfulnessMetric(
            threshold=0.7, 
            model=model,
            include_reason=True
        ),
        ContextualRelevancyMetric(
            threshold=0.7, 
            model=model,
            include_reason=True
        ),
        ContextualRecallMetric(
            threshold=0.6, 
            model=model,
            include_reason=True
        ),
        ContextualPrecisionMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        ),
        
        # General quality metrics
        AnswerRelevancyMetric(
            threshold=0.7, 
            model=model
        ),
        HallucinationMetric(
            threshold=0.5, 
            model=model
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
            model=model,
            threshold=0.7
        ),
        GEval(
            name="cross_session_awareness",
            criteria="Does the response correctly identify and reference information from previous conversations or sessions when relevant?",
            evaluation_params=[
                LLMTestCaseParams.INPUT, 
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=model,
            threshold=0.7
        ),
        GEval(
            name="search_effectiveness",
            criteria="Did the agent use the appropriate search parameters and retrieve relevant information for the user's query?",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.TOOLS_CALLED
            ],
            model=model,
            threshold=0.8
        )
    ]
    
    # Create complete configuration
    return EvaluationConfig(
        agent_id="knowledge_agent",
        synthesizer_config=SynthesizerConfig(
            model=model,
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
    
    # Create evaluation configuration
    config = create_evaluation_config()
    
    # If generating goldens, we don't need test data setup
    if generate_goldens:
        runner = EvaluationRunner(config)
        await runner.run(generate=True, verbose=print_verbose)
    else:
        # Use context manager to set up test data before evaluation
        async with knowledge_agent_test_context() as test_agent:
            logger.info("Test data setup complete, running evaluation...")
            
            # Create and run evaluation
            runner = EvaluationRunner(config)
            await runner.run(generate=False, verbose=print_verbose)
            
            logger.info("Evaluation complete, cleaning up test data...")


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