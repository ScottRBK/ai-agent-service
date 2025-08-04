#!/usr/bin/env python3
"""
Quick test script to verify the knowledge agent context manager works correctly.
This will test the setup and teardown of test documents.
"""

import asyncio
import sys
sys.path.append('.')

from app.evaluation.evals.knowledge_agent import knowledge_agent_test_context
from app.utils.logging import logger


async def test_context_manager():
    """Test the knowledge agent context manager"""
    logger.info("Starting context manager test...")
    
    async with knowledge_agent_test_context() as test_agent:
        logger.info("Inside context manager")
        
        # Verify knowledge base is available
        assert test_agent.knowledge_base is not None, "Knowledge base should be initialized"
        
        # Try searching for some of the test data
        results = await test_agent.knowledge_base.search(
            query="JWT authentication",
            namespaces=[f"conversations:{test_agent.user_id}"],
            limit=5
        )
        
        logger.info(f"Found {len(results)} results for JWT authentication search")
        assert len(results) > 0, "Should find JWT authentication document"
        
        # List documents
        conv_docs = await test_agent.knowledge_base.list_documents(f"conversations:{test_agent.user_id}")
        doc_docs = await test_agent.knowledge_base.list_documents(f"documents:{test_agent.user_id}")
        
        logger.info(f"Found {len(conv_docs)} conversation documents")
        logger.info(f"Found {len(doc_docs)} regular documents")
        
        assert len(conv_docs) >= 5, "Should have at least 5 conversation documents"
        assert len(doc_docs) >= 2, "Should have at least 2 regular documents"
    
    logger.info("Context manager exited, documents should be cleaned up")
    logger.info("Test passed!")


if __name__ == "__main__":
    asyncio.run(test_context_manager())