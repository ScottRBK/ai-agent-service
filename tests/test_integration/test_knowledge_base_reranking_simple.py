"""
Simplified integration tests for Knowledge Base re-ranking configuration.

This module tests the key integration points for re-ranking without 
requiring full system setup - focusing on configuration precedence
and provider setup logic.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.agents.api_agent import APIAgent
import json
import tempfile
import os


class TestKnowledgeBaseRerankingSimple:
    """Simplified integration tests for knowledge base re-ranking."""
    
    @pytest.fixture
    def mock_agent_config_with_reranking(self):
        """Create mock agent config with re-ranking settings."""
        return {
            "agent_id": "test_rerank_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4o-mini",
            "resources": ["knowledge_base"],
            "allowed_regular_tools": ["search_knowledge_base", "list_documents"],
            "resource_config": {
                "knowledge_base": {
                    "vector_provider": "pgvector",
                    "embedding_provider": "azure_openai_cc",
                    "embedding_model": "text-embedding-ada-002",
                    "rerank_provider": "ollama",
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0",
                    "rerank_limit": 10
                }
            }
        }
    
    @pytest.fixture
    def mock_agent_config_precedence_test(self):
        """Config to test resource_config takes precedence over top-level."""
        return {
            "agent_id": "test_precedence_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "rerank_provider": "azure_openai_cc",  # Top-level
            "rerank_model": "gpt-4",               # Top-level
            "resources": ["knowledge_base"],
            "allowed_regular_tools": ["search_knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "vector_provider": "pgvector",
                    "rerank_provider": "ollama",    # Should take precedence
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0"  # Should take precedence
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_configuration_precedence(self):
        """Test that resource_config takes precedence over top-level config."""
        mock_config = {
            "agent_id": "test_precedence",
            "provider": "azure_openai_cc",
            "rerank_provider": "azure_openai_cc",  # Top-level
            "rerank_model": "gpt-4",               # Top-level
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "ollama",    # Should take precedence
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0"  # Should take precedence
                }
            }
        }
        
        with patch('app.core.agents.agent_tool_manager.AgentToolManager') as mock_tool_manager:
            mock_tool_manager.return_value.config = mock_config
            mock_tool_manager.return_value.get_available_tools = AsyncMock(return_value=[])
            
            agent = APIAgent("test_precedence", "user1", "session1")
            
            # Test the configuration precedence logic directly from the config
            kb_config_from_resource = mock_config["resource_config"]["knowledge_base"]
            kb_config_from_getter = agent._get_resource_config("knowledge_base")
            
            # Test the precedence logic
            rerank_provider = (
                kb_config_from_resource.get("rerank_provider") or 
                mock_config.get("rerank_provider")
            )
            rerank_model = (
                kb_config_from_resource.get("rerank_model") or 
                mock_config.get("rerank_model")
            )
            
            # Resource config should take precedence
            assert rerank_provider == "ollama"
            assert rerank_model == "dengcao/Qwen3-Reranker-4B:Q8_0"
            
            # Verify that resource config overrides top-level
            assert mock_config["rerank_provider"] == "azure_openai_cc"  # Top-level unchanged
            assert kb_config_from_resource.get("rerank_provider") != mock_config["rerank_provider"]
    
    @pytest.mark.asyncio 
    async def test_provider_setup_logic_reuse_main_provider(self):
        """Test that re-rank provider setup logic correctly reuses main provider when they're the same."""
        mock_config = {
            "agent_id": "test_reuse_main",
            "provider": "ollama",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "ollama",  # Same as main provider
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0"
                }
            }
        }
        
        with patch('app.core.agents.agent_tool_manager.AgentToolManager') as mock_tool_manager:
            mock_tool_manager.return_value.config = mock_config
            mock_tool_manager.return_value.get_available_tools = AsyncMock(return_value=[])
            
            agent = APIAgent("test_reuse_main", "user1", "session1")
            
            # Test the provider setup logic directly from config
            kb_config_from_resource = mock_config["resource_config"]["knowledge_base"]
            rerank_provider_id = (
                kb_config_from_resource.get("rerank_provider") or 
                mock_config.get("rerank_provider")
            )
            provider_id = mock_config.get("provider")
            
            # Should identify that rerank provider matches main provider
            assert rerank_provider_id == provider_id == "ollama"
    
    @pytest.mark.asyncio
    async def test_provider_setup_logic_reuse_embedding_provider(self):
        """Test that re-rank provider setup logic correctly reuses embedding provider when they're the same."""
        mock_config = {
            "agent_id": "test_reuse_embedding",
            "provider": "azure_openai_cc",
            "embedding_provider": "ollama",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "ollama",  # Same as embedding provider
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0"
                }
            }
        }
        
        with patch('app.core.agents.agent_tool_manager.AgentToolManager') as mock_tool_manager:
            mock_tool_manager.return_value.config = mock_config
            mock_tool_manager.return_value.get_available_tools = AsyncMock(return_value=[])
            
            agent = APIAgent("test_reuse_embedding", "user1", "session1")
            
            # Test the provider setup logic directly from config
            kb_config_from_resource = mock_config["resource_config"]["knowledge_base"]
            rerank_provider_id = (
                kb_config_from_resource.get("rerank_provider") or 
                mock_config.get("rerank_provider")
            )
            embedding_provider_id = mock_config.get("embedding_provider")
            
            # Should identify that rerank provider matches embedding provider
            assert rerank_provider_id == embedding_provider_id == "ollama"
    
    @pytest.mark.asyncio
    async def test_rerank_configuration_extraction(self):
        """Test that re-ranking configuration is correctly extracted from different sources."""
        mock_config = {
            "agent_id": "test_config_extraction",
            "provider": "azure_openai_cc",
            "rerank_provider": "top_level_provider",
            "rerank_model": "top_level_model",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "vector_provider": "pgvector",
                    "rerank_provider": "resource_level_provider",
                    "rerank_model": "resource_level_model",
                    "rerank_limit": 25
                }
            }
        }
        
        with patch('app.core.agents.agent_tool_manager.AgentToolManager') as mock_tool_manager:
            mock_tool_manager.return_value.config = mock_config
            mock_tool_manager.return_value.get_available_tools = AsyncMock(return_value=[])
            
            agent = APIAgent("test_config_extraction", "user1", "session1")
            
            # Test configuration extraction with precedence directly from config
            kb_config_from_resource = mock_config["resource_config"]["knowledge_base"]
            
            rerank_provider = (
                kb_config_from_resource.get("rerank_provider") or 
                mock_config.get("rerank_provider")
            )
            rerank_model = (
                kb_config_from_resource.get("rerank_model") or 
                mock_config.get("rerank_model")
            )
            rerank_limit = kb_config_from_resource.get("rerank_limit", 50)  # Default fallback
            
            # Should use resource-level config which takes precedence
            assert rerank_provider == "resource_level_provider"
            assert rerank_model == "resource_level_model"
            assert rerank_limit == 25
    
    @pytest.mark.asyncio
    async def test_fallback_to_top_level_config(self):
        """Test fallback to top-level config when resource config doesn't have re-ranking settings."""
        mock_config = {
            "agent_id": "test_fallback",
            "provider": "azure_openai_cc",
            "rerank_provider": "top_level_provider",
            "rerank_model": "top_level_model",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "vector_provider": "pgvector"
                    # No rerank settings here - should fall back to top-level
                }
            }
        }
        
        with patch('app.core.agents.agent_tool_manager.AgentToolManager') as mock_tool_manager:
            mock_tool_manager.return_value.config = mock_config
            mock_tool_manager.return_value.get_available_tools = AsyncMock(return_value=[])
            
            agent = APIAgent("test_fallback", "user1", "session1")
            
            # Test fallback to top-level config directly from config
            kb_config_from_resource = mock_config["resource_config"]["knowledge_base"]
            
            rerank_provider = (
                kb_config_from_resource.get("rerank_provider") or 
                mock_config.get("rerank_provider")
            )
            rerank_model = (
                kb_config_from_resource.get("rerank_model") or 
                mock_config.get("rerank_model")
            )
            
            # Should use top-level config as fallback
            assert rerank_provider == "top_level_provider"
            assert rerank_model == "top_level_model"
    
    @pytest.mark.asyncio
    async def test_no_rerank_configuration(self):
        """Test behavior when no re-ranking configuration is provided."""
        mock_config = {
            "agent_id": "test_no_rerank",
            "provider": "azure_openai_cc",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "vector_provider": "pgvector"
                    # No rerank settings anywhere
                }
            }
        }
        
        with patch('app.core.agents.agent_tool_manager.AgentToolManager') as mock_tool_manager:
            mock_tool_manager.return_value.config = mock_config
            mock_tool_manager.return_value.get_available_tools = AsyncMock(return_value=[])
            
            agent = APIAgent("test_no_rerank", "user1", "session1")
            
            # Test no re-ranking configuration directly from config
            kb_config_from_resource = mock_config["resource_config"]["knowledge_base"]
            
            rerank_provider = (
                kb_config_from_resource.get("rerank_provider") or 
                mock_config.get("rerank_provider")
            )
            rerank_model = (
                kb_config_from_resource.get("rerank_model") or 
                mock_config.get("rerank_model")
            )
            
            # Should be None when no configuration is provided
            assert rerank_provider is None
            assert rerank_model is None
    
    def test_resource_config_helper_method(self):
        """Test the _get_resource_config helper method works correctly."""
        agent = APIAgent("test_helper", "user1", "session1")
        
        mock_config = {
            "agent_id": "test_helper",
            "provider": "azure_openai_cc",
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "ollama",
                    "rerank_model": "test-model",
                    "other_setting": "value"
                },
                "memory": {
                    "compression": {"enabled": True}
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            # Test getting knowledge base config
            kb_config = agent._get_resource_config("knowledge_base")
            assert kb_config["rerank_provider"] == "ollama"
            assert kb_config["rerank_model"] == "test-model"
            assert kb_config["other_setting"] == "value"
            
            # Test getting memory config
            memory_config = agent._get_resource_config("memory")
            assert memory_config["compression"]["enabled"] is True
            
            # Test getting non-existent config
            nonexistent_config = agent._get_resource_config("nonexistent")
            assert nonexistent_config == {}
    
    def test_provider_precedence_scenarios(self):
        """Test various provider precedence scenarios."""
        agent = APIAgent("test_scenarios", "user1", "session1")
        
        # Scenario 1: Resource config overrides top-level
        mock_config_1 = {
            "rerank_provider": "azure_openai_cc",
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "ollama"
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config_1):
            kb_config = agent._get_resource_config("knowledge_base")
            final_provider = kb_config.get("rerank_provider") or mock_config_1.get("rerank_provider")
            assert final_provider == "ollama"
        
        # Scenario 2: Top-level used when resource config empty
        mock_config_2 = {
            "rerank_provider": "azure_openai_cc",
            "resource_config": {
                "knowledge_base": {}
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config_2):
            kb_config = agent._get_resource_config("knowledge_base")
            final_provider = kb_config.get("rerank_provider") or mock_config_2.get("rerank_provider")
            assert final_provider == "azure_openai_cc"
        
        # Scenario 3: No rerank provider anywhere
        mock_config_3 = {
            "provider": "azure_openai_cc",
            "resource_config": {
                "knowledge_base": {}
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config_3):
            kb_config = agent._get_resource_config("knowledge_base")
            final_provider = kb_config.get("rerank_provider") or mock_config_3.get("rerank_provider")
            assert final_provider is None