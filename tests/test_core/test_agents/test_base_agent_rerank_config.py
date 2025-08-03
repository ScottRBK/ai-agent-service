"""
Unit tests for BaseAgent re-ranking configuration functionality.

This module tests the BaseAgent's setup_providers() and create_knowledge_base()
methods to ensure proper re-ranking provider configuration and precedence handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.agents.base_agent import BaseAgent


class TestBaseAgentRerankConfiguration:
    """Test cases for BaseAgent re-ranking configuration."""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock provider"""
        provider = Mock()
        provider.initialize = AsyncMock()
        return provider
    
    @pytest.fixture
    def mock_knowledge_base_resource(self):
        """Mock knowledge base resource"""
        resource = Mock()
        resource.initialize = AsyncMock()
        resource.set_rerank_provider = Mock()
        return resource
    
    @pytest.mark.asyncio
    async def test_setup_providers_rerank_from_resource_config(self, mock_provider):
        """Test that setup_providers checks resource_config.knowledge_base.rerank_provider first."""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "rerank_provider": "azure_openai_cc",  # Top-level config
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "ollama",  # Should take precedence
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0"
                }
            }
        }
        
        mock_main_provider = Mock()
        mock_rerank_provider = Mock()
        mock_main_provider.initialize = AsyncMock()
        mock_rerank_provider.initialize = AsyncMock()
        
        with patch.object(agent.tool_manager, 'config', mock_config), \
             patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
            
            with patch.object(agent.provider_manager, 'get_provider') as mock_get_provider:
                def get_provider_side_effect(provider_id):
                    if provider_id == "azure_openai_cc":
                        return {
                            "class": Mock(return_value=mock_main_provider),
                            "config_class": Mock
                        }
                    elif provider_id == "ollama":
                        return {
                            "class": Mock(return_value=mock_rerank_provider),
                            "config_class": Mock
                        }
                
                mock_get_provider.side_effect = get_provider_side_effect
                
                await agent.setup_providers()
                
                # Should use resource_config rerank_provider, not top-level
                assert agent.rerank_provider == mock_rerank_provider
                assert agent.rerank_provider != agent.provider
    
    @pytest.mark.asyncio
    async def test_setup_providers_rerank_fallback_to_top_level(self, mock_provider):
        """Test that setup_providers falls back to top-level rerank_provider."""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "rerank_provider": "ollama",  # Top-level - should be used
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    # No rerank_provider in resource_config
                    "vector_provider": "pgvector"
                }
            }
        }
        
        mock_main_provider = Mock()
        mock_rerank_provider = Mock()
        mock_main_provider.initialize = AsyncMock()
        mock_rerank_provider.initialize = AsyncMock()
        
        with patch.object(agent.tool_manager, 'config', mock_config), \
             patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
            
            with patch.object(agent.provider_manager, 'get_provider') as mock_get_provider:
                def get_provider_side_effect(provider_id):
                    if provider_id == "azure_openai_cc":
                        return {
                            "class": Mock(return_value=mock_main_provider),
                            "config_class": Mock
                        }
                    elif provider_id == "ollama":
                        return {
                            "class": Mock(return_value=mock_rerank_provider),
                            "config_class": Mock
                        }
                
                mock_get_provider.side_effect = get_provider_side_effect
                
                await agent.setup_providers()
                
                # Should use top-level rerank_provider since resource_config doesn't have one
                assert agent.rerank_provider == mock_rerank_provider
    
    @pytest.mark.asyncio
    async def test_setup_providers_rerank_provider_reuse_main(self, mock_provider):
        """Test that rerank provider reuses main provider when they're the same."""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "ollama",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "ollama",  # Same as main provider
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0"
                }
            }
        }
        
        mock_main_provider = Mock()
        mock_main_provider.initialize = AsyncMock()
        
        with patch.object(agent.tool_manager, 'config', mock_config), \
             patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
            
            with patch.object(agent.provider_manager, 'get_provider') as mock_get_provider:
                mock_get_provider.return_value = {
                    "class": Mock(return_value=mock_main_provider),
                    "config_class": Mock
                }
                
                await agent.setup_providers()
                
                # Should reuse the same provider instance
                assert agent.rerank_provider is agent.provider
                assert agent.rerank_provider is mock_main_provider
    
    @pytest.mark.asyncio
    async def test_setup_providers_rerank_provider_reuse_embedding(self, mock_provider):
        """Test that rerank provider reuses embedding provider when they're the same."""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        
        mock_config = {
            "agent_id": "test_agent",
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
        
        mock_main_provider = Mock()
        mock_embedding_provider = Mock()
        mock_main_provider.initialize = AsyncMock()
        mock_embedding_provider.initialize = AsyncMock()
        
        with patch.object(agent.tool_manager, 'config', mock_config), \
             patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
            
            with patch.object(agent.provider_manager, 'get_provider') as mock_get_provider:
                def get_provider_side_effect(provider_id):
                    if provider_id == "azure_openai_cc":
                        return {
                            "class": Mock(return_value=mock_main_provider),
                            "config_class": Mock
                        }
                    elif provider_id == "ollama":
                        return {
                            "class": Mock(return_value=mock_embedding_provider),
                            "config_class": Mock
                        }
                
                mock_get_provider.side_effect = get_provider_side_effect
                
                await agent.setup_providers()
                
                # Should reuse the embedding provider instance
                assert agent.rerank_provider is agent.embedding_provider
                assert agent.rerank_provider is mock_embedding_provider
                assert agent.rerank_provider is not agent.provider
    
    @pytest.mark.asyncio
    async def test_setup_providers_no_rerank_provider(self, mock_provider):
        """Test setup_providers when no rerank provider is configured."""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "vector_provider": "pgvector"
                    # No rerank_provider specified anywhere
                }
            }
        }
        
        mock_main_provider = Mock()
        mock_main_provider.initialize = AsyncMock()
        
        with patch.object(agent.tool_manager, 'config', mock_config), \
             patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
            
            with patch.object(agent.provider_manager, 'get_provider') as mock_get_provider:
                mock_get_provider.return_value = {
                    "class": Mock(return_value=mock_main_provider),
                    "config_class": Mock
                }
                
                await agent.setup_providers()
                
                # Should not have rerank provider
                assert agent.rerank_provider is None
    
    def test_rerank_model_precedence_logic(self):
        """Test the logic for determining rerank_model precedence."""
        agent = BaseAgent("test_agent")
        
        # Test resource_config takes precedence
        mock_config = {
            "rerank_model": "top-level-model",  # Top-level
            "resource_config": {
                "knowledge_base": {
                    "rerank_model": "resource-config-model"  # Should take precedence
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            kb_config = agent._get_resource_config("knowledge_base")
            
            # Check precedence logic
            rerank_model = kb_config.get("rerank_model") or mock_config.get("rerank_model")
            assert rerank_model == "resource-config-model"
    
    def test_rerank_model_fallback_logic(self):
        """Test the logic for falling back to top-level rerank_model."""
        agent = BaseAgent("test_agent")
        
        # Test fallback to top-level when resource_config doesn't have it
        mock_config = {
            "rerank_model": "top-level-model",  # Should be used as fallback
            "resource_config": {
                "knowledge_base": {
                    "vector_provider": "pgvector"
                    # No rerank_model here
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            kb_config = agent._get_resource_config("knowledge_base")
            
            # Check fallback logic
            rerank_model = kb_config.get("rerank_model") or mock_config.get("rerank_model")
            assert rerank_model == "top-level-model"
    
    def test_get_resource_config_knowledge_base(self):
        """Test _get_resource_config method for knowledge_base resource."""
        agent = BaseAgent("test_agent")
        
        mock_config = {
            "resource_config": {
                "memory": {
                    "compression": {"enabled": True}
                },
                "knowledge_base": {
                    "vector_provider": "pgvector",
                    "rerank_provider": "ollama",
                    "rerank_model": "dengcao/Qwen3-Reranker-4B:Q8_0"
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            kb_config = agent._get_resource_config("knowledge_base")
            
            assert kb_config["vector_provider"] == "pgvector"
            assert kb_config["rerank_provider"] == "ollama"
            assert kb_config["rerank_model"] == "dengcao/Qwen3-Reranker-4B:Q8_0"
    
    def test_get_resource_config_nonexistent_resource(self):
        """Test _get_resource_config method for non-existent resource."""
        agent = BaseAgent("test_agent")
        
        mock_config = {
            "resource_config": {
                "memory": {
                    "compression": {"enabled": True}
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            nonexistent_config = agent._get_resource_config("nonexistent")
            
            assert nonexistent_config == {}
    
    def test_get_resource_config_no_resource_config(self):
        """Test _get_resource_config when no resource_config exists."""
        agent = BaseAgent("test_agent")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc"
            # No resource_config
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            kb_config = agent._get_resource_config("knowledge_base")
            
            assert kb_config == {}
    
    def test_rerank_provider_precedence_logic(self):
        """Test the logic for determining rerank_provider precedence."""
        agent = BaseAgent("test_agent")
        
        # Test resource_config takes precedence
        mock_config = {
            "rerank_provider": "top-level-provider",  # Top-level
            "resource_config": {
                "knowledge_base": {
                    "rerank_provider": "resource-config-provider"  # Should take precedence
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            kb_config = agent._get_resource_config("knowledge_base")
            
            # Check precedence logic
            rerank_provider = kb_config.get("rerank_provider") or mock_config.get("rerank_provider")
            assert rerank_provider == "resource-config-provider"