"""
Unit tests for BaseAgent resource_config handling and provider configuration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.agents.base_agent import BaseAgent


class TestBaseAgentResourceConfig:
    """Test cases for BaseAgent resource_config handling"""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock provider"""
        provider = Mock()
        provider.initialize = AsyncMock()
        provider.config = Mock(default_model="gpt-3.5-turbo")
        return provider
    
    @pytest.fixture
    def mock_provider_manager(self, mock_provider):
        """Mock provider manager"""
        manager = Mock()
        manager.get_provider.return_value = {
            "class": Mock(return_value=mock_provider),
            "config_class": Mock
        }
        return manager
    
    def test_get_resource_config_method(self):
        """Test _get_resource_config method"""
        agent = BaseAgent("test_agent")
        
        # Mock config with resource_config
        mock_config = {
            "resource_config": {
                "memory": {
                    "compression": {
                        "threshold_tokens": 5000,
                        "recent_messages_to_keep": 5
                    }
                },
                "knowledge_base": {
                    "embedding_provider": "azure_openai_cc",
                    "embedding_model": "text-embedding-ada-002",
                    "chunk_size": 800
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            # Test getting memory config
            memory_config = agent._get_resource_config("memory")
            assert memory_config == {
                "compression": {
                    "threshold_tokens": 5000,
                    "recent_messages_to_keep": 5
                }
            }
            
            # Test getting knowledge_base config
            kb_config = agent._get_resource_config("knowledge_base")
            assert kb_config == {
                "embedding_provider": "azure_openai_cc",
                "embedding_model": "text-embedding-ada-002",
                "chunk_size": 800
            }
            
            # Test getting non-existent config
            other_config = agent._get_resource_config("nonexistent")
            assert other_config == {}
    
    def test_get_resource_config_no_resource_config(self):
        """Test _get_resource_config when no resource_config exists"""
        agent = BaseAgent("test_agent")
        
        # Mock config without resource_config
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc"
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            config = agent._get_resource_config("memory")
            assert config == {}
    
    @pytest.mark.asyncio
    async def test_setup_providers_embedding_from_resource_config(self, mock_provider_manager):
        """Test setup_providers uses embedding_provider from resource_config.knowledge_base"""
        agent = BaseAgent("test_agent")
        agent.provider_manager = mock_provider_manager
        
        # Mock config with embedding provider in resource_config
        mock_config = {
            "provider": "azure_openai_cc",
            "resource_config": {
                "knowledge_base": {
                    "embedding_provider": "ollama",
                    "embedding_model": "custom-embedding-model"
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            await agent.setup_providers()
            
            # Should call get_provider for both main and embedding providers
            assert mock_provider_manager.get_provider.call_count == 2
            
            # First call should be for main provider
            first_call = mock_provider_manager.get_provider.call_args_list[0][0][0]
            assert first_call == "azure_openai_cc"
            
            # Second call should be for embedding provider from resource_config
            second_call = mock_provider_manager.get_provider.call_args_list[1][0][0]
            assert second_call == "ollama"
    
    @pytest.mark.asyncio
    async def test_setup_providers_embedding_from_top_level_config(self, mock_provider_manager):
        """Test setup_providers falls back to top-level embedding_provider config"""
        agent = BaseAgent("test_agent")
        agent.provider_manager = mock_provider_manager
        
        # Mock config with embedding provider at top level
        mock_config = {
            "provider": "azure_openai_cc",
            "embedding_provider": "ollama",
            "resource_config": {
                "knowledge_base": {
                    "chunk_size": 800
                    # No embedding_provider here
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            await agent.setup_providers()
            
            # Should call get_provider for both providers
            assert mock_provider_manager.get_provider.call_count == 2
            
            # Second call should be for embedding provider from top-level config
            second_call = mock_provider_manager.get_provider.call_args_list[1][0][0]
            assert second_call == "ollama"
    
    @pytest.mark.asyncio
    async def test_setup_providers_same_embedding_and_main_provider(self, mock_provider_manager):
        """Test setup_providers reuses main provider when embedding provider is same"""
        agent = BaseAgent("test_agent")
        agent.provider_manager = mock_provider_manager
        
        # Mock config where embedding provider is same as main provider
        mock_config = {
            "provider": "azure_openai_cc",
            "resource_config": {
                "knowledge_base": {
                    "embedding_provider": "azure_openai_cc"  # Same as main
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            await agent.setup_providers()
            
            # Should only call get_provider once (for main provider)
            assert mock_provider_manager.get_provider.call_count == 1
            
            # Embedding provider should be same as main provider
            assert agent.embedding_provider == agent.provider
    
    @pytest.mark.asyncio
    async def test_create_knowledge_base_embedding_model_from_resource_config(self, mock_provider_manager):
        """Test create_knowledge_base uses embedding_model from resource_config"""
        agent = BaseAgent("test_agent")
        agent.provider_manager = mock_provider_manager
        agent.provider = Mock()
        agent.embedding_provider = Mock()
        
        # Mock config with embedding model in resource_config
        mock_config = {
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "resource_config": {
                "knowledge_base": {
                    "embedding_model": "text-embedding-ada-002",
                    "chunk_size": 800
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                mock_kb = Mock()
                mock_kb.initialize = AsyncMock()
                mock_kb.set_chat_provider = Mock()
                mock_kb.set_embedding_provider = Mock()
                mock_kb_class.return_value = mock_kb
                
                kb = await agent.create_knowledge_base()
                
                # Should call set_embedding_provider with embedding model from resource_config
                mock_kb.set_embedding_provider.assert_called_once_with(
                    agent.embedding_provider, 
                    "text-embedding-ada-002"
                )
    
    @pytest.mark.asyncio
    async def test_create_knowledge_base_embedding_model_fallback(self, mock_provider_manager):
        """Test create_knowledge_base falls back to top-level config for embedding_model"""
        agent = BaseAgent("test_agent")
        agent.provider_manager = mock_provider_manager
        agent.provider = Mock()
        agent.embedding_provider = Mock()
        
        # Mock config with embedding model at top level
        mock_config = {
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "embedding_model": "text-embedding-3-small",
            "resource_config": {
                "knowledge_base": {
                    "chunk_size": 800
                    # No embedding_model here
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                mock_kb = Mock()
                mock_kb.initialize = AsyncMock()
                mock_kb.set_chat_provider = Mock()
                mock_kb.set_embedding_provider = Mock()
                mock_kb_class.return_value = mock_kb
                
                kb = await agent.create_knowledge_base()
                
                # Should call set_embedding_provider with embedding model from top-level config
                mock_kb.set_embedding_provider.assert_called_once_with(
                    agent.embedding_provider, 
                    "text-embedding-3-small"
                )
    
    @pytest.mark.asyncio
    async def test_create_knowledge_base_embedding_model_default_fallback(self, mock_provider_manager):
        """Test create_knowledge_base falls back to default model when no embedding_model configured"""
        agent = BaseAgent("test_agent", model="custom-model")
        agent.provider_manager = mock_provider_manager
        agent.provider = Mock()
        agent.provider.config.default_model = "provider-default"
        agent.embedding_provider = Mock()
        
        # Mock config without embedding_model anywhere
        mock_config = {
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "resource_config": {
                "knowledge_base": {
                    "chunk_size": 800
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                mock_kb = Mock()
                mock_kb.initialize = AsyncMock()
                mock_kb.set_chat_provider = Mock()
                mock_kb.set_embedding_provider = Mock()
                mock_kb_class.return_value = mock_kb
                
                kb = await agent.create_knowledge_base()
                
                # Should use requested model as fallback
                mock_kb.set_embedding_provider.assert_called_once_with(
                    agent.embedding_provider, 
                    "custom-model"
                )
    
    @pytest.mark.asyncio
    async def test_create_knowledge_base_config_passed_correctly(self, mock_provider_manager):
        """Test create_knowledge_base passes resource_config correctly"""
        agent = BaseAgent("test_agent")
        agent.provider_manager = mock_provider_manager
        agent.provider = Mock()
        agent.embedding_provider = Mock()
        
        # Mock config with complete knowledge_base resource_config
        mock_config = {
            "provider": "azure_openai_cc",
            "resource_config": {
                "knowledge_base": {
                    "embedding_provider": "ollama",
                    "embedding_model": "custom-embedding",
                    "chunk_size": 1200,
                    "chunk_overlap": 150,
                    "rerank_limit": 30
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                mock_kb = Mock()
                mock_kb.initialize = AsyncMock()
                mock_kb.set_chat_provider = Mock()
                mock_kb.set_embedding_provider = Mock()
                mock_kb_class.return_value = mock_kb
                
                with patch('app.config.settings.settings') as mock_settings:
                    mock_settings.POSTGRES_USER = "user"
                    mock_settings.POSTGRES_PASSWORD = "pass"
                    mock_settings.POSTGRES_HOST = "localhost"
                    mock_settings.POSTGRES_PORT = "5432"
                    mock_settings.POSTGRES_DB = "testdb"
                    
                    kb = await agent.create_knowledge_base()
                    
                    # Check that KnowledgeBaseResource was created with correct config
                    call_args = mock_kb_class.call_args
                    assert call_args[0][0] == "test_agent_kb"  # resource_id
                    
                    config_arg = call_args[0][1]  # config dict
                    assert "connection_string" in config_arg
                    assert config_arg["chunk_size"] == 1200
                    assert config_arg["chunk_overlap"] == 150
                    assert config_arg["rerank_limit"] == 30
    
    def test_memory_compression_config_from_resource_config(self):
        """Test that memory compression config is read from resource_config"""
        agent = BaseAgent("test_agent")
        
        # Mock config with compression settings in resource_config
        mock_config = {
            "resource_config": {
                "memory": {
                    "compression": {
                        "threshold_tokens": 6000,
                        "recent_messages_to_keep": 8,
                        "enabled": True
                    }
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            with patch('app.core.agents.memory_compression_agent.MemoryCompressionAgent') as mock_compression_class:
                mock_compression_agent = Mock()
                mock_compression_agent.compress_conversation = AsyncMock()
                mock_compression_class.return_value = mock_compression_agent
                
                # Get the compression config that would be used
                compression_config = agent._get_resource_config("memory").get("compression", {})
                
                # Apply defaults for missing values
                compression_config = {
                    "threshold_tokens": compression_config.get("threshold_tokens", 10000),
                    "recent_messages_to_keep": compression_config.get("recent_messages_to_keep", 10),
                    "enabled": compression_config.get("enabled", True)
                }
                
                assert compression_config["threshold_tokens"] == 6000
                assert compression_config["recent_messages_to_keep"] == 8
                assert compression_config["enabled"] == True
    
    def test_memory_compression_config_defaults_when_no_resource_config(self):
        """Test that memory compression uses defaults when no resource_config"""
        agent = BaseAgent("test_agent")
        
        # Mock config without resource_config
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc"
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            # Get the compression config (should be empty)
            compression_config = agent._get_resource_config("memory").get("compression", {})
            
            # Apply defaults for missing values
            compression_config = {
                "threshold_tokens": compression_config.get("threshold_tokens", 10000),
                "recent_messages_to_keep": compression_config.get("recent_messages_to_keep", 10),
                "enabled": compression_config.get("enabled", True)
            }
            
            # Should use default values
            assert compression_config["threshold_tokens"] == 10000
            assert compression_config["recent_messages_to_keep"] == 10
            assert compression_config["enabled"] == True
    
    @pytest.mark.asyncio
    async def test_agent_instance_set_on_provider_during_setup(self, mock_provider_manager):
        """Test that agent sets itself as agent_instance on provider during setup"""
        agent = BaseAgent("test_agent")
        agent.provider_manager = mock_provider_manager
        
        mock_config = {
            "provider": "azure_openai_cc"
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                await agent.setup_providers()
                
                # Provider should have agent_instance set to this agent
                assert agent.provider.agent_instance == agent