"""
Integration tests for resource_config configuration reading and provider selection.
Tests the complete configuration flow from agent_config.json through resource creation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.agents.base_agent import BaseAgent


class TestResourceConfigIntegration:
    """Integration tests for resource_config configuration"""
    
    @pytest.fixture
    def mock_provider_setup(self):
        """Mock provider setup for agent initialization"""
        provider = Mock()
        provider.initialize = AsyncMock()
        provider.config = Mock(default_model="gpt-3.5-turbo")
        provider.agent_instance = None
        
        embedding_provider = Mock()
        embedding_provider.initialize = AsyncMock()
        embedding_provider.config = Mock(default_model="text-embedding-ada-002")
        
        manager = Mock()
        
        def get_provider_side_effect(provider_id):
            if provider_id == "azure_openai_cc":
                return {
                    "class": Mock(return_value=provider),
                    "config_class": Mock
                }
            elif provider_id == "ollama":
                return {
                    "class": Mock(return_value=embedding_provider),
                    "config_class": Mock
                }
            else:
                raise ValueError(f"Unknown provider: {provider_id}")
        
        manager.get_provider.side_effect = get_provider_side_effect
        
        return provider, embedding_provider, manager
    
    @pytest.mark.asyncio
    async def test_resource_config_embedding_provider_selection(self, mock_provider_setup):
        """Test that embedding provider is selected from resource_config.knowledge_base"""
        provider, embedding_provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Configuration with embedding provider in resource_config
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "embedding_provider": "ollama",
                    "embedding_model": "custom-embedding",
                    "chunk_size": 1200,
                    "chunk_overlap": 150
                }
            }
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    await agent.setup_providers()
                    
                    # Main provider should be azure_openai_cc
                    assert agent.provider == provider
                    
                    # Embedding provider should be ollama (from resource_config)
                    assert agent.embedding_provider == embedding_provider
                    
                    # Both providers should be initialized
                    provider.initialize.assert_called_once()
                    embedding_provider.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resource_config_embedding_model_selection(self, mock_provider_setup):
        """Test that embedding model is selected from resource_config.knowledge_base"""
        provider, _, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "embedding_model": "text-embedding-3-large",
                    "chunk_size": 800
                }
            }
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                        mock_kb = Mock()
                        mock_kb.initialize = AsyncMock()
                        mock_kb.set_chat_provider = Mock()
                        mock_kb.set_embedding_provider = Mock()
                        mock_kb_class.return_value = mock_kb
                        
                        await agent.setup_providers()
                        kb = await agent.create_knowledge_base()
                        
                        # Should use embedding model from resource_config
                        mock_kb.set_embedding_provider.assert_called_once_with(
                            agent.embedding_provider,
                            "text-embedding-3-large"
                        )
    
    @pytest.mark.asyncio
    async def test_resource_config_memory_compression_settings(self, mock_provider_setup):
        """Test that memory compression settings are read from resource_config"""
        provider, _, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "resources": ["memory"],
            "resource_config": {
                "memory": {
                    "compression": {
                        "threshold_tokens": 5000,
                        "recent_messages_to_keep": 8,
                        "enabled": True,
                        "archive_conversations": False
                    }
                }
            }
        }
        
        mock_memory = Mock()
        mock_memory.initialize = AsyncMock()
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch.object(agent, 'create_memory', AsyncMock(return_value=mock_memory)):
                        await agent.initialize()
                        
                        # Test that compression config is read correctly
                        compression_config = agent._get_resource_config("memory").get("compression", {})
                        
                        assert compression_config["threshold_tokens"] == 5000
                        assert compression_config["recent_messages_to_keep"] == 8
                        assert compression_config["enabled"] == True
                        assert compression_config["archive_conversations"] == False
    
    @pytest.mark.asyncio
    async def test_resource_config_fallback_to_top_level(self, mock_provider_setup):
        """Test fallback to top-level config when resource_config doesn't have settings"""
        provider, _, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Top-level embedding_provider but not in resource_config
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "embedding_provider": "ollama",  # Top-level
            "embedding_model": "top-level-model",  # Top-level
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "chunk_size": 800
                    # No embedding_provider or embedding_model here
                }
            }
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                        mock_kb = Mock()
                        mock_kb.initialize = AsyncMock()
                        mock_kb.set_chat_provider = Mock()
                        mock_kb.set_embedding_provider = Mock()
                        mock_kb_class.return_value = mock_kb
                        
                        await agent.setup_providers()
                        kb = await agent.create_knowledge_base()
                        
                        # Should fall back to top-level embedding_model
                        mock_kb.set_embedding_provider.assert_called_once_with(
                            agent.embedding_provider,
                            "top-level-model"
                        )
    
    @pytest.mark.asyncio
    async def test_resource_config_complete_knowledge_base_configuration(self, mock_provider_setup):
        """Test complete knowledge base configuration from resource_config"""
        provider, embedding_provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Complete resource_config for knowledge base
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "embedding_provider": "ollama",
                    "embedding_model": "nomic-embed-text",
                    "vector_provider": "pgvector",
                    "chunk_size": 1200,
                    "chunk_overlap": 200,
                    "chunking": {
                        "markdown": "semantic",
                        "text": "simple",
                        "json": "token_aware"
                    },
                    "rerank_limit": 30
                }
            }
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
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
                            
                            await agent.setup_providers()
                            kb = await agent.create_knowledge_base()
                            
                            # Verify KnowledgeBaseResource was created with correct config
                            call_args = mock_kb_class.call_args
                            config_arg = call_args[0][1]
                            
                            # Check all resource_config values were passed
                            assert config_arg["chunk_size"] == 1200
                            assert config_arg["chunk_overlap"] == 200
                            assert config_arg["rerank_limit"] == 30
                            assert config_arg["chunking"]["markdown"] == "semantic"
                            assert config_arg["chunking"]["text"] == "simple"
                            assert config_arg["chunking"]["json"] == "token_aware"
                            
                            # Check providers were set correctly
                            mock_kb.set_chat_provider.assert_called_once_with(provider)
                            mock_kb.set_embedding_provider.assert_called_once_with(
                                embedding_provider,
                                "nomic-embed-text"
                            )
    
    @pytest.mark.asyncio
    async def test_resource_config_mixed_memory_and_knowledge_base(self, mock_provider_setup):
        """Test agent with both memory and knowledge base using resource_config"""
        provider, embedding_provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Configuration with both resources
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "resources": ["memory", "knowledge_base"],
            "resource_config": {
                "memory": {
                    "compression": {
                        "threshold_tokens": 8000,
                        "recent_messages_to_keep": 15,
                        "enabled": True
                    }
                },
                "knowledge_base": {
                    "embedding_provider": "ollama",
                    "embedding_model": "custom-embedding",
                    "chunk_size": 600,
                    "chunk_overlap": 50
                }
            }
        }
        
        mock_memory = Mock()
        mock_memory.initialize = AsyncMock()
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch.object(agent, 'create_memory', AsyncMock(return_value=mock_memory)):
                        with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                            mock_kb = Mock()
                            mock_kb.initialize = AsyncMock()
                            mock_kb.set_chat_provider = Mock()
                            mock_kb.set_embedding_provider = Mock()
                            mock_kb_class.return_value = mock_kb
                            
                            await agent.initialize()
                            
                            # Both resources should be created
                            assert agent.memory == mock_memory
                            assert agent.knowledge_base == mock_kb
                            
                            # Verify memory config
                            memory_config = agent._get_resource_config("memory")
                            assert memory_config["compression"]["threshold_tokens"] == 8000
                            assert memory_config["compression"]["recent_messages_to_keep"] == 15
                            
                            # Verify knowledge base config
                            kb_config = agent._get_resource_config("knowledge_base")
                            assert kb_config["embedding_provider"] == "ollama"
                            assert kb_config["embedding_model"] == "custom-embedding"
                            assert kb_config["chunk_size"] == 600
    
    @pytest.mark.asyncio
    async def test_resource_config_empty_sections(self, mock_provider_setup):
        """Test behavior when resource_config sections are empty"""
        provider, _, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Configuration with empty resource_config sections
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "embedding_provider": "azure_openai_cc",
            "embedding_model": "text-embedding-ada-002",
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {},  # Empty section
                "memory": {}  # Empty section
            }
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                        mock_kb = Mock()
                        mock_kb.initialize = AsyncMock()
                        mock_kb.set_chat_provider = Mock()
                        mock_kb.set_embedding_provider = Mock()
                        mock_kb_class.return_value = mock_kb
                        
                        await agent.setup_providers()
                        kb = await agent.create_knowledge_base()
                        
                        # Should fall back to top-level config
                        mock_kb.set_embedding_provider.assert_called_once_with(
                            agent.embedding_provider,
                            "text-embedding-ada-002"
                        )
    
    @pytest.mark.asyncio
    async def test_resource_config_precedence_over_top_level(self, mock_provider_setup):
        """Test that resource_config takes precedence over top-level config"""
        provider, embedding_provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Configuration with conflicting values
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "embedding_provider": "azure_openai_cc",  # Top-level
            "embedding_model": "text-embedding-ada-002",  # Top-level
            "resources": ["knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "embedding_provider": "ollama",  # Override
                    "embedding_model": "custom-model",  # Override
                    "chunk_size": 1000
                }
            }
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch('app.core.resources.knowledge_base.KnowledgeBaseResource') as mock_kb_class:
                        mock_kb = Mock()
                        mock_kb.initialize = AsyncMock()
                        mock_kb.set_chat_provider = Mock()
                        mock_kb.set_embedding_provider = Mock()
                        mock_kb_class.return_value = mock_kb
                        
                        await agent.setup_providers()
                        kb = await agent.create_knowledge_base()
                        
                        # Should use resource_config values, not top-level
                        assert agent.embedding_provider == embedding_provider  # ollama
                        mock_kb.set_embedding_provider.assert_called_once_with(
                            embedding_provider,
                            "custom-model"  # From resource_config
                        )
    
    def test_resource_config_missing_entirely(self):
        """Test behavior when resource_config is missing entirely"""
        agent = BaseAgent("test_agent")
        
        # Configuration without resource_config
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "embedding_provider": "azure_openai_cc",
            "embedding_model": "text-embedding-ada-002"
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            # Should return empty dict for any resource
            memory_config = agent._get_resource_config("memory")
            kb_config = agent._get_resource_config("knowledge_base")
            
            assert memory_config == {}
            assert kb_config == {}
    
    @pytest.mark.asyncio
    async def test_resource_config_complex_nested_structure(self, mock_provider_setup):
        """Test complex nested resource_config structure"""
        provider, _, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Complex nested configuration
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "resources": ["memory", "knowledge_base"],
            "resource_config": {
                "memory": {
                    "compression": {
                        "threshold_tokens": 12000,
                        "recent_messages_to_keep": 20,
                        "enabled": True,
                        "strategies": {
                            "summarization": "gpt-4",
                            "chunking": "semantic"
                        }
                    },
                    "persistence": {
                        "ttl_hours": 168,
                        "auto_cleanup": True
                    }
                },
                "knowledge_base": {
                    "embedding_provider": "azure_openai_cc",
                    "embedding_model": "text-embedding-3-large",
                    "vector_provider": "pgvector",
                    "search_settings": {
                        "similarity_threshold": 0.7,
                        "max_results": 10,
                        "use_reranking": True
                    },
                    "chunking": {
                        "markdown": "semantic",
                        "text": "simple",
                        "json": "token_aware",
                        "pdf": "document_specific"
                    }
                }
            }
        }
        
        with patch.object(agent.tool_manager, 'config', mock_config):
            # Test nested access
            memory_config = agent._get_resource_config("memory")
            kb_config = agent._get_resource_config("knowledge_base")
            
            # Verify nested memory structure
            assert memory_config["compression"]["threshold_tokens"] == 12000
            assert memory_config["compression"]["strategies"]["summarization"] == "gpt-4"
            assert memory_config["persistence"]["ttl_hours"] == 168
            
            # Verify nested knowledge base structure
            assert kb_config["search_settings"]["similarity_threshold"] == 0.7
            assert kb_config["chunking"]["markdown"] == "semantic"
            assert kb_config["chunking"]["pdf"] == "document_specific"