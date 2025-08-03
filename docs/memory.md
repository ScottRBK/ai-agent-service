# Memory System Documentation

## Overview

The Agent Service includes a comprehensive memory system that provides persistent conversation history, intelligent compression, and cross-session context retrieval. The memory system is built on PostgreSQL and integrates seamlessly with the knowledge base for enhanced context awareness.

### Key Features

- **PostgreSQL-based persistence** - Conversation history stored in PostgreSQL with session isolation
- **AI-powered compression** - Automatic summarization when conversations exceed token thresholds
- **Cross-session context** - Intelligent retrieval of relevant context from past conversations
- **Knowledge base archival** - Compressed conversations archived for future reference
- **Graceful degradation** - Memory features fail gracefully when resources are unavailable

### Architecture Components

- **PostgreSQL Memory Resource** - Core persistence layer with PostgreSQL backend
- **Memory Compression Agent** - AI-powered conversation summarization and compression
- **Memory Compression Manager** - Token counting and compression logic
- **Knowledge Base Integration** - Archival and cross-session context retrieval
- **Base Agent Memory Methods** - Unified memory interface across all agent types

## Configuration

Memory functionality is configuration-driven and activates only when explicitly configured in the agent's `resources` array.

### Basic Memory Configuration

```json
{
  "agent_id": "memory_enabled_agent",
  "resources": ["memory"],
  "provider": "azure_openai_cc",
  "model": "gpt-4o-mini"
}
```

### Advanced Memory with Compression

```json
{
  "agent_id": "advanced_memory_agent",
  "resources": ["memory"],
  "provider": "azure_openai_cc",
  "model": "gpt-4o-mini",
  "resource_config": {
    "memory": {
      "compression": {
        "enabled": true,
        "threshold_tokens": 8000,
        "recent_messages_to_keep": 10,
        "archive_conversations": false
      }
    }
  }
}
```

### Memory with Knowledge Base Archival

```json
{
  "agent_id": "knowledge_memory_agent",
  "resources": ["memory", "knowledge_base"],
  "provider": "azure_openai_cc",
  "model": "gpt-4o-mini",
  "allowed_regular_tools": ["search_knowledge_base", "list_documents"],
  "resource_config": {
    "memory": {
      "compression": {
        "enabled": true,
        "threshold_tokens": 8000,
        "recent_messages_to_keep": 10,
        "archive_conversations": true
      }
    },
    "knowledge_base": {
      "vector_provider": "pgvector",
      "chunk_size": 800,
      "chunk_overlap": 100,
      "embedding_provider": "azure_openai_cc",
      "embedding_model": "text-embedding-ada-002",
      "enable_cross_session_context": true,
      "relevance_threshold": 0.7,
      "search_triggers": [
        "what did we discuss",
        "previous conversation", 
        "last time",
        "remind me",
        "we mentioned",
        "earlier you said"
      ]
    }
  }
}
```

### Configuration Options

#### Memory Resource Configuration

- **`compression.enabled`** (boolean, default: true) - Enable/disable memory compression
- **`compression.threshold_tokens`** (integer, default: 10000) - Token threshold for triggering compression
- **`compression.recent_messages_to_keep`** (integer, default: 10) - Number of recent messages to preserve during compression
- **`compression.archive_conversations`** (boolean, default: false) - Archive compressed conversations to knowledge base

#### Knowledge Base Cross-Session Configuration

- **`enable_cross_session_context`** (boolean, default: false) - Enable automatic cross-session context retrieval
- **`relevance_threshold`** (float, default: 0.7) - Minimum relevance score for cross-session results
- **`search_triggers`** (array of strings) - Phrases that trigger cross-session context search
- **`rerank_provider`** (string, optional) - Provider for result reranking (experimental)
- **`rerank_model`** (string, optional) - Model for reranking operations

## How Memory Works

### Memory Lifecycle

1. **Message Storage** - Every conversation message is stored as a `MemoryEntry` in PostgreSQL
2. **Session Isolation** - Messages are organized by `user_id`, `session_id`, and `agent_id`
3. **Automatic Loading** - Memory is automatically loaded when agents initialize conversations
4. **Compression Monitoring** - Token counts are tracked to determine when compression is needed
5. **Archival** - Compressed conversations can be archived to the knowledge base for cross-session access

### MemoryEntry Format

```python
class MemoryEntry:
    id: str                    # Unique identifier
    user_id: str              # User identifier  
    session_id: str           # Session identifier
    agent_id: str             # Agent identifier
    content: Dict[str, str]   # {"role": "user/assistant", "content": "message"}
    entry_metadata: Dict      # Optional metadata
    created_at: datetime      # Creation timestamp
    updated_at: datetime      # Last update timestamp
    expires_at: datetime      # Optional expiration (default: 1 week)
    is_active: bool           # Soft delete flag
```

### Session Summaries

When compression occurs, a session summary is created and stored:

```python
class MemorySessionSummary:
    id: str           # Unique identifier
    user_id: str      # User identifier
    session_id: str   # Session identifier  
    agent_id: str     # Agent identifier
    summary: str      # AI-generated summary
    created_at: datetime
    updated_at: datetime
```

## Memory Compression

The memory compression system automatically manages conversation history to prevent context window overflow while preserving important information.

### Compression Process

1. **Threshold Detection** - Token counting determines when compression is needed
2. **Message Splitting** - Conversation is split into older messages (to compress) and recent messages (to preserve)
3. **AI Summarization** - Memory Compression Agent creates a narrative summary of older messages
4. **Summary Storage** - Summary is stored as a session summary
5. **Message Cleanup** - Older messages are soft-deleted from the database
6. **Optional Archival** - If configured, compressed conversations are archived to knowledge base

### Compression Configuration

```json
{
  "compression": {
    "enabled": true,
    "threshold_tokens": 8000,
    "recent_messages_to_keep": 10,
    "archive_conversations": true
  }
}
```

### Compression Logic

The `MemoryCompressionManager` handles compression logic:

- **Token Counting** - Uses tiktoken to count conversation tokens
- **Smart Splitting** - Preserves recent messages while compressing older ones
- **Metadata Extraction** - Extracts topics, entities, decisions, and questions when archiving
- **Error Handling** - Graceful degradation if compression fails

### Enhanced Summarization with Metadata

When `archive_conversations` is enabled, the compression agent extracts structured metadata:

```
## SUMMARY
[Detailed narrative summary of the conversation]

## TOPICS  
[Comma-separated list of main topics discussed]

## ENTITIES
[Comma-separated list of key people, projects, technologies mentioned]

## DECISIONS
[List each decision on a new line, or "None" if no decisions were made]

## QUESTIONS
[List each unresolved question on a new line, or "None" if no questions remain]
```

## Knowledge Base Integration

When both memory and knowledge base resources are configured, the system provides powerful cross-session capabilities.

### Conversation Archival

Compressed conversations are automatically archived to the knowledge base with rich metadata:

```python
# Archived document structure
{
  "content": "# Conversation Summary\n## Date Range: ...\n## Summary: ...",
  "namespace": "conversations:{user_id}",
  "doc_type": "conversation", 
  "metadata": {
    "session_id": "session_123",
    "agent_id": "research_agent",
    "conversation_topics": ["authentication", "JWT", "security"],
    "entities_mentioned": ["John", "UserService", "Redis"],
    "decisions_made": ["Use JWT with 1-hour expiration"],
    "open_questions": ["Should we implement refresh tokens?"],
    "message_count": 25,
    "date_range": {
      "start": "2024-01-15T10:00:00Z",
      "end": "2024-01-15T11:30:00Z"
    }
  }
}
```

### Cross-Session Context Retrieval

The system automatically provides relevant context from past conversations:

1. **Trigger Detection** - Checks user messages for configured search triggers
2. **Vector Search** - Searches archived conversations using semantic similarity
3. **Relevance Filtering** - Filters results by relevance threshold and excludes current session
4. **Optional Reranking** - (Experimental) AI-powered reranking for improved relevance
5. **Context Injection** - Injects relevant context before recent messages

#### Experimental Reranking Feature

The knowledge base supports an experimental reranking capability that can improve search result relevance:

**Configuration Example:**
```json
{
  "knowledge_base": {
    "rerank_provider": "azure_openai_cc",
    "rerank_model": "gpt-4o-mini",
    "enable_cross_session_context": true
  }
}
```

**Limitations:**
- **Ollama Provider**: Currently has limited reranking functionality due to missing logits support in the Ollama API. Reranking with Ollama will use a workaround that may not provide optimal results.
- **Performance Impact**: Reranking adds an additional AI call, increasing latency and cost
- **Experimental Status**: This feature is under active development and behavior may change

**Recommended Usage:**
- Use Azure OpenAI or other providers with full logits support for best reranking results
- Consider the latency/accuracy tradeoff for your use case
- Monitor reranking performance and adjust relevance thresholds accordingly

### Search Triggers

Default phrases that trigger cross-session context retrieval:
- "what did we discuss"
- "previous conversation"
- "last time" 
- "remind me"
- "we mentioned"
- "earlier you said"

## API Usage

Memory works seamlessly with the API endpoints and is automatically handled by agents.

### Automatic Memory Loading

```python
# Memory is automatically loaded when agents process requests
conversation_history = await agent.load_memory()
# Returns: [{"role": "system", "content": "Summary..."}, 
#          {"role": "user", "content": "Hello"}, 
#          {"role": "assistant", "content": "Hi there!"}]
```

### Memory Persistence  

```python
# Memory is automatically saved after each exchange
await agent.save_memory("user", user_message)
response = await agent.chat(user_message)
await agent.save_memory("assistant", response)
```

### Clear Conversation

```python
# API agents support clearing conversation history
count = await agent.clear_conversation()
# Returns number of messages cleared
```

### Knowledge Base Tools

Agents with knowledge base access can search past conversations:

```python
# Available as function tools
await search_knowledge_base(
    query="authentication discussion",
    search_type="conversations", 
    limit=5
)

await list_documents(
    doc_type="conversations",
    limit=10  
)
```

## Examples

### Basic Memory-Enabled Agent

```python
from app.core.agents.api_agent import APIAgent

# Create agent with memory
agent = APIAgent("memory_agent", "user123", "session456")
await agent.initialize()

# Chat with persistent memory
response1 = await agent.chat("My name is Alice")
response2 = await agent.chat("What's my name?")
# Agent remembers "Alice" from previous message

await agent.cleanup()
```

### Memory with Compression

```python
# Configure agent with compression
agent_config = {
    "agent_id": "research_agent",
    "resources": ["memory"],
    "resource_config": {
        "memory": {
            "compression": {
                "enabled": True,
                "threshold_tokens": 5000,
                "recent_messages_to_keep": 8
            }
        }
    }
}

agent = APIAgent("research_agent", "user123", "long_session")
await agent.initialize()

# Have a long conversation
for i in range(20):
    await agent.chat(f"Research topic {i}")

# Compression happens automatically when threshold is reached
# Recent messages preserved, older messages summarized
```

### Cross-Session Memory

```python
# Session 1: Initial conversation
agent1 = APIAgent("knowledge_agent", "user123", "session1")
await agent1.initialize()

await agent1.chat("I'm building a REST API with authentication")
await agent1.chat("Planning to use JWT tokens with 1-hour expiration")
await agent1.cleanup()

# Session 2: Reference past conversation  
agent2 = APIAgent("knowledge_agent", "user123", "session2")
await agent2.initialize()

# This triggers cross-session context search
response = await agent2.chat("What did we discuss about authentication?")
# Agent automatically includes relevant context from session1

await agent2.cleanup()
```

### Custom Memory Configuration

```python
# Agent with custom compression and archival settings
memory_config = {
    "agent_id": "custom_agent",
    "resources": ["memory", "knowledge_base"],
    "resource_config": {
        "memory": {
            "compression": {
                "enabled": True,
                "threshold_tokens": 12000,
                "recent_messages_to_keep": 15,
                "archive_conversations": True
            }
        },
        "knowledge_base": {
            "enable_cross_session_context": True,
            "relevance_threshold": 0.8,
            "search_triggers": ["remind me", "we talked about", "previous session"]
        }
    }
}

agent = APIAgent("custom_agent", "user123", "session789")
await agent.initialize()
# Agent will compress at 12k tokens, keep 15 recent messages,
# archive to knowledge base, and provide cross-session context
```

## Implementation Notes

### Direct Resource Composition

The memory system uses direct resource composition rather than complex manager layers:

```python
class BaseAgent:
    def __init__(self, agent_id, user_id, session_id):
        # Direct composition - no managers
        self.memory = None
        self.knowledge_base = None
    
    async def initialize(self):
        if self.wants_memory():
            self.memory = await self.create_memory()
        if self.wants_knowledge_base():
            self.knowledge_base = await self.create_knowledge_base()
```

### Provider Dependency Injection

Knowledge base resources receive provider dependencies during creation:

```python
async def create_knowledge_base(self):
    kb = KnowledgeBaseResource(f"{self.agent_id}_kb", config)
    
    # Inject providers
    kb.set_chat_provider(self.provider)
    kb.set_embedding_provider(self.embedding_provider, embedding_model)
    if self.rerank_provider:
        kb.set_rerank_provider(self.rerank_provider, rerank_model)
    
    await kb.initialize()
    return kb
```

**Note on Reranking Providers:**
- The rerank provider is optional and only used when explicitly configured
- Azure OpenAI providers offer full reranking support with logits
- Ollama provider has limited reranking due to API constraints
- The system will gracefully handle missing rerank providers

### Error Handling

Memory operations include comprehensive error handling with graceful degradation:

```python
async def load_memory(self) -> List[Dict[str, str]]:
    try:
        # Load memory operations
        return conversation_history
    except Exception as e:
        logger.error(f"Error loading memory: {e}")
        return []  # Return empty history on error
```

### Database Schema

The memory system uses two main tables:

```sql
-- Memory entries table
CREATE TABLE memory_entries (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL, 
    agent_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    entry_metadata TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Session summaries table  
CREATE TABLE memory_session_summaries (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL, 
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## Performance Considerations

- **Token Counting** - Uses tiktoken for accurate token counting to manage context windows
- **Soft Deletes** - Uses `is_active` flag for soft deletes to maintain data integrity
- **Connection Pooling** - SQLAlchemy connection pooling for efficient database operations
- **Async Operations** - All memory operations are async for non-blocking I/O
- **Compression Thresholds** - Configurable thresholds prevent excessive compression overhead
- **Cross-Session Caching** - Knowledge base provides efficient vector search for past conversations

## Security Considerations

- **Session Isolation** - Memory is strictly isolated by user_id and session_id
- **PostgreSQL Security** - Leverages PostgreSQL's robust security features
- **Connection Security** - Uses psycopg3 with secure connection strings
- **Data Expiration** - Optional TTL for automatic data cleanup
- **Soft Deletes** - Maintains audit trail while removing from active use

## Troubleshooting

### Memory Not Loading

1. Check if `memory` is in the agent's `resources` array
2. Verify PostgreSQL connection settings
3. Check database initialization and table creation
4. Review agent logs for initialization errors

### Compression Not Working

1. Verify `compression.enabled` is true in resource_config
2. Check if conversation exceeds `threshold_tokens`
3. Ensure Memory Compression Agent can initialize
4. Review compression agent logs for errors

### Cross-Session Context Missing

1. Confirm `knowledge_base` resource is configured
2. Check `enable_cross_session_context` setting
3. Verify `archive_conversations` is enabled for archival
4. Ensure search triggers match user message patterns
5. Check relevance threshold settings

### Database Connection Issues

1. Verify PostgreSQL service is running
2. Check connection string format and credentials
3. Ensure database exists and is accessible
4. Review PostgreSQL logs for connection errors

## Migration and Upgrades

When upgrading existing agents to use the new memory system:

1. Add `memory` to the `resources` array in agent configuration
2. Add `resource_config.memory` section with desired compression settings
3. Ensure PostgreSQL is configured and accessible
4. Test with a development environment before production deployment

For knowledge base integration:

1. Add `knowledge_base` to resources array
2. Configure `resource_config.knowledge_base` with embedding settings
3. Add knowledge base tools to `allowed_regular_tools` if desired
4. Enable cross-session features as needed