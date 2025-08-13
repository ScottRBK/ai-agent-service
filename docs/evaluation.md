# Evaluation Framework

## Overview

The AI Agent Service includes a comprehensive evaluation framework built on [DeepEval](https://docs.confident-ai.com/) that enables systematic assessment of agent performance. This framework supports synthetic test data generation, multiple evaluation metrics, and detailed result analysis to ensure agents perform reliably and accurately.

## Architecture

The evaluation framework consists of several key components:

### Core Components

- **`config.py`** - Pydantic models for evaluation configuration
  - `EvaluationConfig`: Complete evaluation setup including agent, metrics, and datasets
  - `SynthesizerConfig`: Configuration for synthetic data generation
  - `ContextWithMetadata`: Context data paired with expected tool usage
  - `GoldenGenerationType`: Enum supporting DOCUMENT, CONTEXT, SCRATCH, KNOWLEDGE_BASE options

- **`runner.py`** - Main evaluation execution engine
  - Handles the complete evaluation workflow
  - Manages golden dataset generation and loading
  - Executes agent interactions and collects results
  - Provides result analysis and reporting

- **`dataset.py`** - Golden dataset management
  - Creates and manages synthetic test cases
  - Supports serialization/deserialization of test data
  - Integrates with DeepEval's synthesizer for data generation
  - Document-based golden generation from frontmatter metadata

- **`document_processor.py`** - Document processing for evaluations
  - Extends production DocumentLoader for evaluation-specific path resolution
  - Minimal wrapper with clean inheritance pattern

- **`evaluation_utils.py`** - Result analysis utilities
  - Formats and displays evaluation summaries
  - Provides both standard and verbose output modes
  - Calculates metrics and pass rates

- **`evals/`** - Agent-specific evaluation configurations
  - Contains evaluation setups for different agents
  - Each file defines its own metrics and test contexts
  - Examples:
    - `cli_agent.py` - Comprehensive CLI agent evaluation
    - `simple_eval.py` - Basic evaluation with tool correctness and contextual relevancy
    - `temporal_awareness.py` - Time-based information and current event evaluation

## Creating Agent Evaluations

To create an evaluation for a specific agent, follow this pattern (using `cli_agent.py` as an example):

```python
from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import ToolCorrectnessMetric, GEval, HallucinationMetric
from deepeval.models import OllamaModel
from app.evaluation.config import EvaluationConfig, SynthesizerConfig, ContextWithMetadata
from app.evaluation.runner import EvaluationRunner

def create_evaluation_config() -> EvaluationConfig:
    # 1. Define the evaluation model
    model = OllamaModel(model="mistral:7b", temperature=0.0)
    
    # 2. Configure synthetic data generation
    styling_config = StylingConfig(
        scenario="User asking questions that require specific tools",
        task="Generate queries that clearly indicate which tool to use",
        input_format="Natural language queries",
        expected_output_format="Helpful responses using tool information"
    )
    
    # 3. Define test contexts with expected tools
    contexts = [
        ContextWithMetadata(
            context=["Search results show GPT-4 was released in March 2023..."],
            tools=["searxng__searxng_web_search"]
        )
    ]
    
    # 4. Configure evaluation metrics
    metrics = [
        ToolCorrectnessMetric(),  # Validates correct tool usage
        HallucinationMetric(threshold=0.5, model=model),  # Detects false information
        GEval(  # Custom metric for coherence
            name="coherence",
            criteria="Is the response coherent and well-structured?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model
        )
    ]
    
    # 5. Return complete configuration
    return EvaluationConfig(
        agent_id="cli_agent",
        synthesizer_config=SynthesizerConfig(
            model=model,
            styling_config=styling_config,
            max_goldens_per_context=2
        ),
        metrics=metrics,
        contexts=contexts,
        dataset_name="cli_agent",
        dataset_file="cli_agent_goldens.pkl",
        results_file="cli_agent_results"
    )
```

## Document-Based Golden Generation

The evaluation framework supports generating test cases directly from documents with YAML frontmatter metadata. This enables creating evaluations from existing documentation or structured content.

### Configuration

Set `golden_generation_type` to `GoldenGenerationType.DOCUMENT` in your evaluation config:

```python
from app.evaluation.config import EvaluationConfig, GoldenGenerationType

config = EvaluationConfig(
    agent_id="knowledge_agent",
    golden_generation_type=GoldenGenerationType.DOCUMENT,
    input_documents_dir="path/to/documents",
    # ... other config
)
```

### Document Format

Documents should include YAML frontmatter with `contexts_with_metadata` for explicit test control:

```markdown
---
contexts_with_metadata:
  - context: "Information about API authentication"
    tools: ["knowledge_base__search_documents"]
    persist_to_kb: true
  - context: "Database configuration details"
    tools: ["knowledge_base__search_documents"]
    persist_to_kb: false
---

# Document Content

This document contains information that will be used to generate test cases...
```

### Features

- **Frontmatter Control**: Use `contexts_with_metadata` to define expected test behavior
- **RAG Testing**: Optional `persist_to_kb` flag for knowledge base integration testing
- **File Type Support**: Works with 30+ file types including markdown, text, JSON, and code files
- **Metadata Extraction**: Automatic extraction from YAML frontmatter and JSON metadata

## Available Metrics

The framework supports various DeepEval metrics:

### Tool Correctness
Validates that agents use the appropriate tools for given tasks:
```python
ToolCorrectnessMetric(threshold=0.5, include_reason=True)
```

### Hallucination Detection
Measures factual accuracy against provided context:
```python
HallucinationMetric(threshold=0.5, model=model)
```

### Answer Relevancy
Evaluates how well responses address the input query:
```python
AnswerRelevancyMetric(threshold=0.7, model=model)
```

### Custom GEval Metrics
Create custom criteria-based evaluations:
```python
GEval(
    name="format_compliance",
    criteria="Does the response follow the expected API format?",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=model
)
```

## Available Evaluation Scripts

### CLI Agent Evaluation (`cli_agent.py`)
Comprehensive evaluation of the CLI agent with tool correctness, hallucination detection, and coherence metrics.

```bash
# Generate golden test cases
python app/evaluation/evals/cli_agent.py --generate

# Run evaluation
python app/evaluation/evals/cli_agent.py --verbose
```

### Simple Evaluation (`simple_eval.py`)
Basic evaluation focusing on tool correctness and contextual relevancy for straightforward queries.

```bash
# Generate test cases
python app/evaluation/evals/simple_eval.py --generate

# Run evaluation with detailed output
python app/evaluation/evals/simple_eval.py --verbose
```

**Key Features:**
- Tool correctness validation
- Contextual relevancy scoring (threshold: 0.7)
- Tests current information retrieval (e.g., "who won the premier league in 2025")
- Uses Ollama models for evaluation

### Document-Based Evaluation (`simple_doc.py`)
Example of document-based evaluation using frontmatter contexts for controlled test generation.

```bash
# Generate test cases from documents
python app/evaluation/evals/simple_doc.py --generate

# Run document-based evaluation
python app/evaluation/evals/simple_doc.py --verbose
```

**Key Features:**
- Document-based golden generation with frontmatter metadata
- Knowledge base integration testing with RAG metrics
- Contextual precision, recall, relevancy, and faithfulness evaluation
- Example of `contexts_with_metadata` usage for explicit test control

### Temporal Awareness Evaluation (`temporal_awareness.py`)
Evaluates agent's ability to handle time-based queries and current event information.

```bash
# Generate temporal test cases
python app/evaluation/evals/temporal_awareness.py --generate

# Run temporal awareness evaluation
python app/evaluation/evals/temporal_awareness.py --verbose
```

**Key Features:**
- Tests understanding of temporal context
- Validates current date/time awareness
- Evaluates recent event knowledge
- Ensures accurate time-based information retrieval

### Knowledge Agent Evaluation (`knowledge_agent.py`)
Evaluates agent's knowledge base integration and RAG capabilities with structured namespace management.

```bash
# Generate knowledge base test cases
python app/evaluation/evals/knowledge_agent.py --generate

# Run knowledge agent evaluation
python app/evaluation/evals/knowledge_agent.py --verbose
```

**Key Features:**
- RAG-specific metrics: Contextual Precision, Contextual Recall, Contextual Relevancy, Faithfulness
- Knowledge base document ingestion and search testing with structured namespaces
- Multi-user and multi-namespace isolation testing
- Tool correctness for knowledge base operations
- Hallucination detection for retrieved information

## Running Evaluations

### Basic Usage

1. **Generate golden test cases** (first time only):
   ```bash
   python app/evaluation/evals/<evaluation_script>.py --generate
   ```

2. **Run evaluation** using existing golden dataset:
   ```bash
   python app/evaluation/evals/<evaluation_script>.py
   ```

3. **Run with verbose output** for detailed results:
   ```bash
   python app/evaluation/evals/<evaluation_script>.py --verbose
   ```

### Evaluation Workflow

The evaluation process follows these steps:

1. **Golden Generation** (if `--generate` flag is used)
   - Uses DeepEval's synthesizer to create test cases from contexts
   - Generates natural language queries that require specific tools
   - Saves golden dataset to `evaluations/goldens/` directory

2. **Agent Execution**
   - Loads golden test cases
   - Runs each test through the specified agent
   - Collects responses and tool usage information

3. **Metric Evaluation**
   - Applies configured metrics to each test case
   - Measures tool correctness, hallucination, relevancy, etc.
   - Generates scores and pass/fail results

4. **Result Analysis**
   - Saves detailed results to `evaluations/results/` with timestamps
   - Displays summary statistics and per-metric breakdowns
   - Provides verbose output option for debugging

## Understanding Results

### Standard Output
Shows high-level summary:
```
EVALUATION SUMMARY
==================

📊 OVERALL RESULTS:
  Total tests: 6
  Passed: 5 (83.3%)
  Failed: 1 (16.7%)

📈 METRIC BREAKDOWN:
  tool_correctness_metric:
    Pass rate: 6/6 (100.0%)
    Avg score: 1.000
    Threshold: 0.5
```

### Verbose Output
Provides detailed test-by-test results:
```
Test 1: What's the latest news about artificial intelligence?...
  Expected tools: ['searxng__searxng_web_search']
  Actual tools:   ['searxng__searxng_web_search']
  Metrics: ✅ToolCorrectness:1.0 | ✅Coherence:0.9 | ✅Hallucination:0.8
  Overall: ✅ PASSED
```

## Best Practices

1. **Context Design**: Create contexts that clearly indicate which tools should be used
2. **Metric Selection**: Choose metrics that align with your agent's purpose
3. **Golden Dataset Size**: Generate enough test cases for statistical significance (10-20 per context)
4. **Regular Evaluation**: Run evaluations after significant agent changes
5. **Result Tracking**: Save evaluation results for trend analysis

## Directory Structure

Evaluation outputs are organized as follows:
```
evaluations/
├── goldens/           # Synthetic test datasets
│   └── cli_agent_goldens.pkl
└── results/           # Evaluation results with timestamps
    └── cli_agent_results-20250130123456.pkl
```

## Integration with Agent Development

The evaluation framework integrates seamlessly with the agent architecture:

- Uses the same agent configurations from `agent_config.json`
- Leverages existing provider and tool systems
- Supports all agent types (CLI, API, etc.)
- Provides isolated test sessions to avoid state contamination

## Advanced Usage

### Custom Evaluation Models

#### Standard DeepEval Models
Use different models for evaluation:
```python
from deepeval.models import AzureOpenAI

eval_model = AzureOpenAI(
    deployment_name="gpt-4",
    azure_api_version="2024-02-01"
)
```

#### CustomOllamaModel with Instructor Integration
For evaluations requiring strict JSON schema compliance and structured outputs, use the `CustomOllamaModel`:

```python
from app.evaluation.custom_ollama import CustomOllamaModel

# Initialize with robust JSON output capabilities
eval_model = CustomOllamaModel(
    model="mistral:7b",
    base_url="http://localhost:11434",  # Automatically converts to OpenAI-compatible endpoint
    temperature=0.0  # For deterministic evaluation results
)
```

**When to Use CustomOllamaModel:**
- Evaluations that require structured JSON responses with strict schema adherence
- Metrics that need validated Pydantic model outputs
- Scenarios where evaluation consistency is critical and fallback behavior is preferred over failures
- Local evaluations where you want the reliability of instructor's JSON confinement

**Example with Schema Validation:**
```python
from pydantic import BaseModel
from typing import List

class EvaluationResponse(BaseModel):
    score: float
    reasoning: str
    criteria_met: List[str]

# The model will enforce this schema automatically
eval_model = CustomOllamaModel("qwen3:8b", temperature=0.0)
response = eval_model.generate(prompt, schema=EvaluationResponse)
# response is guaranteed to be a valid EvaluationResponse instance
```

**Note**: CustomOllamaModel is particularly useful for metrics that require structured outputs and provides more reliable JSON parsing compared to the standard OllamaModel, especially when working with local models that may have inconsistent JSON formatting.

### Batch Evaluations
Evaluate multiple agents:
```python
for agent_id in ["cli_agent", "api_agent", "research_agent"]:
    config = create_evaluation_config(agent_id)
    runner = EvaluationRunner(config)
    await runner.run()
```

### Custom Result Processing
Access raw evaluation data:
```python
results = await runner.run()
df = results['dataframe']  # Pandas DataFrame for custom analysis
raw = results['raw_results']  # Complete DeepEval output
```