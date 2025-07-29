from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToolCorrectnessMetric, GEval, HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import ToolCall, LLMTestCaseParams
from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.dataset import EvaluationDataset, Golden

from app.core.agents.cli_agent import CLIAgent
from app.utils.chat_utils import clean_response_for_memory

from typing import Dict, Any, List
from contextlib import redirect_stdout, redirect_stderr

import asyncio
import argparse
import pandas as pd
import json
import io


dataset_file = "synthesizer_goldens_with_multiple_evals.pkl"

model = OllamaModel(model="mistral:7b", temperature=0.0)
# 1. Create contexts that represent tool responses
contexts_with_metadata = [
    {
        "context": ["Search results show that OpenAI released GPT-4 in March 2023, with significant improvements in reasoning and reduced hallucinations."],
        "tools": ["searxng__searxng_web_search"],
    },
    {
        "context": ["GitHub search found 15 repositories related to MCP implementation, with fastmcp being the most popular Python library."],
        "tools": ["github__search_repositories"],
    },
    {
        "context": ["The DeepWiki page for ScottRBK/ai-agent-service shows it's a production-ready AI agent framework with MCP (Model Context Protocol) integration."],
        "tools": ["deepwiki__read_wiki_contents"]
    }
]

styling_config = StylingConfig(
    scenario="User asking questions that require specific tools",
    task="Generate queries that clearly indicate which tool to use",
    input_format="Generate natural questions that would require tool usage",
    expected_output_format="A helpful response using information from the appropriate tool"
)

# 2. Configure synthesizer to generate tool-specific queries
styling_config = StylingConfig(
    scenario="User asking questions that require specific tools",
    task="Generate queries that clearly indicate which tool to use",
    input_format="""
    - Search queries: "What's the latest news about X?", "Find information about Y", "Search for Z"
    - GitHub queries: "Show me repositories about X", "Find GitHub projects for Y", "List repos related to Z"
    - Deepwiki queries: "What does the wiki say about X?", "Tell me about the Y project on DeepWiki", "Read the DeepWiki page for Z"
    """,
    expected_output_format="A helpful response using information from the appropriate tool"
)

def create_metrics():
    """Create all metrics in one place"""
    return [
        ToolCorrectnessMetric(),
        GEval(
            name="coherence",
            criteria="Is the response coherent and well-structured?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model
        ),
        HallucinationMetric(threshold=0.5, model=model),
        AnswerRelevancyMetric(threshold=0.7, model=model)
    ]

async def run_agent_and_create_test_case(golden: Golden) -> LLMTestCase:
    """Run agent once and create test case"""
    agent = CLIAgent("azure_agent")
    await agent.initialize()
    agent.provider.config.track_tool_calls = True

    response = await agent.chat(golden.input)
    cleaned_response = clean_response_for_memory(response)

    tools_called = [ToolCall(name=tool["tool_name"]) for tool in agent.provider.get_tool_calls_made()]

    return LLMTestCase(
        input=golden.input,
        expected_tools=golden.expected_tools,
        expected_output=golden.expected_output,
        actual_output=cleaned_response,
        tools_called=tools_called,
        context=[golden.context] if hasattr(golden, "context") else [],
        additional_metadata={
            "expected_tool_names": [t.name for t in golden.expected_tools] if golden.expected_tools else [],
            "actual_tool_names": [t.name for t in tools_called] if tools_called else []
        }
    )

async def generate_tool_goldens() -> list[Golden]:

    synthesizer = Synthesizer(model=model, styling_config=styling_config)
    goldens_to_return = []

    for context in contexts_with_metadata:
        print(f"\nGenerating goldens for tool: {context['tools']}")

        goldens = await synthesizer.a_generate_goldens_from_contexts(
            contexts=[context["context"]],
            include_expected_output=True,
            max_goldens_per_context=2
        )

        for i, base_golden in enumerate(goldens):
            print(f"  Golden {i+1}: {base_golden.input[:50]}...")
            
            exp_tools = []

            for tool in context["tools"]:
                exp_tools.append(ToolCall(name=tool))
                
            base_golden.expected_tools = exp_tools
            base_golden.context = context["context"][0]

        goldens_to_return.extend(goldens)
        print(f"\nTotal goldens generated: {len(goldens_to_return)}")
    return goldens_to_return

async def get_tool_goldens(generate_goldens: bool) -> EvaluationDataset:
    dataset = EvaluationDataset()
    if generate_goldens:
        goldens = await generate_tool_goldens()
        dataset.goldens = goldens
        await save_dataset_with_tools(dataset, dataset_file)
    else:
        goldens = await load_dataset_with_tools(dataset_file)
        dataset.goldens = goldens
    return dataset

async def save_dataset_with_tools(dataset: EvaluationDataset, filename: str):
    """Save dataset using pandas pickle - preserves everything perfectly"""
    data = [{
        'input': g.input,
        'expected_output': g.expected_output,
        'expected_tools': g.expected_tools if hasattr(g, 'expected_tools') else [],
        "context": g.context if hasattr(g, "context") else []
    } for g in dataset.goldens]
    
    df = pd.DataFrame(data)
    df.to_pickle(filename)
    print(f"Saved {len(df)} goldens to {filename}")

async def load_dataset_with_tools(filename: str) -> List[Golden]:
    """Load dataset using pandas pickle"""
    df = pd.read_pickle(filename)
    
    goldens = []
    for _, row in df.iterrows():
        golden = Golden(
            input=row['input'],
            expected_output=row['expected_output']
        )
        golden.expected_tools = row['expected_tools']
        golden.context = row['context']
        goldens.append(golden)
    print(f"Loaded {len(goldens)} goldens from file {filename}")
    return goldens

async def main(generate_goldens=False, print_verbose=False):

    dataset = await get_tool_goldens(generate_goldens)
    print(f"Loaded {len(dataset.goldens)} goldens")

    test_cases = []
    
    for i, golden in enumerate(dataset.goldens):
        print(f"Running test {i+1}/{len(dataset.goldens)}: {golden.input[:60]}...")

        test_case = await run_agent_and_create_test_case(golden)
        
        test_cases.append(test_case)

    metrics = create_metrics()

    print("\nEvaluating...")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )
    print("Evaluation complete")



    results_json = json.loads(results.model_dump_json())

    df = create_evaluation_dataframe(results_json)

    if print_verbose:
        print_evaluation_summary_verbose(df)
    else:
        print_evaluation_summary(df)

    df.to_pickle("synthesizer_with_multiple_evals_results.pkl")

def create_evaluation_dataframe(results_json):
    """Convert DeepEval results JSON to a comprehensive DataFrame"""
    rows = []
    
    for test_result in results_json.get('test_results', []):

        metadata = test_result.get('additional_metadata', {})

        # Base information for each test
        base_info = {
            'test_name': test_result['name'],
            'overall_success': test_result['success'],
            'input': test_result['input'],
            'actual_output': test_result.get('actual_output', ''),
            'expected_output': test_result.get('expected_output', ''),
            'context': str(test_result.get('context', [])),
            'expected_tools': metadata.get('expected_tool_names', []),
            'actual_tools': metadata.get('actual_tool_names', []),
        }
        
        # Create a row for each metric result
        for metric in test_result['metrics_data']:
            row = base_info.copy()
            row.update({
                'metric_name': metric['name'],
                'metric_success': metric['success'],
                'metric_score': metric['score'],
                'metric_threshold': metric['threshold'],
                'metric_reason': metric['reason'],
                'evaluation_model': metric.get('evaluation_model', 'N/A'),
                'error': metric.get('error', None),
                'verbose_logs': metric.get('verbose_logs', '')
            })
            rows.append(row)
    
    return pd.DataFrame(rows)

def print_evaluation_summary(df, results_json):
    """Print a comprehensive summary of the evaluation"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Overall stats
    total_tests = len(results_json.get('test_results', []))
    passed_tests = sum(1 for t in results_json.get('test_results', []) if t['success'])
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"  Failed: {total_tests - passed_tests} ({(total_tests-passed_tests)/total_tests*100:.1f}%)")
    
    # Per-metric summary
    print("\n📈 METRIC BREAKDOWN:")
    metric_summary = df.groupby('metric_name').agg({
        'metric_success': ['sum', 'count'],
        'metric_score': ['mean', 'min', 'max']
    }).round(3)
    
    for metric in df['metric_name'].unique():
        metric_df = df[df['metric_name'] == metric]
        passed = metric_df['metric_success'].sum()
        total = len(metric_df)
        avg_score = metric_df['metric_score'].mean()
        
        print(f"\n  {metric}:")
        print(f"    Pass rate: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"    Avg score: {avg_score:.3f}")
        print(f"    Threshold: {metric_df['metric_threshold'].iloc[0]}")
    
    # Link to Confident AI (if available)
    if 'confident_link' in results_json:
        print(f"\n🔗 View in Confident AI: {results_json['confident_link']}")

def print_evaluation_summary_verbose(df):
    """Simple summary using the DataFrame"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    
    # Group by test to show each test once
    for test_name, group in df.groupby('test_name'):
        first_row = group.iloc[0]
        test_num = int(test_name.split('_')[-1]) + 1
        
        print(f"Test {test_num}: {first_row['input'][:60]}...")
        print(f"  Expected tools: {first_row['expected_tools']}")
        print(f"  Actual tools:   {first_row['actual_tools']}")
        
        # Collect all metrics for this test
        scores = []
        for _, metric_row in group.iterrows():
            status = "✅" if metric_row['metric_success'] else "❌"
            name = metric_row['metric_name'].split()[0]
            scores.append(f"{status}{name}:{metric_row['metric_score']:.1f}")
        
        print(f"  Metrics: {' | '.join(scores)}")
        print(f"  Overall: {'✅ PASSED' if first_row['overall_success'] else '❌ FAILED'}\n")
    
    # Summary
    unique_tests = df['test_name'].nunique()
    passed_tests = df.groupby('test_name')['overall_success'].first().sum()
    print("-" * 80)
    print(f"TOTAL: {passed_tests}/{unique_tests} passed ({passed_tests/unique_tests*100:.0f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthesizer with tools evaluation")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate golden test cases")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print out each test case and its results")
    
    args = parser.parse_args()
    
    asyncio.run(main(generate_goldens=args.generate, print_verbose=args.verbose))