yo#!/usr/bin/env python3
import pickle
import sys
import pandas as pd

def show_results(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Handle direct DataFrame or dict with dataframe
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict) and 'dataframe' in data:
        df = data['dataframe']
    else:
        print(f"Unexpected data type: {type(data)}")
        return
    
    print(f"\n=== Knowledge Agent Evaluation Results ===")
    print(f"Total tests: {len(df)}")
    
    if 'success' in df.columns:
        print(f"Success rate: {df['success'].mean():.1%}")
    
    # Get metric columns
    metrics = [col for col in df.columns if '_score' in col or '_metric' in col or col.endswith('_metric')]
    print("\nMetrics:")
    for metric in sorted(metrics):
        if metric in df.columns:
            scores = df[metric].dropna()
            if len(scores) > 0:
                print(f"  {metric}: {scores.mean():.3f} (n={len(scores)})")
    
    # Show column names for debugging
    print(f"\nAll columns: {list(df.columns)}")
    
    # Show sample test cases
    print("\nSample test cases:")
    for i in range(min(3, len(df))):
        print(f"\nTest {i+1}:")
        row = df.iloc[i]
        if 'input' in row:
            print(f"  Input: {str(row['input'])[:80]}...")
        if 'actual_output' in row:
            print(f"  Output: {str(row['actual_output'])[:80]}...")

if __name__ == "__main__":
    show_results(sys.argv[1] if len(sys.argv) > 1 else "evaluations/output/results/knowledge_agent_results-20250804003917.pkl")