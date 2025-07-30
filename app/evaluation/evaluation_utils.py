import pandas as pd

class EvaluationUtils:
    def __init__(self):
        pass
    @staticmethod
    def print_evaluation_summary(df: pd.DataFrame, results_json: dict) -> None:
        """Print a comprehensive summary of the evaluation"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Overall stats
        total_tests = len(results_json.get('test_results', []))
        passed_tests = sum(1 for t in results_json.get('test_results', []) if t['success'])
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"  Total tests: {total_tests}")
        if total_tests == 0:
            print("\n NO TESTS FOUND")
            return
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

    @staticmethod
    def print_evaluation_summary_verbose(df: pd.DataFrame) -> None:
        """Simple summary using the DataFrame"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80 + "\n")
        if df.empty or 'test_name' not in df.columns:
            print("No test results to display.")
            print("TOTAL: 0/0 passed (0%)")
            return
        
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