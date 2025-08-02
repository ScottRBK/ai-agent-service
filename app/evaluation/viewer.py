import streamlit as st
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from app.config.settings import settings


class EvaluationViewer:
    """Main viewer class for evaluation results"""

    def __init__(self):
        self.evaluation_dir = settings.EVALUATION_OUTPUT_DIR
        self._setup_page_config()
        self._inject_custom_css()

    def _setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Agent Evaluation Results",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _inject_custom_css(self):
        """Add custom CSS styling"""
        st.markdown("""
        <style>
            .metric-card {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .pass { color: #28a745; font-weight: 500; }
            .fail { color: #dc3545; font-weight: 500; }
            .warning { color: #ffc107; font-weight: 500; }
            
            /* Improve table styling */
            [data-testid="stDataFrame"] {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            
            /* Details panel styling */
            div[data-testid="column"]:last-child {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                min-height: 700px;
            }
            
            /* Make columns equal height */
            div[data-testid="column"] {
                display: flex;
                flex-direction: column;
            }
            
            /* Compact tool section */
            .tool-section {
                font-size: 0.75rem;
                line-height: 1.1;
            }
            .tool-section b {
                font-size: 0.8rem;
            }
            .tool-section span {
                display: block;
                padding: 0;
                margin: 0;
                font-size: 0.7rem;
                line-height: 1.1;
            }
            .tool-section p {
                margin: 0 !important;
                padding: 0 !important;
                line-height: 1.1 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    def _get_file_time(self, filename):
        """Get formatted modification time for a file"""
        results_dir = self._get_results_dir()
        file_path = os.path.join(results_dir, filename)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return mod_time.strftime("%Y-%m-%d %H:%M")

    def _get_results_dir(self):
        """Get the results directory"""
        if os.path.isabs(self.evaluation_dir):
            return os.path.join(self.evaluation_dir, "results")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            return os.path.join(project_root, self.evaluation_dir, "results")

    def _get_evaluations(self):
        """Get all evaluation results from the results directory"""
        results_dir = self._get_results_dir()
        evaluations = []

        # Check if directory exists
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            return []

        try:
            for file in os.listdir(results_dir):
                if file.endswith(".pkl"):
                    file_path = os.path.join(results_dir, file)
                    mod_time = os.path.getmtime(file_path)
                    evaluations.append((file, mod_time))
        except Exception as e:
            st.error(f"Error reading evaluation files: {e}")
            return []

        evaluations.sort(key=lambda x: x[1], reverse=True)
        return [file for file, _ in evaluations]

    def _get_evaluation_results(self, evaluation_file: str) -> pd.DataFrame:
        """Get the evaluation results from the evaluation file"""
        results_dir = self._get_results_dir()
        file_path = os.path.join(results_dir, evaluation_file)

        with open(file_path, "rb") as f:
            return pd.read_pickle(f)

    def _render_metrics_row(self, df: pd.DataFrame):
        """Render the top metrics row"""
        col1, col2, col3, col4 = st.columns(4)
        
        total_tests = df['test_name'].nunique()
        pass_rate = df.groupby('test_name')['overall_success'].first().mean() if total_tests > 0 else 0
        avg_score = df['metric_score'].mean() if len(df) > 0 else 0
        
        # Calculate tool accuracy
        tool_accuracy = 0
        if 'expected_tools' in df.columns and 'actual_tools' in df.columns:
            test_tool_matches = []
            for test_name in df['test_name'].unique():
                test_row = df[df['test_name'] == test_name].iloc[0]
                expected = set(test_row.get('expected_tools', []))
                actual = set(test_row.get('actual_tools', []))
                test_tool_matches.append(expected == actual)
            tool_accuracy = sum(test_tool_matches) / len(test_tool_matches) if test_tool_matches else 0
        
        with col1:
            st.metric("Total Tests", f"{total_tests:,}")
        with col2:
            st.metric(
                "Pass Rate", 
                f"{pass_rate:.1%}",
                delta=f"{(pass_rate - 0.75):.1%}" if total_tests > 0 else None,
                delta_color="normal" if pass_rate > 0.75 else "inverse"
            )
        with col3:
            st.metric("Avg Score", f"{avg_score:.2f}")
        with col4:
            st.metric("Tool Accuracy", f"{tool_accuracy:.1%}")

    def _render_analysis_tabs(self, df):
        """Render analysis visualization tabs"""
        st.divider()
        st.subheader("📊 Analysis")
        tab1, tab2, tab3 = st.tabs(["Tool Usage", "Score Distribution", "Metric Breakdown"])
        
        with tab1:
            self._render_tool_usage_analysis(df)
        with tab2:
            self._render_score_distribution(df)
        with tab3:
            self._render_metric_breakdown(df)

    def _render_tool_usage_analysis(self, df):
        """Render tool usage analysis chart"""
        tool_analysis = []
        for test_name in df['test_name'].unique():
            test_row = df[df['test_name'] == test_name].iloc[0]
            expected = set(test_row.get('expected_tools', []))
            actual = set(test_row.get('actual_tools', []))
            missing = expected - actual
            extra = actual - expected
            
            for tool in missing:
                tool_analysis.append({'tool': tool, 'type': 'Missing', 'count': 1})
            for tool in extra:
                tool_analysis.append({'tool': tool, 'type': 'Extra', 'count': 1})
        
        if tool_analysis:
            tool_df = pd.DataFrame(tool_analysis)
            tool_summary = tool_df.groupby(['tool', 'type']).sum().reset_index()
            
            fig = px.bar(
                tool_summary, x='tool', y='count', color='type',
                title="Tool Usage Issues",
                color_discrete_map={'Missing': '#dc3545', 'Extra': '#ffc107'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tool usage issues found")

    def _render_score_distribution(self, df):
        """Render score distribution histogram"""
        fig = px.histogram(
            df, x='metric_score', nbins=20,
            title="Score Distribution",
            color='overall_success',
            color_discrete_map={True: '#28a745', False: '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_metric_breakdown(self, df):
        """Render metric breakdown analysis"""
        metric_summary = df.groupby('metric_name').agg({
            'metric_score': ['mean', 'min', 'max'],
            'metric_success': 'mean'
        }).round(3)
        
        metric_summary.columns = ['Avg Score', 'Min Score', 'Max Score', 'Success Rate']
        metric_summary['Success Rate'] = metric_summary['Success Rate'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(metric_summary, use_container_width=True)

    def _render_test_details(self, test_data_all_metrics):
        """Render detailed view of selected test"""
        # Get the first row for common data
        first_row = test_data_all_metrics.iloc[0]
        
        # Test header with name and status
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown(f"#### {first_row['test_name']}")
        with header_col2:
            status_icon = "✅" if first_row['overall_success'] else "❌"
            st.markdown(f"### {status_icon} {'PASS' if first_row['overall_success'] else 'FAIL'}")
        
        # Metrics breakdown for this test
        st.markdown("##### 📊 Metrics")
        for _, row in test_data_all_metrics.iterrows():
            icon = "✅" if row['metric_success'] else "❌"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"{icon} **{row['metric_name']}**: {row['metric_reason']}")
            with col2:
                st.markdown(f"**Score: {row['metric_score']:.2f}**")
        
        st.divider()
        
        # Input/Output section
        st.markdown("##### 📥 Input")
        st.text_area("", first_row['input'], height=100, key="detail_input", disabled=True)
        
        st.markdown("##### 📤 Output")
        st.text_area("", first_row['actual_output'], height=100, key="detail_output", disabled=True)
        
        # Tool Analysis section
        st.markdown("##### 🔧 Tool Analysis")
        expected_tools = set(first_row.get('expected_tools', []))
        actual_tools = set(first_row.get('actual_tools', []))
        match_rate = len(expected_tools & actual_tools) / max(len(expected_tools), 1) if expected_tools else 1.0
        
        # Compact tool match indicator
        tool_status_col1, tool_status_col2 = st.columns([2, 1])
        with tool_status_col1:
            if match_rate == 1.0:
                st.markdown("✅ **Perfect tool match**")
            elif match_rate >= 0.5:
                st.markdown("⚠️ **Partial tool match**")
            else:
                st.markdown("❌ **Poor tool match**")
        with tool_status_col2:
            st.markdown(f"**Match: {match_rate:.0%}**")
        
        # Compact tool comparison
        tool_col1, tool_col2, tool_col3 = st.columns(3)
        with tool_col1:
            expected_html = "<div class='tool-section'><b>Expected</b><br>"
            if expected_tools:
                for tool in expected_tools:
                    if tool in actual_tools:
                        expected_html += f"<span class='pass'>✓ {tool}</span>"
                    else:
                        expected_html += f"<span class='fail'>✗ {tool}</span>"
            else:
                expected_html += "<span>None</span>"
            expected_html += "</div>"
            st.markdown(expected_html, unsafe_allow_html=True)
        
        with tool_col2:
            actual_html = "<div class='tool-section'><b>Actual</b><br>"
            if actual_tools:
                for tool in actual_tools:
                    if tool in expected_tools:
                        actual_html += f"<span class='pass'>✓ {tool}</span>"
                    else:
                        actual_html += f"<span class='warning'>+ {tool}</span>"
            else:
                actual_html += "<span>None</span>"
            actual_html += "</div>"
            st.markdown(actual_html, unsafe_allow_html=True)
        
        with tool_col3:
            issues_html = "<div class='tool-section'><b>Issues</b><br>"
            missing = expected_tools - actual_tools
            extra = actual_tools - expected_tools
            if missing:
                for tool in missing:
                    issues_html += f"<span class='fail'>- {tool}</span>"
            if extra:
                for tool in extra:
                    issues_html += f"<span class='warning'>+ {tool}</span>"
            if not missing and not extra:
                issues_html += "<span class='pass'>None</span>"
            issues_html += "</div>"
            st.markdown(issues_html, unsafe_allow_html=True)
        
        # Context and verbose logs
        if first_row.get('context'):
            st.markdown("##### 📋 Context")
            st.info(first_row['context'])
        
        # Check if any row has verbose logs
        verbose_logs = [row.get('verbose_logs', '') for _, row in test_data_all_metrics.iterrows() if row.get('verbose_logs')]
        if verbose_logs:
            with st.expander("🔍 Verbose Logs"):
                for _, row in test_data_all_metrics.iterrows():
                    if row.get('verbose_logs'):
                        st.markdown(f"**{row['metric_name']}:**")
                        st.code(row['verbose_logs'], language='text')

    def run(self):
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🔍 Evaluation Results Viewer")
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        # Initialize the sidebar to get file selection
        with st.sidebar:
            st.header("🎯 Filters")
            
            # File selection
            self.evaluation_file = st.selectbox(
                "📁 Evaluation File", 
                self._get_evaluations(),
                format_func=lambda x: f"{x} ({self._get_file_time(x)})",
                key="evaluation_file_selector"
            )
        
        # Load data if file is selected
        df = None
        if self.evaluation_file:
            df = self._get_evaluation_results(self.evaluation_file)
            
            # Apply filters if data is loaded
            if df is not None and len(df) > 0:
                # Generate the rest of the sidebar filters
                with st.sidebar:
                    st.divider()
                    
                    # Metric filter
                    st.subheader("📊 Metrics")
                    available_metrics = df['metric_name'].unique()
                    selected_metrics = st.multiselect(
                        "Select Metrics",
                        available_metrics,
                        default=available_metrics,
                        key="metrics_selector"
                    )
                    df = df[df['metric_name'].isin(selected_metrics)]
                    
                    st.divider()
                    
                    # Status filters
                    st.subheader("✅ Status")
                    status_filter = st.radio("Show", ["All", "Passed Only", "Failed Only"], key="status_filter")
                    if status_filter == "Passed Only":
                        df = df[df['overall_success'] == True]
                    elif status_filter == "Failed Only":
                        df = df[df['overall_success'] == False]
                    
                    score_range = st.slider("Score Range", 0.0, 1.0, (0.0, 1.0), 0.1, key="score_range")
                    df = df[(df['metric_score'] >= score_range[0]) & (df['metric_score'] <= score_range[1])]
                    
                    st.divider()
                    
                    # Tool filters
                    st.subheader("🔧 Tool Analysis")
                    tool_filter_mode = st.selectbox(
                        "Filter Mode",
                        ["All", "Tool Mismatch", "Missing Expected", "Unexpected Usage"],
                        key="tool_filter_mode"
                    )
                    
                    # Apply tool filtering
                    if tool_filter_mode == "Tool Mismatch":
                        df = df[df.apply(lambda r: set(r.get('expected_tools', [])) != set(r.get('actual_tools', [])), axis=1)]
                    elif tool_filter_mode == "Missing Expected":
                        df = df[df.apply(lambda r: len(set(r.get('expected_tools', [])) - set(r.get('actual_tools', []))) > 0, axis=1)]
                    elif tool_filter_mode == "Unexpected Usage":
                        df = df[df.apply(lambda r: len(set(r.get('actual_tools', [])) - set(r.get('expected_tools', []))) > 0, axis=1)]
                    
                    # Test name filter
                    st.divider()
                    st.subheader("🔍 Test Cases")
                    test_names = df['test_name'].unique()
                    if len(test_names) > 1:
                        selected_tests = st.multiselect(
                            "Select Tests",
                            test_names,
                            default=test_names,
                            key="test_selector"
                        )
                        df = df[df['test_name'].isin(selected_tests)]

        if self.evaluation_file and df is not None and len(df) > 0:
            # Display metrics
            self._render_metrics_row(df)
            
            # Create side-by-side layout
            col_table, col_details = st.columns([2, 3])
            
            with col_table:
                st.markdown("### 📋 Test Results")
                st.markdown("*Click on any row to view details →*")
                
                # Group by test_name to show summary
                test_summary = df.groupby('test_name').agg({
                    'overall_success': 'first',
                    'metric_score': 'mean',
                    'metric_success': lambda x: (x == True).all()  # All metrics must pass
                }).reset_index()
                
                test_summary['status'] = test_summary['overall_success'].map({True: '✅ Pass', False: '❌ Fail'})
                test_summary['score_display'] = test_summary['metric_score'].apply(lambda x: f"{x:.2f}")
                
                # Show simplified table with selection
                event = st.dataframe(
                    test_summary[['test_name', 'score_display', 'status']],
                    use_container_width=True,
                    height=700,
                    column_config={
                        'test_name': st.column_config.TextColumn('Test Case'),
                        'score_display': st.column_config.TextColumn('Avg Score'),
                        'status': st.column_config.TextColumn('Status')
                    },
                    on_select="rerun",
                    selection_mode="single-row"
                )
            
            with col_details:
                st.markdown("### 📄 Test Details")
                
                # Show details if a row is selected
                if event.selection and event.selection['rows']:
                    selected_idx = event.selection['rows'][0]
                    selected_test = test_summary.iloc[selected_idx]['test_name']
                    test_data_all_metrics = df[df['test_name'] == selected_test]
                    self._render_test_details(test_data_all_metrics)
                else:
                    st.info("👈 Click on any test case in the table to view its details here")
            
            # Display analysis
            self._render_analysis_tabs(df)
            
        elif self.evaluation_file:
            st.warning("No data to display after applying filters")
        else:
            st.info("Select an evaluation file from the sidebar to begin")


def main():
    """Standalone entry point"""
    viewer = EvaluationViewer()
    viewer.run()


if __name__ == "__main__":
    main()