import streamlit as st
import os
import pandas as pd
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import json

from app.config.settings import settings


class EnhancedRAGViewer:
    """Enhanced viewer for RAG evaluation results with improved UX"""

    def __init__(self):
        self.evaluation_dir = settings.EVALUATION_OUTPUT_DIR
        self._setup_page_config()
        self._inject_custom_css()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = 'summary'
        if 'selected_test' not in st.session_state:
            st.session_state.selected_test = None
        if 'show_cot' not in st.session_state:
            st.session_state.show_cot = False

    def _setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="RAG Evaluation Analysis",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def _inject_custom_css(self):
        """Add custom CSS for enhanced styling"""
        st.markdown("""
        <style>
            /* Modern card styling */
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            
            .metric-card h3 {
                margin: 0;
                font-size: 1.8rem;
                font-weight: 700;
            }
            
            .metric-card p {
                margin: 0.5rem 0 0 0;
                opacity: 0.9;
                font-size: 0.9rem;
            }
            
            /* Score badges */
            .score-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.85rem;
            }
            
            .score-high { background: #10b981; color: white; }
            .score-medium { background: #f59e0b; color: white; }
            .score-low { background: #ef4444; color: white; }
            
            /* Test case cards */
            .test-card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 1.25rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .test-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                transform: translateY(-2px);
            }
            
            .test-card.selected {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            /* Metric group styling */
            .metric-group {
                background: #f9fafb;
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            
            .metric-group-title {
                font-weight: 600;
                font-size: 0.9rem;
                color: #4b5563;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.75rem;
            }
            
            /* Context display */
            .context-box {
                background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                border-left: 4px solid #f59e0b;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
            
            .context-box h4 {
                margin: 0 0 0.5rem 0;
                color: #92400e;
                font-size: 0.9rem;
                font-weight: 600;
            }
            
            /* COT display */
            .cot-box {
                background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%);
                border-left: 4px solid #8b5cf6;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
            
            .cot-box h4 {
                margin: 0 0 0.5rem 0;
                color: #5b21b6;
                font-size: 0.9rem;
                font-weight: 600;
            }
            
            /* Output comparison */
            .output-comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin: 1rem 0;
            }
            
            .output-box {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
            }
            
            .output-box.actual {
                border-left: 4px solid #10b981;
            }
            
            .output-box.expected {
                border-left: 4px solid #3b82f6;
            }
            
            /* Progress bar styling */
            .progress-bar {
                width: 100%;
                height: 8px;
                background: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
                margin: 0.5rem 0;
            }
            
            .progress-fill {
                height: 100%;
                transition: width 0.3s ease;
            }
            
            .progress-fill.high { background: #10b981; }
            .progress-fill.medium { background: #f59e0b; }
            .progress-fill.low { background: #ef4444; }
            
            /* Navigation buttons */
            .nav-button {
                background: white;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: 600;
                transition: all 0.2s ease;
                cursor: pointer;
            }
            
            .nav-button:hover {
                border-color: #667eea;
                background: #f3f4f6;
            }
            
            .nav-button.active {
                background: #667eea;
                color: white;
                border-color: #667eea;
            }
            
            /* Verbose reasoning */
            .verbose-reason {
                background: #f3f4f6;
                border-radius: 6px;
                padding: 0.75rem;
                font-size: 0.85rem;
                color: #4b5563;
                margin-top: 0.5rem;
            }
            
            /* Tool analysis */
            .tool-match {
                display: inline-block;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-size: 0.8rem;
                margin: 0.2rem;
            }
            
            .tool-match.correct { background: #d1fae5; color: #065f46; }
            .tool-match.missing { background: #fee2e2; color: #991b1b; }
            .tool-match.extra { background: #fed7aa; color: #9a3412; }
        </style>
        """, unsafe_allow_html=True)

    def _get_results_dir(self):
        """Get the results directory path"""
        if os.path.isabs(self.evaluation_dir):
            return os.path.join(self.evaluation_dir, "results")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            return os.path.join(project_root, self.evaluation_dir, "results")

    def _get_evaluations(self) -> List[str]:
        """Get all evaluation files sorted by modification time"""
        results_dir = self._get_results_dir()
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            return []
        
        evaluations = []
        for file in os.listdir(results_dir):
            if file.endswith(".pkl"):
                file_path = os.path.join(results_dir, file)
                mod_time = os.path.getmtime(file_path)
                evaluations.append((file, mod_time))
        
        evaluations.sort(key=lambda x: x[1], reverse=True)
        return [file for file, _ in evaluations]

    def _load_evaluation_data(self, filename: str) -> pd.DataFrame:
        """Load evaluation data from pickle file"""
        file_path = os.path.join(self._get_results_dir(), filename)
        with open(file_path, "rb") as f:
            return pd.read_pickle(f)

    def _get_metric_category(self, metric_name: str) -> str:
        """Categorize metrics for grouping"""
        retrieval_metrics = ['Contextual Relevancy', 'Contextual Recall', 'Contextual Precision']
        faithfulness_metrics = ['Faithfulness', 'Hallucination']
        relevancy_metrics = ['Answer Relevancy']
        tool_metrics = ['Tool Correctness']
        
        if metric_name in retrieval_metrics:
            return "Retrieval Quality"
        elif metric_name in faithfulness_metrics:
            return "Faithfulness"
        elif metric_name in relevancy_metrics:
            return "Answer Quality"
        elif metric_name in tool_metrics:
            return "Tool Usage"
        else:
            return "Other"

    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score value"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"

    def _render_metric_card(self, title: str, value: str, subtitle: str = "", color: str = "blue"):
        """Render a styled metric card"""
        colors = {
            "blue": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "green": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
            "orange": "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
            "red": "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
        }
        
        st.markdown(f"""
        <div class="metric-card" style="background: {colors.get(color, colors['blue'])}">
            <h3>{value}</h3>
            <p style="font-weight: 600; font-size: 1rem;">{title}</p>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)

    def _render_summary_view(self, df: pd.DataFrame):
        """Render the summary dashboard view"""
        st.header("📊 Evaluation Summary Dashboard")
        
        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_tests = df['test_name'].nunique()
        pass_rate = df.groupby('test_name')['overall_success'].first().mean()
        avg_score = df['metric_score'].mean()
        
        # Calculate retrieval quality
        retrieval_metrics = df[df['metric_name'].isin(['Contextual Relevancy', 'Contextual Recall', 'Contextual Precision'])]
        retrieval_score = retrieval_metrics['metric_score'].mean() if len(retrieval_metrics) > 0 else 0
        
        with col1:
            self._render_metric_card("Total Tests", str(total_tests), "Evaluated cases", "blue")
        
        with col2:
            color = "green" if pass_rate >= 0.8 else "orange" if pass_rate >= 0.6 else "red"
            self._render_metric_card("Pass Rate", f"{pass_rate:.1%}", "Overall success", color)
        
        with col3:
            color = "green" if avg_score >= 0.8 else "orange" if avg_score >= 0.6 else "red"
            self._render_metric_card("Avg Score", f"{avg_score:.2f}", "All metrics", color)
        
        with col4:
            color = "green" if retrieval_score >= 0.8 else "orange" if retrieval_score >= 0.6 else "red"
            self._render_metric_card("Retrieval Quality", f"{retrieval_score:.1%}", "Context relevance", color)
        
        # Metric breakdown by category
        st.subheader("📈 Metric Performance by Category")
        
        metric_categories = {}
        for metric in df['metric_name'].unique():
            category = self._get_metric_category(metric)
            if category not in metric_categories:
                metric_categories[category] = []
            metric_categories[category].append(metric)
        
        cols = st.columns(len(metric_categories))
        for idx, (category, metrics) in enumerate(metric_categories.items()):
            with cols[idx]:
                st.markdown(f"**{category}**")
                category_df = df[df['metric_name'].isin(metrics)]
                for metric in metrics:
                    metric_df = category_df[category_df['metric_name'] == metric]
                    if len(metric_df) > 0:
                        score = metric_df['metric_score'].mean()
                        success_rate = metric_df['metric_success'].mean()
                        
                        # Progress bar
                        progress_class = self._get_score_class(score)
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <span style="font-size: 0.85rem;">{metric}</span>
                                <span class="score-badge score-{progress_class}">{score:.2f}</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill {progress_class}" style="width: {score*100}%"></div>
                            </div>
                            <span style="font-size: 0.75rem; color: #6b7280;">Success: {success_rate:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Test case overview
        st.subheader("🧪 Test Cases Overview")
        
        test_summary = df.groupby('test_name').agg({
            'overall_success': 'first',
            'metric_score': 'mean',
            'input': 'first'
        }).reset_index()
        
        # Create clickable test cards
        cols = st.columns(2)
        for idx, row in test_summary.iterrows():
            with cols[idx % 2]:
                status_icon = "✅" if row['overall_success'] else "❌"
                score_class = self._get_score_class(row['metric_score'])
                
                if st.button(
                    f"{status_icon} {row['test_name'][:30]}... (Score: {row['metric_score']:.2f})",
                    key=f"test_{idx}",
                    use_container_width=True
                ):
                    st.session_state.selected_test = row['test_name']
                    st.session_state.view_mode = 'detail'
                    st.rerun()
                
                # Show preview of input
                st.caption(f"📝 {row['input'][:100]}...")

    def _render_detail_view(self, df: pd.DataFrame, test_name: str):
        """Render detailed view for a specific test case"""
        test_data = df[df['test_name'] == test_name]
        first_row = test_data.iloc[0]
        
        # Header with navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("← Back to Summary", use_container_width=True):
                st.session_state.view_mode = 'summary'
                st.session_state.selected_test = None
                st.rerun()
        
        with col2:
            status = "✅ PASSED" if first_row['overall_success'] else "❌ FAILED"
            st.markdown(f"<h2 style='text-align: center;'>{test_name} - {status}</h2>", unsafe_allow_html=True)
        
        with col3:
            # Navigation between tests
            test_names = df['test_name'].unique().tolist()
            current_idx = test_names.index(test_name)
            
            col_prev, col_next = st.columns(2)
            with col_prev:
                if current_idx > 0 and st.button("← Prev", use_container_width=True):
                    st.session_state.selected_test = test_names[current_idx - 1]
                    st.rerun()
            
            with col_next:
                if current_idx < len(test_names) - 1 and st.button("Next →", use_container_width=True):
                    st.session_state.selected_test = test_names[current_idx + 1]
                    st.rerun()
        
        # Main content layout
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Input/Output", "📊 Metrics", "🔍 Context & Retrieval", "🧠 Analysis"])
        
        with tab1:
            self._render_io_tab(first_row)
        
        with tab2:
            self._render_metrics_tab(test_data)
        
        with tab3:
            self._render_context_tab(first_row)
        
        with tab4:
            self._render_analysis_tab(first_row, test_data)

    def _render_io_tab(self, row):
        """Render Input/Output comparison tab"""
        st.subheader("Input")
        st.info(row['input'])
        
        st.subheader("Output Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Actual Output**")
            st.markdown("""
            <div class="output-box actual">
            """, unsafe_allow_html=True)
            st.write(row['actual_output'])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if 'expected_output' in row and row.get('expected_output'):
                st.markdown("**📋 Expected Output**")
                st.markdown("""
                <div class="output-box expected">
                """, unsafe_allow_html=True)
                st.write(row['expected_output'])
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("**📋 Expected Context**")
                if row.get('context'):
                    st.markdown("""
                    <div class="output-box expected">
                    """, unsafe_allow_html=True)
                    if isinstance(row['context'], list):
                        for ctx in row['context']:
                            st.write(f"• {ctx}")
                    else:
                        st.write(row['context'])
                    st.markdown("</div>", unsafe_allow_html=True)

    def _render_metrics_tab(self, test_data):
        """Render metrics breakdown tab"""
        st.subheader("Metric Scores & Reasoning")
        
        # Group metrics by category
        categories = {}
        for _, row in test_data.iterrows():
            category = self._get_metric_category(row['metric_name'])
            if category not in categories:
                categories[category] = []
            categories[category].append(row)
        
        for category, metrics in categories.items():
            st.markdown(f"""
            <div class="metric-group">
                <div class="metric-group-title">{category}</div>
            """, unsafe_allow_html=True)
            
            for metric_row in metrics:
                icon = "✅" if metric_row['metric_success'] else "❌"
                score_class = self._get_score_class(metric_row['metric_score'])
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.markdown(f"{icon} **{metric_row['metric_name']}**")
                
                with col2:
                    # Progress bar for score
                    st.markdown(f"""
                    <div class="progress-bar">
                        <div class="progress-fill {score_class}" style="width: {metric_row['metric_score']*100}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <span class="score-badge score-{score_class}">{metric_row['metric_score']:.2f}</span>
                    """, unsafe_allow_html=True)
                
                # Show reasoning
                if metric_row.get('metric_reason'):
                    with st.expander(f"View reasoning for {metric_row['metric_name']}"):
                        st.write(metric_row['metric_reason'])
                
                # Show verbose logs if available
                if metric_row.get('verbose_logs'):
                    with st.expander(f"Verbose logs for {metric_row['metric_name']}"):
                        st.code(metric_row['verbose_logs'], language='text')
            
            st.markdown("</div>", unsafe_allow_html=True)

    def _render_context_tab(self, row):
        """Render context and retrieval information tab"""
        st.subheader("🔍 Context & Retrieval Information")
        
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Actual Retrieved Context")
            
            # Check for retrieval_context in the data
            has_retrieval = False
            
            # Try different possible fields for retrieval context
            retrieval_fields = ['retrieval_context', 'retrieved_context', 'actual_context']
            for field in retrieval_fields:
                if row.get(field):
                    has_retrieval = True
                    st.markdown("""
                    <div class="context-box">
                        <h4>What the agent actually retrieved:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if isinstance(row[field], list):
                        for idx, ctx in enumerate(row[field], 1):
                            with st.expander(f"📄 Retrieved Chunk {idx}", expanded=(idx == 1)):
                                st.write(ctx)
                    else:
                        st.info(row[field])
                    break
            
            # If no explicit retrieval context, try to extract from actual tool calls
            if not has_retrieval and row.get('actual_tool_calls'):
                actual_calls = row.get('actual_tool_calls', [])
                
                # Find knowledge base searches
                kb_searches = [tc for tc in actual_calls if tc.get('name') == 'search_knowledge_base']
                
                if kb_searches:
                    has_retrieval = True
                    st.markdown("""
                    <div class="context-box">
                        <h4>What the agent retrieved from knowledge base:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display each knowledge base search
                    for idx, search in enumerate(kb_searches, 1):
                        if len(kb_searches) > 1:
                            st.markdown(f"#### Search {idx}")
                        
                        # Show query parameters
                        params = search.get('input_parameters', {})
                        if params:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Query:** `{params.get('query', 'N/A')}`")
                            with col2:
                                st.write(f"**Limit:** {params.get('limit', 'N/A')}")
                        
                        # Show actual results
                        output = search.get('output', '')
                        if output:
                            # Try to parse structured output
                            import re
                            if 'Found' in output and 'results:' in output:
                                # Split by numbered results
                                results = re.split(r'\n\d+\.\s+\[', output)
                                
                                if len(results) > 1:
                                    # Show the header
                                    header = results[0].strip()
                                    if header:
                                        st.info(header)
                                    
                                    # Show each result in an expander
                                    for i, result in enumerate(results[1:], 1):
                                        # Extract file name and score
                                        file_match = re.match(r'([^\]]+)\]\s*\(Score:\s*([\d.]+)\)', result)
                                        if file_match:
                                            filename = file_match.group(1)
                                            score = file_match.group(2)
                                            # Get the content after the score line
                                            content_start = result.find('\n')
                                            content = result[content_start:].strip() if content_start != -1 else result
                                            
                                            with st.expander(f"📄 Result {i}: [{filename}] (Score: {score})", expanded=(i <= 2)):
                                                st.text(content[:1000])  # Limit content length
                                        else:
                                            with st.expander(f"📄 Result {i}", expanded=(i <= 2)):
                                                st.text(result[:1000])
                                else:
                                    # Just show the raw content if we can't parse it
                                    with st.expander("📄 Retrieved Content", expanded=True):
                                        st.text(output[:3000])
                            else:
                                # Show raw retrieval content
                                with st.expander("📄 Retrieved Content", expanded=True):
                                    st.text(output[:3000])
            
            if not has_retrieval:
                st.info("ℹ️ No retrieval context data available. The agent may have answered from its training or the retrieval wasn't logged.")
        
        with col2:
            st.markdown("### 📋 Expected Context")
            
            if row.get('context'):
                st.markdown("""
                <div class="context-box" style="background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); border-left-color: #0284c7;">
                    <h4 style="color: #075985;">What should have been retrieved:</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Handle both string and list formats
                if isinstance(row['context'], str):
                    # Try to parse as JSON list first
                    try:
                        import json
                        context_list = json.loads(row['context'].replace("'", '"'))
                        if isinstance(context_list, list):
                            for idx, ctx in enumerate(context_list, 1):
                                with st.expander(f"📌 Expected Item {idx}", expanded=(idx == 1)):
                                    st.write(ctx)
                        else:
                            st.info(row['context'])
                    except:
                        # If not JSON, display as is
                        st.info(row['context'])
                elif isinstance(row['context'], list):
                    for idx, ctx in enumerate(row['context'], 1):
                        with st.expander(f"📌 Expected Item {idx}", expanded=(idx == 1)):
                            st.write(ctx)
                else:
                    st.info(str(row['context']))
            else:
                st.info("ℹ️ No expected context defined for this test case.")
        
        # Comparison Analysis
        st.markdown("---")
        st.markdown("### 🔬 Retrieval Quality Analysis")
        
        # Tool usage analysis
        if row.get('expected_tool_calls') or row.get('actual_tool_calls'):
            st.subheader("🔧 Tool Usage Analysis")
            
            # Extract tool names from the new format
            expected = set(t.get('name') for t in row.get('expected_tool_calls', []))
            actual = set(t.get('name') for t in row.get('actual_tool_calls', []))
            
            correct = expected & actual
            missing = expected - actual
            extra = actual - expected
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**✅ Correctly Used**")
                if correct:
                    for tool in correct:
                        st.markdown(f"""
                        <span class="tool-match correct">{tool}</span>
                        """, unsafe_allow_html=True)
                else:
                    st.write("None")
            
            with col2:
                st.markdown("**❌ Missing Tools**")
                if missing:
                    for tool in missing:
                        st.markdown(f"""
                        <span class="tool-match missing">{tool}</span>
                        """, unsafe_allow_html=True)
                else:
                    st.write("None")
            
            with col3:
                st.markdown("**⚠️ Extra Tools**")
                if extra:
                    for tool in extra:
                        st.markdown(f"""
                        <span class="tool-match extra">{tool}</span>
                        """, unsafe_allow_html=True)
                else:
                    st.write("None")
        
        # Show tool call details if available
        actual_calls = row.get('actual_tool_calls', [])
        if actual_calls:
            st.markdown("---")
            st.markdown("### 📊 Tool Call Details")
            
            for idx, tool_call in enumerate(actual_calls, 1):
                tool_name = tool_call.get('name', 'Unknown')
                
                with st.expander(f"🔧 {tool_name} (Call #{idx})", expanded=(idx == 1)):
                    # Show input parameters
                    params = tool_call.get('input_parameters')
                    if params:
                        st.markdown("**Input Parameters:**")
                        st.json(params)
                    else:
                        st.info("No input parameters")
                    
                    # Show output (truncated)
                    output = tool_call.get('output')
                    if output:
                        st.markdown("**Output:**")
                        if isinstance(output, str) and len(output) > 500:
                            st.text(output[:500] + "...")
                            if st.button(f"Show full output", key=f"full_output_{idx}"):
                                st.text(output)
                        else:
                            st.text(str(output))
                    else:
                        st.info("No output captured")

    def _render_analysis_tab(self, row, test_data):
        """Render analysis and chain of thought tab"""
        # Chain of Thought
        if row.get('chain_of_thought'):
            st.subheader("🧠 Chain of Thought Analysis")
            
            st.markdown("""
            <div class="cot-box">
                <h4>Agent's Reasoning Process</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Option to show full or extracted COT
            col1, col2 = st.columns([3, 1])
            with col2:
                show_full = st.checkbox("Show full COT", value=False)
            
            if show_full:
                st.text_area("Full Chain of Thought", row['chain_of_thought'], height=300, disabled=True)
            else:
                # Extract key reasoning points (simplified extraction)
                cot_text = row['chain_of_thought']
                
                # Try to extract key reasoning steps
                lines = cot_text.split('\n')
                key_points = []
                
                for line in lines:
                    line = line.strip()
                    if line and (
                        line.startswith(('I ', 'Let me', 'Looking', 'Based on', 'The ', 'This ')) or
                        '?' in line or
                        any(keyword in line.lower() for keyword in ['because', 'therefore', 'however', 'but', 'so'])
                    ):
                        key_points.append(line)
                
                if key_points:
                    st.markdown("**Key Reasoning Points:**")
                    for point in key_points[:10]:  # Limit to 10 key points
                        st.write(f"• {point}")
                else:
                    st.write(cot_text[:500] + "..." if len(cot_text) > 500 else cot_text)
        
        # Metric Performance Summary
        st.subheader("📊 Performance Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Metric scores chart
            fig = go.Figure(data=[
                go.Bar(
                    x=test_data['metric_name'],
                    y=test_data['metric_score'],
                    marker_color=['green' if s else 'red' for s in test_data['metric_success']]
                )
            ])
            fig.update_layout(
                title="Metric Scores",
                yaxis_title="Score",
                xaxis_title="",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pass/Fail pie chart
            success_count = test_data['metric_success'].sum()
            fail_count = len(test_data) - success_count
            
            fig = go.Figure(data=[go.Pie(
                labels=['Passed', 'Failed'],
                values=[success_count, fail_count],
                marker_colors=['#10b981', '#ef4444']
            )])
            fig.update_layout(
                title="Metric Pass/Fail",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main application runner"""
        # Header
        st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>
            🔍 RAG Evaluation Analysis Platform
        </h1>
        """, unsafe_allow_html=True)
        
        # File selection
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            evaluation_files = self._get_evaluations()
            
            if not evaluation_files:
                st.warning("No evaluation files found. Please run an evaluation first.")
                return
            
            selected_file = st.selectbox(
                "Select Evaluation File",
                evaluation_files,
                format_func=lambda x: f"📁 {x}",
                label_visibility="collapsed"
            )
        
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Load data
        if selected_file:
            df = self._load_evaluation_data(selected_file)
            
            if df is None or len(df) == 0:
                st.error("No data found in the selected file.")
                return
            
            # View mode toggle
            if st.session_state.view_mode == 'summary':
                self._render_summary_view(df)
            elif st.session_state.view_mode == 'detail' and st.session_state.selected_test:
                self._render_detail_view(df, st.session_state.selected_test)
            else:
                # Fallback to summary if no test selected
                st.session_state.view_mode = 'summary'
                self._render_summary_view(df)


def main():
    """Main entry point"""
    viewer = EnhancedRAGViewer()
    viewer.run()


if __name__ == "__main__":
    main()