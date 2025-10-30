"""
AI Data Analysis Agent - Enhanced Main Application v2.0
Support for large datasets, auto research questions, comprehensive analysis
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from pathlib import Path
import gc

# Import enhanced custom modules
try:
    from analysis_engine import EnhancedAIAnalysisEngine
    from visualization_module import EnhancedVisualizationEngine
    from report_generator import EnhancedReportGenerator
    from utils_module import DataValidator, ModelPersistence
except ImportError:
    st.error("⚠️ Enhanced modules not found. Please ensure all files are in the same directory.")

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent v2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .research-question {
        background-color: #f8f9fa;
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .research-question:hover {
        background-color: #e9ecef;
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'research_questions' not in st.session_state:
    st.session_state.research_questions = []

def load_large_dataset(uploaded_file, chunk_size=100000):
    """Load large datasets with chunking support - up to 2GB"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
        
        st.info(f"📦 Loading file: {uploaded_file.name} ({file_size:.1f} MB)")
        
        if file_extension == 'csv':
            # For very large files, use chunking
            if file_size > 200:  # If larger than 200MB, use chunking
                st.warning(f"⚡ Large file detected ({file_size:.1f} MB). Using optimized chunking...")
                
                progress_bar = st.progress(0)
                chunks = []
                total_rows = 0
                
                # Read in chunks
                chunk_iterator = pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=False)
                
                for i, chunk in enumerate(chunk_iterator):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    # Update progress
                    progress_bar.progress(min((i + 1) * 10 / 100, 0.9))
                    
                    # Limit to 2 million rows for memory safety
                    if total_rows >= 2000000:
                        st.warning(f"⚠️ Dataset limited to first {total_rows:,} rows for optimal performance")
                        break
                    
                    # Memory check - stop if we've collected enough
                    if len(chunks) >= 20:
                        break
                
                progress_bar.progress(1.0)
                df = pd.concat(chunks, ignore_index=True)
                st.success(f"✅ Loaded {len(df):,} rows successfully!")
                
            else:
                # Normal loading for smaller files
                df = pd.read_csv(uploaded_file, low_memory=False)
        
        elif file_extension in ['xlsx', 'xls']:
            if file_size > 100:
                st.warning("⚠️ Large Excel file detected. This may take a moment...")
            df = pd.read_excel(uploaded_file)
        
        else:
            st.error(f"❌ Unsupported file format: {file_extension}")
            return None
        
        # Final memory check
        mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
        st.info(f"📊 Dataset loaded: {len(df):,} rows × {len(df.columns)} columns | Memory: {mem_usage:.1f} MB")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.exception(e)
        return None

def main():
    """Main application logic"""
    
    # Header
    st.markdown('<p class="main-header">🤖 AI Data Analysis Agent v2.0</p>', unsafe_allow_html=True)
    st.markdown("**Upload → Auto-Analyze → Download** | Advanced ML with Smart Research Questions")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🎯 Analysis Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        st.divider()
        
        st.subheader("📊 Data Processing")
        chunk_size = st.number_input("Chunk Size (for large files)", 
                                    min_value=10000, max_value=500000, 
                                    value=100000, step=10000)
        
        st.divider()
        
        st.subheader("ℹ️ About v2.0")
        st.info("""
        **New Features:**
        - 🎯 Smart research questions
        - 📊 15+ visualization types
        - 📈 12+ ML models
        - 🔍 Comprehensive statistics
        - 📄 Detailed reports
        - 💾 Large file support (2GB)
        """)
        
        st.subheader("📊 Supported Formats")
        st.markdown("""
        - 📄 CSV (up to 2GB)
        - 📊 Excel (.xlsx, .xls)
        - 🔢 Up to 2M rows
        - ⚡ Auto-chunking for speed
        """)
        
        st.divider()
        st.caption("v2.0 - Enhanced Edition")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload & Research Questions",
        "📊 Descriptive Statistics",
        "📈 EDA Visualizations",
        "🎯 Model Results",
        "⬇️ Download Reports"
    ])
    
    with tab1:
        st.header("Step 1: Upload Your Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Supports files up to 2GB. Files over 200MB will be automatically chunked for optimal performance."
            )
        
        with col2:
            st.metric("Max File Size", "2 GB")
            st.metric("Max Rows", "2 Million")
        
        if uploaded_file is not None:
            try:
                # Load data with progress
                with st.spinner("Loading dataset..."):
                    df = load_large_dataset(uploaded_file, chunk_size)
                
                if df is None:
                    return
                
                st.session_state.df = df
                
                # Validate data
                validator = DataValidator(max_size_mb=2000, min_rows=10, min_cols=2)  # 2GB limit
                is_valid, message = validator.validate_dataframe(df)
                
                if not is_valid:
                    st.error(f"❌ Data validation failed: {message}")
                    return
                
                # Display dataset info
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", f"{len(df.columns):,}")
                with col3:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Missing %", f"{missing_pct:.1f}%")
                with col4:
                    health = validator.get_dataset_health_score(df)
                    st.metric("Health Score", f"{health['health_score']:.0f}/100")
                with col5:
                    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
                    st.metric("Memory", f"{mem_mb:.1f} MB")
                
                # Dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Column info
                with st.expander("📋 Column Information & Data Types"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.count().values,
                        'Null': df.isnull().sum().values,
                        'Null %': ((df.isnull().sum() / len(df)) * 100).round(2).values,
                        'Unique': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                st.divider()
                
                # Generate research questions
                st.header("Step 2: Smart Research Question Generation")
                
                if st.button("🎯 Generate Research Questions", type="primary", use_container_width=True):
                    with st.spinner("Analyzing dataset and generating research questions..."):
                        engine = EnhancedAIAnalysisEngine(df)
                        questions = engine.generate_research_questions()
                        st.session_state.research_questions = questions
                        
                        gc.collect()  # Free memory
                    
                    st.success(f"✅ Generated {len(questions)} research questions!")
                
                # Display generated questions
                if st.session_state.research_questions:
                    st.subheader("📝 AI-Generated Research Questions")
                    st.info("💡 These questions are intelligently generated based on your data patterns. Select one to analyze:")
                    
                    # Use radio buttons for better selection
                    question_options = [f"Q{i+1}: {q['question']}" for i, q in enumerate(st.session_state.research_questions)]
                    
                    selected_option = st.radio(
                        "Choose a research question:",
                        question_options,
                        key="question_selector"
                    )
                    
                    # Get selected index
                    selected_idx = question_options.index(selected_option) if selected_option else 0
                    
                    # Show details of selected question
                    if selected_option:
                        selected_q = st.session_state.research_questions[selected_idx]
                        
                        with st.expander("📋 Question Details", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Type:** {selected_q['type'].replace('_', ' ').title()}")
                                st.markdown(f"**Target Variable:** `{selected_q['target'] if selected_q['target'] else 'N/A'}`")
                            with col2:
                                st.markdown(f"**Rationale:** {selected_q['rationale']}")
                    
                    st.divider()
                    
                    # Analyze button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button("🚀 Start Analysis with Selected Question", 
                                   type="primary", use_container_width=True, key="analyze_selected"):
                            selected_q = st.session_state.research_questions[selected_idx]
                            run_analysis(df, selected_q['question'], 
                                       test_size, cv_folds, selected_idx)
                    
                    st.markdown("---")
                    
                    # Custom question option
                    st.subheader("💭 Or Provide Your Own Research Question")
                    custom_question = st.text_area(
                        "Custom Research Question",
                        placeholder="Example: 'Predict customer churn based on usage patterns and demographics'",
                        help="Provide your own specific research question if the generated ones don't match your needs",
                        height=100
                    )
                    
                    if custom_question and len(custom_question) > 10:
                        if st.button("🚀 Analyze Custom Question", 
                                   type="secondary", use_container_width=True, key="analyze_custom"):
                            # Try to auto-detect target from custom question
                            run_analysis(df, custom_question, test_size, cv_folds, 0)
                else:
                    st.info("👆 Click 'Generate Research Questions' to automatically identify analysis opportunities")
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.exception(e)
        else:
            # Landing state
            st.info("👆 Upload a dataset to begin automated analysis")
            
            st.subheader("✨ What's New in v2.0")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🎯 Smart Analysis**
                - Auto research questions
                - Intelligent target detection
                - Task type identification
                """)
            
            with col2:
                st.markdown("""
                **📊 Enhanced Stats**
                - Comprehensive descriptives
                - Data quality assessment
                - Outlier detection
                """)
            
            with col3:
                st.markdown("""
                **🤖 More Models**
                - 10+ classification models
                - 12+ regression models
                - Advanced ensemble methods
                """)
    
    with tab2:
        st.header("📊 Descriptive Statistics & Data Quality")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            if 'descriptive_stats' in results and results['descriptive_stats']:
                stats = results['descriptive_stats']
                
                # Overview metrics
                st.subheader("Dataset Overview")
                
                if 'overview' in stats:
                    overview = stats['overview']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                        st.metric("Total Rows", f"{overview.get('total_rows', 0):,}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                        st.metric("Total Columns", f"{overview.get('total_columns', 0):,}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                        missing_pct = overview.get('missing_percentage', 0)
                        st.metric("Missing Data", f"{missing_pct:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                        dup_pct = overview.get('duplicate_percentage', 0)
                        st.metric("Duplicates", f"{dup_pct:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.divider()
                
                # Numeric features statistics
                if 'numeric_summary' in stats and stats['numeric_summary']:
                    st.subheader("📈 Numeric Features Summary")
                    
                    numeric_stats = stats['numeric_summary']
                    
                    # Create summary dataframe
                    summary_data = []
                    for feature, feature_stats in numeric_stats.items():
                        summary_data.append({
                            'Feature': feature,
                            'Mean': f"{feature_stats.get('mean', 0):.2f}",
                            'Median': f"{feature_stats.get('median', 0):.2f}",
                            'Std Dev': f"{feature_stats.get('std', 0):.2f}",
                            'Min': f"{feature_stats.get('min', 0):.2f}",
                            'Max': f"{feature_stats.get('max', 0):.2f}",
                            'Skewness': f"{feature_stats.get('skewness', 0):.2f}",
                            'Missing %': f"{feature_stats.get('missing_pct', 0):.2f}%"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                st.divider()
                
                # Categorical features statistics
                if 'categorical_summary' in stats and stats['categorical_summary']:
                    st.subheader("🏷️ Categorical Features Summary")
                    
                    cat_stats = stats['categorical_summary']
                    
                    cat_data = []
                    for feature, feature_stats in cat_stats.items():
                        cat_data.append({
                            'Feature': feature,
                            'Unique Values': feature_stats.get('unique_values', 0),
                            'Most Common': str(feature_stats.get('most_common', 'N/A')),
                            'Frequency': f"{feature_stats.get('most_common_pct', 0):.2f}%",
                            'Cardinality': f"{feature_stats.get('cardinality', 0):.3f}",
                            'Missing %': f"{feature_stats.get('missing_pct', 0):.2f}%"
                        })
                    
                    cat_df = pd.DataFrame(cat_data)
                    st.dataframe(cat_df, use_container_width=True)
                
                st.divider()
                
                # Outlier analysis
                if 'outlier_analysis' in stats:
                    st.subheader("🔍 Outlier Detection Results")
                    
                    outlier_data = []
                    for feature, outlier_stats in stats['outlier_analysis'].items():
                        outlier_pct = outlier_stats.get('outlier_percentage', 0)
                        status = "🔴 High" if outlier_pct > 10 else "🟡 Moderate" if outlier_pct > 5 else "🟢 Normal"
                        
                        outlier_data.append({
                            'Feature': feature,
                            'Outlier Count': outlier_stats.get('outlier_count', 0),
                            'Outlier %': f"{outlier_pct:.2f}%",
                            'Status': status,
                            'Lower Bound': f"{outlier_stats.get('lower_bound', 0):.2f}",
                            'Upper Bound': f"{outlier_stats.get('upper_bound', 0):.2f}"
                        })
                    
                    outlier_df = pd.DataFrame(outlier_data)
                    st.dataframe(outlier_df, use_container_width=True)
                
                # Data quality report
                if 'data_quality_report' in results and 'wrangling_log' in results['data_quality_report']:
                    st.divider()
                    st.subheader("🔧 Data Wrangling Steps")
                    
                    with st.expander("View Data Cleaning Log", expanded=True):
                        for step in results['data_quality_report']['wrangling_log']:
                            st.markdown(f"✓ {step}")
            else:
                st.info("Run analysis to see detailed statistics")
        else:
            st.info("👈 Upload data and run analysis first")
    
    with tab3:
        st.header("📈 Exploratory Data Analysis Visualizations")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            if 'eda_plots' in results:
                st.info("💡 All plots are interactive. Hover over them for details.")
                
                for plot_name, fig in results['eda_plots'].items():
                    st.subheader(f"📊 {plot_name}")
                    st.pyplot(fig)
                    st.divider()
            else:
                st.info("EDA plots are being generated...")
        else:
            st.info("👈 Upload data and run analysis first")
    
    with tab4:
        st.header("🎯 Model Training & Performance Results")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            # Selected research question
            if 'selected_question' in results and results['selected_question']:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Research Question:** {results['selected_question']['question']}")
                st.markdown(f"**Task Type:** {results['selected_question']['type'].replace('_', ' ').title()}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Task identification
            st.subheader("🎯 Task Identification")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Task Type:** {results['task_type'].title()}")
            with col2:
                st.info(f"**Target Variable:** {results['target_column']}")
            
            st.divider()
            
            # Model comparison
            st.subheader("🤖 Model Performance Comparison")
            if 'model_results' in results and results['model_results']:
                model_df = pd.DataFrame(results['model_results']).T
                
                # Highlight best performance
                st.dataframe(
                    model_df.style.highlight_max(axis=0, color='lightgreen')
                              .highlight_min(axis=0, color='lightcoral', subset=[col for col in model_df.columns if 'RMSE' in col or 'MAE' in col or 'MAPE' in col]),
                    use_container_width=True
                )
                
                st.divider()
                
                # Best model highlight
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"### 🏆 Best Model: {results['best_model']}")
                
                best_metrics = results['model_results'][results['best_model']]
                
                cols = st.columns(len(best_metrics))
                for idx, (metric, value) in enumerate(best_metrics.items()):
                    with cols[idx]:
                        if value is not None:
                            st.metric(metric, f"{value:.4f}")
                        else:
                            st.metric(metric, "N/A")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.divider()
                
                # Model comparison visualization
                if 'model_comparison_plot' in results:
                    st.subheader("📊 Visual Model Comparison")
                    st.pyplot(results['model_comparison_plot'])
            
            # Feature importance
            if 'feature_importance' in results and results['feature_importance'] is not None:
                st.divider()
                st.subheader("🎯 Feature Importance Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(results['feature_importance'].head(15), use_container_width=True)
                
                with col2:
                    st.markdown("**Interpretation:**")
                    top_feature = results['feature_importance'].iloc[0]
                    st.write(f"Most important feature: **{top_feature['Feature']}**")
                    st.write(f"Importance score: **{top_feature['Importance']:.4f}**")
                    
                    top_5_importance = results['feature_importance'].head(5)['Importance'].sum()
                    st.write(f"Top 5 features account for **{top_5_importance:.1%}** of predictive power")
                
                # Feature importance plot
                if 'feature_importance_plot' in results:
                    st.pyplot(results['feature_importance_plot'])
            
            # Key insights
            if 'insights' in results:
                st.divider()
                st.subheader("🔍 Key Insights & Findings")
                
                for idx, insight in enumerate(results['insights'], 1):
                    st.markdown(f"**{idx}.** {insight}")
        else:
            st.info("👈 Upload data and run analysis first")
    
    with tab5:
        st.header("⬇️ Download Analysis Results")
        
        if st.session_state.analyzed and st.session_state.results:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ Analysis Complete!")
            st.markdown("Download your comprehensive results below:")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📄 Full Report")
                if 'report_bytes' in st.session_state.results:
                    st.download_button(
                        label="📥 Download Report (DOCX)",
                        data=st.session_state.results['report_bytes'],
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        type="primary"
                    )
                    st.caption("Comprehensive analysis report with all findings")
            
            with col2:
                st.subheader("💾 Trained Model")
                if 'model_bytes' in st.session_state.results:
                    st.download_button(
                        label="📥 Download Model (PKL)",
                        data=st.session_state.results['model_bytes'],
                        file_name=f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                        type="primary"
                    )
                    st.caption("Best performing model for predictions")
            
            with col3:
                st.subheader("📊 Performance Metrics")
                if 'model_results' in st.session_state.results:
                    metrics_df = pd.DataFrame(st.session_state.results['model_results']).T
                    csv = metrics_df.to_csv()
                    st.download_button(
                        label="📥 Download Metrics (CSV)",
                        data=csv,
                        file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                    st.caption("All model performance metrics")
            
            st.divider()
            
            # Additional downloads
            st.subheader("📦 Additional Resources")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'descriptive_stats' in st.session_state.results:
                    stats_json = pd.DataFrame(st.session_state.results['descriptive_stats']['overview'], index=[0]).to_json()
                    st.download_button(
                        label="📥 Download Statistics (JSON)",
                        data=stats_json,
                        file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col2:
                if 'feature_importance' in st.session_state.results and st.session_state.results['feature_importance'] is not None:
                    fi_csv = st.session_state.results['feature_importance'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Feature Importance (CSV)",
                        data=fi_csv,
                        file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.divider()
            
            st.markdown("### 🎉 What's Next?")
            st.markdown("""
            **Recommended Actions:**
            1. 📖 **Review the comprehensive report** for detailed insights and methodology
            2. 💾 **Deploy the trained model** for predictions on new data
            3. 📊 **Share metrics** with stakeholders for decision-making
            4. 🔄 **Iterate and improve** by trying different configurations
            5. 🎯 **Validate predictions** on holdout data before production deployment
            6. 📈 **Monitor performance** regularly and retrain with fresh data
            """)
        else:
            st.info("👈 Run analysis first to generate downloadable results")
            
            st.markdown("### 📋 What You'll Get:")
            st.markdown("""
            - 📄 **Comprehensive Report** (20+ pages with visualizations)
            - 💾 **Trained ML Model** (production-ready)
            - 📊 **Performance Metrics** (detailed comparison)
            - 📈 **Feature Importance** (top influential factors)
            - 🔍 **Data Quality Assessment** (complete audit)
            """)

def run_analysis(df, research_question, test_size, cv_folds, selected_question_idx=0):
    """Execute the enhanced analysis pipeline"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize engine
        status_text.text("🔧 Initializing enhanced analysis engine...")
        progress_bar.progress(5)
        
        engine = EnhancedAIAnalysisEngine(df, research_question)
        
        # Generate research questions if not already done
        status_text.text("🎯 Generating research questions...")
        progress_bar.progress(10)
        engine.generate_research_questions()
        
        # Perform descriptive statistics
        status_text.text("📊 Calculating descriptive statistics...")
        progress_bar.progress(20)
        engine.perform_descriptive_statistics()
        
        # Data wrangling
        status_text.text("🧹 Performing data wrangling...")
        progress_bar.progress(30)
        engine.perform_data_wrangling()
        
        # Run pipeline
        status_text.text("🤖 Training multiple models...")
        progress_bar.progress(50)
        
        results = engine.run_full_pipeline(
            selected_question_idx=selected_question_idx,
            test_size=test_size,
            cv_folds=cv_folds
        )
        
        # Generate visualizations
        status_text.text("📊 Creating comprehensive visualizations...")
        progress_bar.progress(70)
        
        viz = EnhancedVisualizationEngine()
        results['eda_plots'] = viz.generate_all_eda_plots(df)
        
        # Model comparison plot
        if 'model_results' in results:
            results['model_comparison_plot'] = viz.plot_model_comparison(
                results['model_results'], 
                results['task_type']
            )
        
        # Feature importance plot
        if 'feature_importance' in results and results['feature_importance'] is not None:
            results['feature_importance_plot'] = viz.plot_feature_importance(
                results['feature_importance']
            )
        
        # Generate report
        status_text.text("📄 Generating comprehensive report...")
        progress_bar.progress(85)
        
        report_gen = EnhancedReportGenerator()
        results['report_bytes'] = report_gen.generate_full_report(
            research_question=research_question,
            task_type=results['task_type'],
            target_col=results['target_column'],
            model_results=results['model_results'],
            feature_importance=results.get('feature_importance'),
            insights=results['insights'],
            descriptive_stats=results.get('descriptive_stats'),
            data_quality_report=results.get('data_quality_report')
        )
        
        # Save model
        status_text.text("💾 Serializing trained model...")
        progress_bar.progress(95)
        
        model_persist = ModelPersistence()
        results['model_bytes'] = model_persist.serialize_model(results['best_model_object'])
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        st.session_state.analyzed = True
        st.session_state.results = results
        
        # Clean up
        gc.collect()
        
        st.balloons()
        st.success("🎉 Analysis complete! Check all tabs for comprehensive results.")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)
        status_text.text("Analysis failed")
        progress_bar.empty()

if __name__ == "__main__":
    main()
