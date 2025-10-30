"""
AI Data Analysis Agent - Streamlit Cloud Compatible Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import gc

# Set page config FIRST
st.set_page_config(
    page_title="AI Data Analysis Agent v2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import custom modules with better error handling
try:
    from analysis_engine import EnhancedAIAnalysisEngine
    ANALYSIS_ENGINE_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Analysis engine not available: {str(e)}")
    ANALYSIS_ENGINE_AVAILABLE = False

try:
    from visualization_module import EnhancedVisualizationEngine
    VIZ_ENGINE_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Visualization module not available: {str(e)}")
    VIZ_ENGINE_AVAILABLE = False

try:
    from report_generator import EnhancedReportGenerator
    REPORT_GEN_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Report generator not available: {str(e)}")
    REPORT_GEN_AVAILABLE = False

try:
    from utils_module import DataValidator, ModelPersistence
    UTILS_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Utils module not available: {str(e)}")
    UTILS_AVAILABLE = False

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

def load_dataset(uploaded_file):
    """Load dataset with error handling"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_size = uploaded_file.size / (1024 * 1024)  # MB
        
        st.info(f"📦 Loading: {uploaded_file.name} ({file_size:.1f} MB)")
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"❌ Unsupported format: {file_extension}")
            return None
        
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def main():
    """Main application"""
    
    # Check if modules are available
    if not ANALYSIS_ENGINE_AVAILABLE:
        st.error("❌ Analysis engine module is missing. Please check your repository files.")
        st.stop()
    
    # Header
    st.markdown('<p class="main-header">🤖 AI Data Analysis Agent v2.0</p>', unsafe_allow_html=True)
    st.markdown("**Upload → Auto-Analyze → Download**")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🎯 Analysis Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        st.divider()
        
        st.subheader("ℹ️ About")
        st.info("""
        **Features:**
        - 🎯 Smart research questions
        - 📊 15+ visualizations
        - 📈 12+ ML models
        - 📄 Detailed reports
        - 💾 Up to 1GB files
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload & Questions",
        "📊 Statistics",
        "🎯 Results",
        "⬇️ Download"
    ])
    
    with tab1:
        st.header("Step 1: Upload Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Supports files up to 1GB"
        )
        
        if uploaded_file:
            df = load_dataset(uploaded_file)
            
            if df is not None:
                st.session_state.df = df
                
                # Preview
                st.subheader("Preview")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", f"{len(df.columns):,}")
                with col3:
                    missing = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Missing %", f"{missing:.1f}%")
                with col4:
                    mem = df.memory_usage(deep=True).sum() / (1024**2)
                    st.metric("Memory", f"{mem:.1f} MB")
                
                st.divider()
                
                # Generate questions
                st.header("Step 2: Research Questions")
                
                if st.button("🎯 Generate Questions", type="primary", use_container_width=True):
                    with st.spinner("Analyzing dataset..."):
                        try:
                            engine = EnhancedAIAnalysisEngine(df)
                            questions = engine.generate_research_questions()
                            st.session_state.research_questions = questions
                            st.success(f"✅ Generated {len(questions)} questions!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                
                # Display questions
                if st.session_state.research_questions:
                    st.subheader("📝 AI-Generated Questions")
                    
                    options = [f"Q{i+1}: {q['question']}" 
                              for i, q in enumerate(st.session_state.research_questions)]
                    
                    selected = st.radio("Choose question:", options, key="q_selector")
                    selected_idx = options.index(selected) if selected else 0
                    
                    if selected:
                        q = st.session_state.research_questions[selected_idx]
                        with st.expander("Details", expanded=True):
                            st.write(f"**Type:** {q['type']}")
                            st.write(f"**Target:** {q['target']}")
                            st.write(f"**Rationale:** {q['rationale']}")
                    
                    st.divider()
                    
                    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                        run_analysis(df, selected_idx, test_size, cv_folds)
        else:
            st.info("👆 Upload a dataset to begin")
    
    with tab2:
        st.header("📊 Statistics")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            if 'descriptive_stats' in results:
                stats = results['descriptive_stats']
                
                if 'overview' in stats:
                    st.subheader("Overview")
                    overview = stats['overview']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{overview.get('total_rows', 0):,}")
                    with col2:
                        st.metric("Columns", f"{overview.get('total_columns', 0):,}")
                    with col3:
                        st.metric("Missing %", f"{overview.get('missing_percentage', 0):.2f}%")
                
                if 'numeric_summary' in stats and stats['numeric_summary']:
                    st.subheader("Numeric Features")
                    
                    data = []
                    for feat, s in list(stats['numeric_summary'].items())[:10]:
                        data.append({
                            'Feature': feat,
                            'Mean': f"{s.get('mean', 0):.2f}",
                            'Median': f"{s.get('median', 0):.2f}",
                            'Std': f"{s.get('std', 0):.2f}",
                            'Min': f"{s.get('min', 0):.2f}",
                            'Max': f"{s.get('max', 0):.2f}"
                        })
                    
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("Run analysis first")
    
    with tab3:
        st.header("🎯 Results")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            # Task info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Task:** {results.get('task_type', 'N/A').title()}")
            with col2:
                st.info(f"**Target:** {results.get('target_column', 'N/A')}")
            
            # Models
            if 'model_results' in results:
                st.subheader("Model Performance")
                model_df = pd.DataFrame(results['model_results']).T
                st.dataframe(model_df.style.highlight_max(axis=0, color='lightgreen'), 
                           use_container_width=True)
                
                st.success(f"🏆 Best: {results.get('best_model', 'N/A')}")
            
            # Feature importance
            if 'feature_importance' in results and results['feature_importance'] is not None:
                st.subheader("Feature Importance")
                st.dataframe(results['feature_importance'].head(10), 
                           use_container_width=True)
            
            # Insights
            if 'insights' in results:
                st.subheader("Key Insights")
                for insight in results['insights']:
                    st.markdown(f"- {insight}")
        else:
            st.info("Run analysis first")
    
    with tab4:
        st.header("⬇️ Downloads")
        
        if st.session_state.analyzed and st.session_state.results:
            st.success("✅ Analysis complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'report_bytes' in st.session_state.results:
                    st.download_button(
                        "📄 Report (DOCX)",
                        st.session_state.results['report_bytes'],
                        f"report_{datetime.now().strftime('%Y%m%d')}.docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
            
            with col2:
                if 'model_bytes' in st.session_state.results:
                    st.download_button(
                        "💾 Model (PKL)",
                        st.session_state.results['model_bytes'],
                        f"model_{datetime.now().strftime('%Y%m%d')}.pkl",
                        "application/octet-stream",
                        use_container_width=True
                    )
            
            with col3:
                if 'model_results' in st.session_state.results:
                    csv = pd.DataFrame(st.session_state.results['model_results']).T.to_csv()
                    st.download_button(
                        "📊 Metrics (CSV)",
                        csv,
                        f"metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
        else:
            st.info("Run analysis first")

def run_analysis(df, selected_idx, test_size, cv_folds):
    """Run analysis pipeline"""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        status.text("🔧 Initializing...")
        progress.progress(10)
        
        engine = EnhancedAIAnalysisEngine(df, "")
        
        status.text("📊 Calculating statistics...")
        progress.progress(20)
        engine.perform_descriptive_statistics()
        
        status.text("🧹 Cleaning data...")
        progress.progress(30)
        engine.perform_data_wrangling()
        
        status.text("🤖 Training models...")
        progress.progress(50)
        
        results = engine.run_full_pipeline(
            selected_question_idx=selected_idx,
            test_size=test_size,
            cv_folds=cv_folds
        )
        
        if VIZ_ENGINE_AVAILABLE:
            status.text("📊 Creating visualizations...")
            progress.progress(70)
            viz = EnhancedVisualizationEngine()
            results['eda_plots'] = viz.generate_all_eda_plots(df)
        
        if REPORT_GEN_AVAILABLE:
            status.text("📄 Generating report...")
            progress.progress(85)
            
            q = st.session_state.research_questions[selected_idx]
            report_gen = EnhancedReportGenerator()
            results['report_bytes'] = report_gen.generate_full_report(
                research_question=q['question'],
                task_type=results['task_type'],
                target_col=results['target_column'],
                model_results=results['model_results'],
                feature_importance=results.get('feature_importance'),
                insights=results['insights'],
                descriptive_stats=results.get('descriptive_stats')
            )
        
        if UTILS_AVAILABLE:
            status.text("💾 Saving model...")
            progress.progress(95)
            persist = ModelPersistence()
            results['model_bytes'] = persist.serialize_model(results['best_model_object'])
        
        progress.progress(100)
        status.text("✅ Complete!")
        
        st.session_state.analyzed = True
        st.session_state.results = results
        
        gc.collect()
        st.balloons()
        st.success("🎉 Analysis complete! Check other tabs.")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)
        progress.empty()
        status.empty()

if __name__ == "__main__":
    main()
