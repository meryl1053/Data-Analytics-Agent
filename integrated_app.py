"""
AI Data Analysis Agent - Main Streamlit Application
Upload → Analyze → Download in 3 steps
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from pathlib import Path

# Import custom modules (will be created)
try:
    from analysis_engine import AIAnalysisEngine
    from visualization_module import VisualizationEngine
    from report_generator import ReportGenerator
    from utils_module import DataValidator, ModelPersistence
except ImportError:
    st.warning("⚠️ Core modules not found. Please ensure all files are in the same directory.")

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent",
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
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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

def main():
    """Main application logic"""
    
    # Header
    st.markdown('<p class="main-header">🤖 AI Data Analysis Agent</p>', unsafe_allow_html=True)
    st.markdown("**Upload → Analyze → Download** | Automated ML in 60 seconds")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Analysis settings
        st.subheader("Model Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        use_smote = st.checkbox("Use SMOTE (Imbalanced Data)", value=True)
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ About")
        st.info("""
        This agent automatically:
        - Detects target variable
        - Identifies task type
        - Preprocesses data
        - Trains multiple models
        - Generates insights
        - Creates professional reports
        """)
        
        st.subheader("📊 Supported Tasks")
        st.markdown("""
        - 🎯 Classification
        - 📈 Regression  
        - 🔍 Clustering
        """)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload & Configure",
        "📊 EDA Results",
        "🎯 Model Results",
        "⬇️ Download"
    ])
    
    with tab1:
        st.header("Step 1: Upload Your Dataset")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                
                # Validate data
                validator = DataValidator()
                is_valid, message = validator.validate_dataframe(df)
                
                if not is_valid:
                    st.error(f"❌ Data validation failed: {message}")
                    return
                
                # Display dataset info
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", f"{len(df.columns):,}")
                with col3:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Missing %", f"{missing_pct:.1f}%")
                with col4:
                    health = validator.get_dataset_health_score(df)
                    st.metric("Health Score", f"{health['health_score']:.1f}/100")
                
                # Dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column info
                with st.expander("📋 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.count().values,
                        'Null %': ((df.isnull().sum() / len(df)) * 100).round(2).values
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                st.divider()
                
                # Research question
                st.header("Step 2: Define Your Research Question")
                research_question = st.text_input(
                    "What do you want to predict or analyze?",
                    placeholder="e.g., 'Predict customer churn' or 'Forecast sales revenue'",
                    help="This helps the AI identify the target variable"
                )
                
                # Start analysis button
                st.divider()
                if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                    if not research_question:
                        st.warning("⚠️ Please enter a research question first!")
                    else:
                        run_analysis(df, research_question, test_size, cv_folds, use_smote)
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        else:
            # Landing state
            st.info("👆 Upload a dataset to begin analysis")
            
            # Example datasets
            st.subheader("🎯 Try Sample Dataset")
            if st.button("Load Sample Loan Data"):
                try:
                    df = pd.read_csv('data/sample_loan_data.csv')
                    st.session_state.df = df
                    st.rerun()
                except FileNotFoundError:
                    st.error("Sample data not found. Run setup.py first!")
    
    with tab2:
        st.header("📊 Exploratory Data Analysis")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            # Dataset summary
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Total Features", len(st.session_state.df.columns) - 1)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Numeric Features", len(numeric_cols))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Categorical Features", len(cat_cols))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display EDA plots if available
            if 'eda_plots' in results:
                st.subheader("Visual Analysis")
                
                for plot_name, fig in results['eda_plots'].items():
                    with st.expander(f"📊 {plot_name.replace('_', ' ').title()}", expanded=True):
                        st.pyplot(fig)
            else:
                st.info("Run analysis to see EDA visualizations")
        else:
            st.info("👈 Upload data and run analysis first")
    
    with tab3:
        st.header("🎯 Model Training Results")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            # Task identification
            st.subheader("Task Identification")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Task Type:** {results['task_type'].title()}")
            with col2:
                st.info(f"**Target Variable:** {results['target_column']}")
            
            # Model comparison
            st.subheader("Model Performance Comparison")
            if 'model_results' in results and results['model_results']:
                model_df = pd.DataFrame(results['model_results']).T
                st.dataframe(model_df.style.highlight_max(axis=0, color='lightgreen'), 
                           use_container_width=True)
                
                # Best model highlight
                st.success(f"✨ **Best Model:** {results['best_model']}")
            
            # Feature importance
            if 'feature_importance' in results and results['feature_importance'] is not None:
                st.subheader("Feature Importance")
                st.dataframe(results['feature_importance'].head(10), use_container_width=True)
            
            # Key insights
            if 'insights' in results:
                st.subheader("🔍 Key Insights")
                for insight in results['insights']:
                    st.markdown(f"- {insight}")
        else:
            st.info("👈 Upload data and run analysis first")
    
    with tab4:
        st.header("⬇️ Download Results")
        
        if st.session_state.analyzed and st.session_state.results:
            st.success("✅ Analysis complete! Download your results below:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download report
                if 'report_bytes' in st.session_state.results:
                    st.download_button(
                        label="📄 Download Report (DOCX)",
                        data=st.session_state.results['report_bytes'],
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
            
            with col2:
                # Download model
                if 'model_bytes' in st.session_state.results:
                    st.download_button(
                        label="💾 Download Model (PKL)",
                        data=st.session_state.results['model_bytes'],
                        file_name=f"trained_model_{datetime.now().strftime('%Y%m%d')}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            
            with col3:
                # Download metrics
                if 'model_results' in st.session_state.results:
                    metrics_df = pd.DataFrame(st.session_state.results['model_results']).T
                    csv = metrics_df.to_csv()
                    st.download_button(
                        label="📊 Download Metrics (CSV)",
                        data=csv,
                        file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.divider()
            st.markdown("### 🎉 What's Next?")
            st.markdown("""
            - 📖 Review the comprehensive report for detailed insights
            - 💾 Use the trained model for predictions on new data
            - 📊 Analyze the metrics to understand model performance
            - 🔄 Try different configurations to improve results
            """)
        else:
            st.info("👈 Run analysis first to generate downloadable results")

def run_analysis(df, research_question, test_size, cv_folds, use_smote):
    """Execute the full analysis pipeline"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize engine
        status_text.text("🔧 Initializing analysis engine...")
        progress_bar.progress(10)
        
        engine = AIAnalysisEngine(df, research_question)
        
        # Run pipeline
        status_text.text("🔍 Detecting target variable...")
        progress_bar.progress(20)
        
        status_text.text("🧹 Preprocessing data...")
        progress_bar.progress(40)
        
        status_text.text("🤖 Training models...")
        progress_bar.progress(60)
        
        results = engine.run_full_pipeline(
            test_size=test_size,
            cv_folds=cv_folds,
            use_smote=use_smote
        )
        
        # Generate visualizations
        status_text.text("📊 Creating visualizations...")
        progress_bar.progress(80)
        
        viz = VisualizationEngine()
        results['eda_plots'] = viz.generate_all_eda_plots(df)
        
        # Generate report
        status_text.text("📄 Generating report...")
        progress_bar.progress(90)
        
        report_gen = ReportGenerator()
        results['report_bytes'] = report_gen.generate_full_report(
            research_question=research_question,
            task_type=results['task_type'],
            target_col=results['target_column'],
            model_results=results['model_results'],
            feature_importance=results.get('feature_importance'),
            insights=results['insights']
        )
        
        # Save model
        model_persist = ModelPersistence()
        results['model_bytes'] = model_persist.serialize_model(results['best_model_object'])
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        st.session_state.analyzed = True
        st.session_state.results = results
        
        st.balloons()
        st.success("🎉 Analysis complete! Check the tabs above for results.")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        status_text.text("Analysis failed")
        progress_bar.empty()

if __name__ == "__main__":
    main()
