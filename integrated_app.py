"""
AI Data Analysis Agent - Minimal Working Version
This version has all functionality in ONE file to avoid import issues
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Simple Analysis Class
class SimpleAnalyzer:
    def __init__(self, df):
        self.df = df
        self.target = None
        self.task_type = None
        
    def detect_target(self):
        """Detect target column"""
        # Look for common patterns
        patterns = ['target', 'label', 'approved', 'churn', 'price', 'revenue', 'class']
        for col in self.df.columns:
            if any(p in col.lower() for p in patterns):
                self.target = col
                return col
        # Use last column
        self.target = self.df.columns[-1]
        return self.target
    
    def identify_task(self):
        """Identify task type"""
        if self.target is None:
            self.detect_target()
        
        unique = self.df[self.target].nunique()
        if unique < 20:
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        return self.task_type
    
    def generate_questions(self):
        """Generate research questions"""
        questions = []
        
        # Find potential targets
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        for col in self.df.columns[:5]:
            unique = self.df[col].nunique()
            if unique < 20 and unique > 1:
                questions.append({
                    'question': f"Can we predict {col}?",
                    'target': col,
                    'type': 'classification'
                })
        
        for col in numeric_cols[:3]:
            if self.df[col].nunique() > 20:
                questions.append({
                    'question': f"What factors influence {col}?",
                    'target': col,
                    'type': 'regression'
                })
        
        return questions[:5]
    
    def train_models(self, test_size=0.2):
        """Train models"""
        self.identify_task()
        
        # Prepare data
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        # Handle missing
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(X.mode().iloc[0])
        
        # Encode categorical
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Encode target if needed
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # Scale
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        results = {}
        
        if self.task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
            }
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results[name] = {
                    'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 4)
                }
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100)
            }
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results[name] = {
                    'R² Score': round(r2_score(y_test, y_pred), 4),
                    'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    'MAE': round(mean_absolute_error(y_test, y_pred), 4)
                }
        
        # Get best model
        if self.task_type == 'classification':
            best = max(results.items(), key=lambda x: x[1]['Accuracy'])[0]
        else:
            best = max(results.items(), key=lambda x: x[1]['R² Score'])[0]
        
        return {
            'target': self.target,
            'task_type': self.task_type,
            'results': results,
            'best_model': best
        }

# Main app
def main():
    st.markdown('<p class="main-header">🤖 AI Data Analysis Agent</p>', unsafe_allow_html=True)
    st.markdown("**Upload → Analyze → Results**")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        st.info("**Features:**\n- Auto target detection\n- 6 ML models\n- Smart insights")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "🎯 Results", "📊 Statistics"])
    
    with tab1:
        st.header("Upload Dataset")
        
        uploaded = st.file_uploader("Choose CSV or Excel", type=['csv', 'xlsx'])
        
        if uploaded:
            try:
                # Load
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                
                st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
                
                # Preview
                st.subheader("Preview")
                st.dataframe(df.head(10))
                
                # Info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    missing = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Missing %", f"{missing:.1f}%")
                
                st.divider()
                
                # Questions
                st.header("Research Questions")
                analyzer = SimpleAnalyzer(df)
                questions = analyzer.generate_questions()
                
                if questions:
                    options = [f"Q{i+1}: {q['question']}" for i, q in enumerate(questions)]
                    selected = st.radio("Select question:", options)
                    idx = options.index(selected) if selected else 0
                    
                    st.info(f"**Target:** {questions[idx]['target']}")
                    
                    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                        with st.spinner("Analyzing..."):
                            # Set target
                            analyzer.target = questions[idx]['target']
                            results = analyzer.train_models(test_size)
                            st.session_state.results = results
                            st.session_state.analyzed = True
                            st.success("✅ Analysis complete!")
                            st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload a dataset to begin")
    
    with tab2:
        st.header("Analysis Results")
        
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            # Task info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Task Type:** {results['task_type'].title()}")
            with col2:
                st.info(f"**Target:** {results['target']}")
            
            # Model results
            st.subheader("Model Performance")
            df_results = pd.DataFrame(results['results']).T
            st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'), 
                        use_container_width=True)
            
            st.success(f"🏆 Best Model: **{results['best_model']}**")
            
            # Best model metrics
            st.subheader("Best Model Metrics")
            best_metrics = results['results'][results['best_model']]
            cols = st.columns(len(best_metrics))
            for i, (metric, value) in enumerate(best_metrics.items()):
                cols[i].metric(metric, f"{value:.4f}")
        else:
            st.info("👈 Run analysis first")
    
    with tab3:
        st.header("Dataset Statistics")
        
        if st.session_state.analyzed and st.session_state.results:
            # Would show detailed stats here
            st.write("Statistics will appear here after analysis")
        else:
            st.info("👈 Run analysis first")

if __name__ == "__main__":
    main()
