"""
AI Data Analysis Agent - Enhanced Analysis Engine v2.0
Advanced ML pipeline with intelligent research question generation and expanded capabilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor, BaggingClassifier, BaggingRegressor)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, 
                                 Lasso, ElasticNet, SGDClassifier, SGDRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                            r2_score, mean_squared_error, mean_absolute_error,
                            silhouette_score, classification_report, precision_score, recall_score)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedAIAnalysisEngine:
    """Advanced analysis engine with auto research questions and expanded capabilities"""
    
    def __init__(self, dataframe, research_question="", chunk_size=50000):
        self.df = dataframe.copy()
        self.research_question = research_question.lower()
        self.target_column = None
        self.task_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = {}
        self.feature_importance = None
        self.label_encoders = {}
        self.scaler = None
        self.chunk_size = chunk_size
        self.descriptive_stats = {}
        self.data_quality_report = {}
        self.generated_research_questions = []
        
    def generate_research_questions(self):
        """Automatically generate intelligent research questions based on dataset analysis"""
        questions = []
        
        # Analyze column names and types
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Advanced pattern matching with priority scoring
        target_candidates = {}
        
        # Comprehensive outcome patterns with priorities
        outcome_patterns = {
            'target': 10, 'label': 10, 'y': 9, 'outcome': 9, 'result': 8,
            'status': 7, 'approved': 9, 'rejected': 9, 'accepted': 9,
            'churn': 10, 'fraud': 10, 'default': 9, 'risk': 8,
            'price': 9, 'revenue': 9, 'sales': 8, 'profit': 8, 'cost': 7,
            'value': 7, 'amount': 7, 'total': 6,
            'score': 8, 'rating': 8, 'grade': 7, 'rank': 7,
            'class': 9, 'category': 8, 'type': 6, 'group': 6,
            'diagnosis': 9, 'disease': 9, 'condition': 8,
            'success': 8, 'failure': 8, 'win': 7, 'loss': 7,
            'conversion': 9, 'response': 8, 'click': 7, 'purchase': 8,
            'satisfaction': 8, 'quality': 7, 'performance': 7
        }
        
        # Score each column as potential target
        for col in self.df.columns:
            col_lower = col.lower()
            max_score = 0
            for pattern, priority in outcome_patterns.items():
                if pattern in col_lower:
                    max_score = max(max_score, priority)
            
            if max_score > 0:
                # Additional scoring based on position (last columns often targets)
                position_bonus = (len(self.df.columns) - list(self.df.columns).index(col)) / len(self.df.columns)
                final_score = max_score + position_bonus
                target_candidates[col] = final_score
        
        # Sort by score
        sorted_targets = sorted(target_candidates.items(), key=lambda x: x[1], reverse=True)
        
        # Generate classification questions for top scored categorical/low-cardinality targets
        for col, score in sorted_targets[:5]:
            unique_vals = self.df[col].nunique()
            total_vals = len(self.df[col].dropna())
            
            if unique_vals < 20 and unique_vals >= 2:  # Classification candidate
                # Determine if binary or multiclass
                class_type = "binary" if unique_vals == 2 else "multiclass"
                
                # Get actual values for context
                top_values = self.df[col].value_counts().head(3).index.tolist()
                value_str = ", ".join([str(v) for v in top_values[:2]])
                
                # Create contextual question
                if 'churn' in col.lower() or 'attrition' in col.lower():
                    question_text = f"Which customers are most likely to churn? (Predict {col})"
                elif 'fraud' in col.lower():
                    question_text = f"Can we detect fraudulent transactions? (Predict {col})"
                elif 'approved' in col.lower() or 'accepted' in col.lower():
                    question_text = f"What factors determine approval/rejection? (Predict {col})"
                elif 'satisfaction' in col.lower() or 'rating' in col.lower():
                    question_text = f"What drives customer satisfaction levels? (Predict {col})"
                elif 'diagnosis' in col.lower() or 'disease' in col.lower():
                    question_text = f"Can we predict medical outcomes? (Predict {col})"
                else:
                    question_text = f"What determines {col}? Predict: {value_str}..."
                
                questions.append({
                    'question': question_text,
                    'target': col,
                    'type': 'classification',
                    'rationale': f"{class_type.title()} classification with {unique_vals} classes. Distribution analysis shows potential patterns."
                })
        
        # Generate regression questions for continuous numeric targets
        high_cardinality_numeric = [col for col in numeric_cols 
                                    if self.df[col].nunique() > 20 and 
                                    col not in [q['target'] for q in questions]]
        
        for col in high_cardinality_numeric[:3]:
            # Create contextual question based on column name
            if 'price' in col.lower() or 'cost' in col.lower():
                question_text = f"What factors influence pricing? (Predict {col})"
            elif 'revenue' in col.lower() or 'sales' in col.lower() or 'profit' in col.lower():
                question_text = f"How can we forecast revenue/sales? (Predict {col})"
            elif 'age' in col.lower() or 'time' in col.lower() or 'duration' in col.lower():
                question_text = f"What determines the duration/age? (Predict {col})"
            elif 'score' in col.lower() or 'rating' in col.lower():
                question_text = f"What drives the score/rating? (Predict {col})"
            else:
                # Analyze correlation with other features
                corr_features = []
                for other_col in numeric_cols[:5]:
                    if other_col != col:
                        corr = abs(self.df[[col, other_col]].corr().iloc[0, 1])
                        if corr > 0.3:
                            corr_features.append(other_col)
                
                if corr_features:
                    question_text = f"How do {', '.join(corr_features[:2])} impact {col}?"
                else:
                    question_text = f"What are the key drivers of {col}?"
            
            # Calculate stats for rationale
            col_range = self.df[col].max() - self.df[col].min()
            col_mean = self.df[col].mean()
            
            questions.append({
                'question': question_text,
                'target': col,
                'type': 'regression',
                'rationale': f"Continuous variable (range: {col_range:.2f}, mean: {col_mean:.2f}). Regression models can predict values."
            })
        
        # Advanced: Feature importance question
        if len(numeric_cols) >= 3 and len(questions) < 5:
            questions.append({
                'question': f"Which features are most important for predicting {questions[0]['target'] if questions else 'outcomes'}?",
                'target': questions[0]['target'] if questions else numeric_cols[-1],
                'type': 'feature_analysis',
                'rationale': "Feature importance analysis reveals which variables drive predictions"
            })
        
        # Time series if date column exists
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols and len(numeric_cols) > 0 and len(questions) < 5:
            questions.append({
                'question': f"How has {numeric_cols[0]} changed over time? (Time series analysis)",
                'target': numeric_cols[0],
                'type': 'time_series',
                'rationale': f"Date column '{date_cols[0]}' detected. Temporal patterns can be analyzed."
            })
        
        # Ensure we have at least one question
        if len(questions) == 0:
            # Fallback: use last column
            last_col = self.df.columns[-1]
            unique_vals = self.df[last_col].nunique()
            
            if unique_vals < 20:
                questions.append({
                    'question': f"Can we predict {last_col}?",
                    'target': last_col,
                    'type': 'classification',
                    'rationale': f"Target detection: {last_col} has {unique_vals} categories"
                })
            else:
                questions.append({
                    'question': f"What influences {last_col}?",
                    'target': last_col,
                    'type': 'regression',
                    'rationale': f"Target detection: {last_col} is continuous"
                })
        
        self.generated_research_questions = questions[:5]
        return self.generated_research_questions
    
    def perform_descriptive_statistics(self):
        """Comprehensive descriptive statistics and data profiling"""
        import pandas as pd
        
        stats_report = {
            'overview': {},
            'numeric_summary': {},
            'categorical_summary': {},
            'distribution_analysis': {},
            'outlier_analysis': {}
        }
        
        # Overview
        stats_report['overview'] = {
            'total_rows': int(len(self.df)),
            'total_columns': int(len(self.df.columns)),
            'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / (1024**2)),
            'duplicate_rows': int(self.df.duplicated().sum()),
            'duplicate_percentage': float((self.df.duplicated().sum() / len(self.df)) * 100),
            'missing_cells': int(self.df.isnull().sum().sum()),
            'missing_percentage': float((self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100)
        }
        
        # Numeric features
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            try:
                data = self.df[col].dropna()
                
                if len(data) == 0:
                    continue
                    
                # Calculate mode safely
                mode_values = data.mode()
                mode_val = mode_values.iloc[0] if len(mode_values) > 0 else None
                
                # Calculate statistics
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()
                var_val = data.var()
                min_val = data.min()
                max_val = data.max()
                q1_val = data.quantile(0.25)
                q3_val = data.quantile(0.75)
                skew_val = data.skew()
                kurt_val = data.kurtosis()
                
                stats_report['numeric_summary'][col] = {
                    'count': int(len(data)),
                    'mean': float(mean_val) if not pd.isna(mean_val) else 0.0,
                    'median': float(median_val) if not pd.isna(median_val) else 0.0,
                    'mode': float(mode_val) if mode_val is not None and not pd.isna(mode_val) else None,
                    'std': float(std_val) if not pd.isna(std_val) else 0.0,
                    'variance': float(var_val) if not pd.isna(var_val) else 0.0,
                    'min': float(min_val) if not pd.isna(min_val) else 0.0,
                    'max': float(max_val) if not pd.isna(max_val) else 0.0,
                    'range': float(max_val - min_val) if not pd.isna(max_val) and not pd.isna(min_val) else 0.0,
                    'q1': float(q1_val) if not pd.isna(q1_val) else 0.0,
                    'q3': float(q3_val) if not pd.isna(q3_val) else 0.0,
                    'iqr': float(q3_val - q1_val) if not pd.isna(q3_val) and not pd.isna(q1_val) else 0.0,
                    'skewness': float(skew_val) if not pd.isna(skew_val) else 0.0,
                    'kurtosis': float(kurt_val) if not pd.isna(kurt_val) else 0.0,
                    'cv': float((std_val / mean_val * 100)) if mean_val != 0 and not pd.isna(mean_val) and not pd.isna(std_val) else 0.0,
                    'missing': int(self.df[col].isnull().sum()),
                    'missing_pct': float((self.df[col].isnull().sum() / len(self.df)) * 100),
                    'zeros': int((data == 0).sum()),
                    'negative': int((data < 0).sum()),
                    'unique_values': int(data.nunique())
                }
                
                # Outlier detection using IQR
                iqr = q3_val - q1_val
                lower_bound = q1_val - 1.5 * iqr
                upper_bound = q3_val + 1.5 * iqr
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                stats_report['outlier_analysis'][col] = {
                    'outlier_count': int(len(outliers)),
                    'outlier_percentage': float((len(outliers) / len(data)) * 100) if len(data) > 0 else 0.0,
                    'lower_bound': float(lower_bound) if not pd.isna(lower_bound) else 0.0,
                    'upper_bound': float(upper_bound) if not pd.isna(upper_bound) else 0.0,
                }
            except Exception as e:
                print(f"Warning: Could not process numeric column '{col}': {str(e)}")
                continue
        
        # Categorical features
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            try:
                data = self.df[col].dropna()
                
                if len(data) == 0:
                    continue
                    
                value_counts = data.value_counts()
                
                stats_report['categorical_summary'][col] = {
                    'count': int(len(data)),
                    'unique_values': int(data.nunique()),
                    'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_common_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'most_common_pct': float((value_counts.iloc[0] / len(data) * 100)) if len(value_counts) > 0 else 0.0,
                    'least_common': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                    'least_common_freq': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'missing': int(self.df[col].isnull().sum()),
                    'missing_pct': float((self.df[col].isnull().sum() / len(self.df)) * 100),
                    'cardinality': float(data.nunique() / len(data)) if len(data) > 0 else 0.0,
                    'mode_frequency': float(value_counts.iloc[0] / len(data)) if len(value_counts) > 0 and len(data) > 0 else 0.0
                }
            except Exception as e:
                print(f"Warning: Could not process categorical column '{col}': {str(e)}")
                continue
        
        self.descriptive_stats = stats_report
        return stats_report
    
    def perform_data_wrangling(self):
        """Advanced data cleaning and transformation"""
        import pandas as pd
        
        wrangling_log = []
        
        # 1. Handle duplicates
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.df = self.df.drop_duplicates()
            wrangling_log.append(f"Removed {dup_count} duplicate rows")
        
        # 2. Handle missing values intelligently
        columns_to_drop = []
        for col in self.df.columns:
            try:
                # Ensure we're working with a Series
                col_data = self.df[col]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                
                missing_count = col_data.isnull().sum()
                # Convert numpy/pandas types to native Python types
                if hasattr(missing_count, 'item'):
                    missing_count = missing_count.item()
                else:
                    missing_count = int(missing_count)
                
                total_rows = len(self.df)
                missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
                
                if missing_pct > 50:
                    columns_to_drop.append((col, missing_pct))
                elif missing_pct > 0:
                    if col_data.dtype in ['int64', 'float64', 'int32', 'float32', 'float16']:
                        # Use median imputation for numeric
                        median_val = col_data.median()
                        if pd.notna(median_val):
                            self.df[col] = col_data.fillna(median_val)
                            wrangling_log.append(f"Median imputed '{col}' ({missing_pct:.1f}% missing)")
                    else:
                        # Mode for categorical
                        mode_val = col_data.mode()
                        if len(mode_val) > 0:
                            self.df[col] = col_data.fillna(mode_val.iloc[0])
                            wrangling_log.append(f"Mode imputed '{col}' ({missing_pct:.1f}% missing)")
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {str(e)}")
                continue
        
        # Drop columns with >50% missing
        if columns_to_drop:
            cols_to_remove = [col for col, _ in columns_to_drop]
            self.df = self.df.drop(columns=cols_to_remove)
            for col, pct in columns_to_drop:
                wrangling_log.append(f"Dropped column '{col}' ({pct:.1f}% missing)")
        
        # 3. Handle outliers in numeric columns (only if reasonable number of columns)
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) <= 50:  # Only process if manageable number of columns
            for col in numeric_cols:
                try:
                    col_data = self.df[col]
                    if isinstance(col_data, pd.DataFrame):
                        col_data = col_data.iloc[:, 0]
                    
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    # Convert to native Python int
                    if hasattr(outlier_count, 'item'):
                        outlier_count = outlier_count.item()
                    else:
                        outlier_count = int(outlier_count)
                    
                    if outlier_count > 0:
                        self.df[col] = col_data.clip(lower=lower_bound, upper=upper_bound)
                        wrangling_log.append(f"Capped {outlier_count} extreme outliers in '{col}'")
                except Exception as e:
                    print(f"Warning: Could not process outliers for '{col}': {str(e)}")
                    continue
        
        # 4. Normalize column names
        original_cols = self.df.columns.tolist()
        self.df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in self.df.columns]
        if original_cols != self.df.columns.tolist():
            wrangling_log.append("Standardized column names")
        
        # 5. Convert data types appropriately
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.05:  # Low cardinality
                    self.df[col] = self.df[col].astype('category')
                    wrangling_log.append(f"Converted '{col}' to category type")
            except Exception as e:
                print(f"Warning: Could not convert '{col}' to category: {str(e)}")
                continue
        
        self.data_quality_report['wrangling_log'] = wrangling_log
        return wrangling_log
    
    def detect_target_column(self):
        """Enhanced target detection with research questions"""
        if self.generated_research_questions and not self.research_question:
            # Use first generated question
            self.target_column = self.generated_research_questions[0]['target']
            return self.target_column
        
        if self.research_question:
            for col in self.df.columns:
                if col.lower() in self.research_question:
                    self.target_column = col
                    return col
        
        target_patterns = ['target', 'label', 'y', 'outcome', 'class', 
                          'prediction', 'result', 'status', 'approved',
                          'churn', 'fraud', 'default', 'price', 'revenue',
                          'sales', 'value', 'score']
        
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in target_patterns):
                self.target_column = col
                return col
        
        self.target_column = self.df.columns[-1]
        return self.target_column
    
    def train_expanded_models(self, X_train, X_test, y_train, y_test, task_type, cv_folds=5):
        """Train comprehensive set of models with hyperparameter tuning"""
        results = {}
        
        if task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
                'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Naive Bayes': GaussianNB(),
                'SGD Classifier': SGDClassifier(random_state=42),
                'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            # Handle imbalanced data
            if len(np.unique(y_train)) == 2:
                try:
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                except:
                    pass
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    try:
                        y_pred_proba = model.predict_proba(X_test)
                        if len(np.unique(y_test)) == 2:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    except:
                        auc = None
                    
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                    
                    results[name] = {
                        'Accuracy': round(accuracy, 4),
                        'Precision': round(precision, 4),
                        'Recall': round(recall, 4),
                        'F1 Score': round(f1, 4),
                        'AUC': round(auc, 4) if auc else None,
                        'CV Mean': round(cv_scores.mean(), 4),
                        'CV Std': round(cv_scores.std(), 4)
                    }
                    
                    self.trained_models[name] = model
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
        
        else:  # Regression
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Lasso Regression': Lasso(alpha=1.0, random_state=42),
                'ElasticNet': ElasticNet(alpha=1.0, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=200, random_state=42),
                'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'SGD Regressor': SGDRegressor(random_state=42),
                'MLP Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                    
                    results[name] = {
                        'R² Score': round(r2, 4),
                        'RMSE': round(rmse, 4),
                        'MAE': round(mae, 4),
                        'MAPE': round(mape, 4),
                        'CV Mean': round(cv_scores.mean(), 4),
                        'CV Std': round(cv_scores.std(), 4)
                    }
                    
                    self.trained_models[name] = model
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
        
        return results
    
    def run_full_pipeline(self, selected_question_idx=0, test_size=0.2, cv_folds=5):
        """Execute enhanced analysis pipeline"""
        
        # Generate research questions if not provided
        if not self.generated_research_questions:
            self.generate_research_questions()
        
        # Perform descriptive statistics
        self.perform_descriptive_statistics()
        
        # Data wrangling
        self.perform_data_wrangling()
        
        # Select research question
        if self.generated_research_questions:
            selected_q = self.generated_research_questions[selected_question_idx]
            self.target_column = selected_q['target']
            self.task_type = selected_q['type']
        else:
            self.detect_target_column()
            self.identify_task_type()
        
        # Preprocess and train
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Train expanded model set
        model_results = self.train_expanded_models(X_train, X_test, y_train, y_test, self.task_type, cv_folds)
        
        # Extract feature importance
        if 'Random Forest' in self.trained_models:
            self.extract_feature_importance('Random Forest')
        
        # Generate insights
        insights = self.generate_enhanced_insights(model_results)
        
        # Determine best model
        if self.task_type == 'classification':
            best_model = max(model_results.items(), key=lambda x: x[1].get('F1 Score', 0))[0]
        else:
            best_model = max(model_results.items(), key=lambda x: x[1].get('R² Score', 0))[0]
        
        return {
            'research_questions': self.generated_research_questions,
            'selected_question': self.generated_research_questions[selected_question_idx] if self.generated_research_questions else None,
            'descriptive_stats': self.descriptive_stats,
            'data_quality_report': self.data_quality_report,
            'target_column': self.target_column,
            'task_type': self.task_type,
            'model_results': model_results,
            'best_model': best_model,
            'best_model_object': self.trained_models[best_model],
            'feature_importance': self.feature_importance,
            'insights': insights,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
    
    def identify_task_type(self):
        """Identify task type"""
        if self.target_column is None:
            self.detect_target_column()
        
        target_data = self.df[self.target_column]
        
        if target_data.isnull().all():
            return 'clustering'
        
        unique_values = target_data.nunique()
        total_values = len(target_data.dropna())
        
        if target_data.dtype == 'object' or unique_values < 20:
            self.task_type = 'classification'
        elif unique_values / total_values > 0.05:
            self.task_type = 'regression'
        else:
            self.task_type = 'classification'
        
        return self.task_type
    
    def preprocess_data(self):
        """Enhanced preprocessing"""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        if y.isnull().any():
            y = y.fillna(y.mode()[0] if self.task_type == 'classification' else y.median())
        
        if self.task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders['target'] = le
        
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_features:
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        self.scaler = RobustScaler()  # More robust to outliers
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X, y
    
    def extract_feature_importance(self, model_name='Random Forest'):
        """Extract feature importance"""
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.X_train.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            self.feature_importance = importance_df
            return importance_df
        
        return None
    
    def generate_enhanced_insights(self, model_results):
        """Generate comprehensive insights"""
        insights = []
        
        # Model performance insights
        if self.task_type == 'classification':
            best_model = max(model_results.items(), key=lambda x: x[1].get('F1 Score', 0))
            insights.append(f"Best model: {best_model[0]} with {best_model[1]['F1 Score']:.1%} F1 score")
            
            avg_accuracy = np.mean([v.get('Accuracy', 0) for v in model_results.values()])
            insights.append(f"Average model accuracy: {avg_accuracy:.1%}")
        else:
            best_model = max(model_results.items(), key=lambda x: x[1].get('R² Score', 0))
            insights.append(f"Best model: {best_model[0]} with R² = {best_model[1]['R² Score']:.3f}")
        
        # Feature insights
        if self.feature_importance is not None:
            top_feature = self.feature_importance.iloc[0]
            insights.append(f"Most influential feature: {top_feature['Feature']} (importance: {top_feature['Importance']:.3f})")
            
            top_3 = self.feature_importance.head(3)['Feature'].tolist()
            insights.append(f"Top 3 features account for major predictive power: {', '.join(top_3)}")
        
        # Data quality insights
        if self.descriptive_stats:
            missing_pct = self.descriptive_stats['overview']['missing_percentage']
            if missing_pct > 10:
                insights.append(f"Dataset had {missing_pct:.1f}% missing values - advanced imputation applied")
            
            dup_pct = self.descriptive_stats['overview']['duplicate_percentage']
            if dup_pct > 5:
                insights.append(f"Removed {dup_pct:.1f}% duplicate records during preprocessing")
        
        return insights
