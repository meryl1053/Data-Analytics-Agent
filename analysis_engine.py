"""
AI Data Analysis Agent - Analysis Engine
Automated ML pipeline with intelligent target detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                            r2_score, mean_squared_error, mean_absolute_error,
                            silhouette_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class AIAnalysisEngine:
    """Main analysis engine for automated ML pipeline"""
    
    def __init__(self, dataframe, research_question=""):
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
        
    def detect_target_column(self):
        """Intelligently detect target variable using 3 strategies"""
        # Strategy 1: Parse research question
        if self.research_question:
            for col in self.df.columns:
                if col.lower() in self.research_question:
                    self.target_column = col
                    return col
        
        # Strategy 2: Look for common target column names
        target_patterns = ['target', 'label', 'y', 'outcome', 'class', 
                          'prediction', 'result', 'status', 'approved',
                          'churn', 'fraud', 'default', 'price', 'revenue',
                          'sales', 'value', 'score']
        
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in target_patterns):
                self.target_column = col
                return col
        
        # Strategy 3: Use last column as default
        self.target_column = self.df.columns[-1]
        return self.target_column
    
    def identify_task_type(self):
        """Identify whether task is classification, regression, or clustering"""
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
        """Automated preprocessing: imputation, encoding, scaling"""
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
        
        if numeric_features:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
        
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
        
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X, y
    
    def train_classification_models(self, X_train, X_test, y_train, y_test, cv_folds=5, use_smote=True):
        """Train multiple classification models"""
        if use_smote and len(np.unique(y_train)) == 2:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except:
                pass
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                
                results[name] = {
                    'Accuracy': round(accuracy, 4),
                    'F1 Score': round(f1, 4),
                    'CV Mean': round(cv_scores.mean(), 4),
                    'CV Std': round(cv_scores.std(), 4)
                }
                
                self.trained_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test, cv_folds=5):
        """Train multiple regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                
                results[name] = {
                    'R² Score': round(r2, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'CV Mean': round(cv_scores.mean(), 4),
                    'CV Std': round(cv_scores.std(), 4)
                }
                
                self.trained_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results
    
    def extract_feature_importance(self, model_name='Random Forest'):
        """Extract feature importance from tree-based models"""
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
    
    def generate_insights(self, model_results):
        """Generate automated insights from analysis"""
        insights = []
        
        if self.task_type == 'classification':
            best_model = max(model_results.items(), key=lambda x: x[1].get('Accuracy', 0))
            insights.append(f"Best performing model is {best_model[0]} with {best_model[1]['Accuracy']:.1%} accuracy")
        elif self.task_type == 'regression':
            best_model = max(model_results.items(), key=lambda x: x[1].get('R² Score', 0))
            insights.append(f"Best performing model is {best_model[0]} with R² = {best_model[1]['R² Score']:.3f}")
        
        if self.feature_importance is not None:
            top_feature = self.feature_importance.iloc[0]
            insights.append(f"Most important feature: {top_feature['Feature']} ({top_feature['Importance']:.3f})")
        
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 10:
            insights.append(f"Dataset has {missing_pct:.1f}% missing values - imputation was applied")
        
        return insights
    
    def run_full_pipeline(self, test_size=0.2, cv_folds=5, use_smote=True):
        """Execute complete analysis pipeline"""
        self.detect_target_column()
        self.identify_task_type()
        
        X, y = self.preprocess_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        if self.task_type == 'classification':
            model_results = self.train_classification_models(X_train, X_test, y_train, y_test, cv_folds, use_smote)
        else:
            model_results = self.train_regression_models(X_train, X_test, y_train, y_test, cv_folds)
        
        if 'Random Forest' in self.trained_models:
            self.extract_feature_importance('Random Forest')
        
        insights = self.generate_insights(model_results)
        
        if self.task_type == 'classification':
            best_model = max(model_results.items(), key=lambda x: x[1].get('Accuracy', 0))[0]
        else:
            best_model = max(model_results.items(), key=lambda x: x[1].get('R² Score', 0))[0]
        
        return {
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
