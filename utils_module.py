"""
AI Data Analysis Agent - Utilities Module
Helper functions for validation, persistence, and configuration
"""

import pandas as pd
import numpy as np
import pickle
import io
from datetime import datetime

class DataValidator:
    """Validates dataset quality and health"""
    
    def __init__(self, max_size_mb=200, min_rows=10, min_cols=2):
        self.max_size_mb = max_size_mb
        self.min_rows = min_rows
        self.min_cols = min_cols
    
    def validate_dataframe(self, df):
        """Validate dataframe for analysis"""
        if df is None or len(df) == 0:
            return False, "DataFrame is empty"
        
        if len(df) < self.min_rows:
            return False, f"Dataset has fewer than {self.min_rows} rows"
        
        if len(df.columns) < self.min_cols:
            return False, f"Dataset has fewer than {self.min_cols} columns"
        
        null_cols = df.columns[df.isnull().all()].tolist()
        if len(null_cols) == len(df.columns):
            return False, "All columns are completely null"
        
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_mb > self.max_size_mb:
            return False, f"Dataset size ({memory_mb:.1f} MB) exceeds limit ({self.max_size_mb} MB)"
        
        if len(df.columns) != len(set(df.columns)):
            return False, "Dataset contains duplicate column names"
        
        return True, "Dataset validation passed"
    
    def get_dataset_health_score(self, df):
        """Calculate overall dataset health score (0-100)"""
        health_metrics = {}
        
        # Completeness (40 points)
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        completeness_score = max(0, 40 - (missing_pct * 2))
        health_metrics['completeness'] = completeness_score
        
        # Consistency (30 points)
        consistency_issues = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_types = df[col].dropna().apply(type).nunique()
                if unique_types > 1:
                    consistency_issues += 1
        
        consistency_score = max(0, 30 - (consistency_issues * 5))
        health_metrics['consistency'] = consistency_score
        
        # Uniqueness (15 points)
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        uniqueness_score = max(0, 15 - (duplicate_pct * 1.5))
        health_metrics['uniqueness'] = uniqueness_score
        
        # Validity (15 points)
        validity_issues = 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                validity_issues += 1
        
        validity_score = max(0, 15 - (validity_issues * 3))
        health_metrics['validity'] = validity_score
        
        health_metrics['health_score'] = sum(health_metrics.values())
        
        return health_metrics

class ModelPersistence:
    """Handles model saving and loading"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
    
    def serialize_model(self, model):
        """Serialize model to bytes for download"""
        model_stream = io.BytesIO()
        pickle.dump(model, model_stream)
        model_stream.seek(0)
        return model_stream.getvalue()
    
    def deserialize_model(self, model_bytes):
        """Deserialize model from bytes"""
        model_stream = io.BytesIO(model_bytes)
        return pickle.load(model_stream)

class DataProfiler:
    """Provides detailed data profiling"""
    
    @staticmethod
    def profile_dataframe(df):
        """Generate comprehensive data profile"""
        profile = {
            'overview': {},
            'numeric_features': {},
            'categorical_features': {},
            'missing_values': {}
        }
        
        profile['overview'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicates': df.duplicated().sum()
        }
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            profile['numeric_features'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isnull().sum()
            }
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            profile['categorical_features'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                'missing': df[col].isnull().sum()
            }
        
        missing = df.isnull().sum()
        profile['missing_values'] = {
            'total': missing.sum(),
            'percentage': (missing.sum() / (len(df) * len(df.columns))) * 100,
            'by_column': missing[missing > 0].to_dict()
        }
        
        return profile

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self):
        self.config = {
            'random_state': 42,
            'max_features_plot': 15,
            'plot_dpi': 100,
            'test_size': 0.2,
            'cv_folds': 5,
            'use_smote': True
        }
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
    
    def update(self, config_dict):
        self.config.update(config_dict)
