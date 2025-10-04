"""
AI Data Analysis Agent - Comprehensive Test Suite
Unit and integration tests for all modules
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import modules to test
from analysis_engine import AIAnalysisEngine
from visualization_module import VisualizationEngine
from report_generator import ReportGenerator
from utils_module import (DataValidator, ModelPersistence, DataProfiler,
                          ConfigManager, PerformanceTracker)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset"""
    np.random.seed(42)
    n = 200
    
    data = {
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(20000, 120000, n),
        'credit_score': np.random.randint(300, 850, n),
        'loan_amount': np.random.randint(5000, 50000, n),
        'employment_years': np.random.randint(0, 40, n),
        'education': np.random.choice(['HS', 'BA', 'MA', 'PhD'], n),
        'approved': np.random.choice([0, 1], n, p=[0.4, 0.6])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset"""
    np.random.seed(42)
    n = 200
    
    X = np.random.randn(n, 5)
    y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5
    
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['target_value'] = y
    
    return df


@pytest.fixture
def sample_data_with_missing():
    """Create dataset with missing values"""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'col1': np.random.randn(n),
        'col2': np.random.randn(n),
        'col3': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.choice([0, 1], n)
    })
    
    # Add missing values
    missing_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_idx, 'col1'] = np.nan
    df.loc[missing_idx[:10], 'col3'] = np.nan
    
    return df


# ============================================================================
# Data Validation Tests
# ============================================================================

class TestDataValidator:
    """Test DataValidator class"""
    
    def test_valid_dataframe(self, sample_classification_data):
        """Test validation of valid dataframe"""
        validator = DataValidator()
        is_valid, message = validator.validate_dataframe(sample_classification_data)
        
        assert is_valid == True
        assert "passed" in message.lower()
    
    def test_empty_dataframe(self):
        """Test validation of empty dataframe"""
        validator = DataValidator()
        empty_df = pd.DataFrame()
        is_valid, message = validator.validate_dataframe(empty_df)
        
        assert is_valid == False
        assert "empty" in message.lower()
    
    def test_too_few_rows(self):
        """Test validation with insufficient rows"""
        validator = DataValidator(min_rows=100)
        small_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        is_valid, message = validator.validate_dataframe(small_df)
        
        assert is_valid == False
        assert "fewer than" in message.lower()
    
    def test_health_score(self, sample_classification_data):
        """Test health score calculation"""
        validator = DataValidator()
        health = validator.get_dataset_health_score(sample_classification_data)
        
        assert 'health_score' in health
        assert 0 <= health['health_score'] <= 100
        assert health['completeness'] >= 0
        assert health['consistency'] >= 0
    
    def test_quality_report(self, sample_classification_data):
        """Test quality report generation"""
        validator = DataValidator()
        report = validator.get_data_quality_report(sample_classification_data)
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'health_score' in report
        assert report['total_rows'] == len(sample_classification_data)


# ============================================================================
# Analysis Engine Tests
# ============================================================================

class TestAnalysisEngine:
    """Test AIAnalysisEngine class"""
    
    def test_target_detection_from_question(self, sample_classification_data):
        """Test target detection from research question"""
        engine = AIAnalysisEngine(sample_classification_data, 
                                 research_question="Predict loan approved")
        target = engine.detect_target_column()
        
        assert target == 'approved'
    
    def test_target_detection_default(self, sample_regression_data):
        """Test default target detection (last column)"""
        engine = AIAnalysisEngine(sample_regression_data, research_question="")
        target = engine.detect_target_column()
        
        assert target == 'target_value'
    
    def test_classification_task_identification(self, sample_classification_data):
        """Test classification task identification"""
        engine = AIAnalysisEngine(sample_classification_data)
        task_type = engine.identify_task_type()
        
        assert task_type == 'classification'
    
    def test_regression_task_identification(self, sample_regression_data):
        """Test regression task identification"""
        engine = AIAnalysisEngine(sample_regression_data)
        task_type = engine.identify_task_type()
        
        assert task_type == 'regression'
    
    def test_preprocessing(self, sample_data_with_missing):
        """Test data preprocessing"""
        engine = AIAnalysisEngine(sample_data_with_missing)
        engine.detect_target_column()
        engine.identify_task_type()
        
        X, y = engine.preprocess_data()
        
        # Check no missing values after preprocessing
        assert X.isnull().sum().sum() == 0
        assert pd.Series(y).isnull().sum() == 0
    
    def test_full_classification_pipeline(self, sample_classification_data):
        """Test complete classification pipeline"""
        engine = AIAnalysisEngine(sample_classification_data, 
                                 research_question="Predict approved")
        results = engine.run_full_pipeline(test_size=0.2, cv_folds=3, use_smote=False)
        
        assert 'target_column' in results
        assert 'task_type' in results
        assert 'model_results' in results
        assert 'best_model' in results
        assert 'insights' in results
        assert results['task_type'] == 'classification'
        assert len(results['model_results']) > 0
    
    def test_full_regression_pipeline(self, sample_regression_data):
        """Test complete regression pipeline"""
        engine = AIAnalysisEngine(sample_regression_data)
        results = engine.run_full_pipeline(test_size=0.2, cv_folds=3)
        
        assert results['task_type'] == 'regression'
        assert len(results['model_results']) > 0
        assert 'R² Score' in list(results['model_results'].values())[0]
    
    def test_feature_importance_extraction(self, sample_classification_data):
        """Test feature importance extraction"""
        engine = AIAnalysisEngine(sample_classification_data)
        results = engine.run_full_pipeline(test_size=0.2, cv_folds=3)
        
        if results['feature_importance'] is not None:
            assert 'Feature' in results['feature_importance'].columns
            assert 'Importance' in results['feature_importance'].columns
            assert len(results['feature_importance']) > 0


# ============================================================================
# Visualization Tests
# ============================================================================

class TestVisualizationEngine:
    """Test VisualizationEngine class"""
    
    def test_missing_values_plot(self, sample_data_with_missing):
        """Test missing values visualization"""
        viz = VisualizationEngine()
        fig = viz.plot_missing_values(sample_data_with_missing)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_correlation_heatmap(self, sample_classification_data):
        """Test correlation heatmap creation"""
        viz = VisualizationEngine()
        fig = viz.plot_correlation_heatmap(sample_classification_data)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_distributions_plot(self, sample_classification_data):
        """Test distribution plots"""
        viz = VisualizationEngine()
        fig = viz.plot_distributions(sample_classification_data, max_features=3)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_outliers_plot(self, sample_classification_data):
        """Test outlier detection plots"""
        viz = VisualizationEngine()
        fig = viz.plot_outliers(sample_classification_data, max_features=3)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_feature_importance_plot(self):
        """Test feature importance visualization"""
        viz = VisualizationEngine()
        
        # Create sample importance data
        importance_df = pd.DataFrame({
            'Feature': ['f1', 'f2', 'f3', 'f4', 'f5'],
            'Importance': [0.3, 0.25, 0.2, 0.15, 0.1]
        })
        
        fig = viz.plot_feature_importance(importance_df)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_model_comparison_plot(self):
        """Test model comparison visualization"""
        viz = VisualizationEngine()
        
        # Sample model results
        model_results = {
            'Model1': {'Accuracy': 0.85, 'F1 Score': 0.83},
            'Model2': {'Accuracy': 0.82, 'F1 Score': 0.80}
        }
        
        fig = viz.plot_model_comparison(model_results, 'classification')
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_generate_all_eda_plots(self, sample_classification_data):
        """Test generation of all EDA plots"""
        viz = VisualizationEngine()
        plots = viz.generate_all_eda_plots(sample_classification_data)
        
        assert isinstance(plots, dict)
        assert len(plots) > 0
        assert all(fig is not None for fig in plots.values())


# ============================================================================
# Report Generator Tests
# ============================================================================

class TestReportGenerator:
    """Test ReportGenerator class"""
    
    def test_document_creation(self):
        """Test document creation"""
        report_gen = ReportGenerator()
        doc = report_gen.create_document()
        
        assert doc is not None
        assert hasattr(doc, 'add_paragraph')
    
    def test_full_report_generation(self, sample_classification_data):
        """Test complete report generation"""
        report_gen = ReportGenerator()
        
        model_results = {
            'Random Forest': {'Accuracy': 0.85, 'F1 Score': 0.83},
            'Logistic Regression': {'Accuracy': 0.80, 'F1 Score': 0.78}
        }
        
        feature_importance = pd.DataFrame({
            'Feature': ['age', 'income', 'credit_score'],
            'Importance': [0.4, 0.35, 0.25]
        })
        
        insights = [
            'Best model achieved 85% accuracy',
            'Age is the most important feature'
        ]
        
        report_bytes = report_gen.generate_full_report(
            research_question="Predict loan approval",
            task_type='classification',
            target_col='approved',
            model_results=model_results,
            feature_importance=feature_importance,
            insights=insights
        )
        
        assert report_bytes is not None
        assert len(report_bytes) > 0
        assert isinstance(report_bytes, bytes)


# ============================================================================
# Utilities Tests
# ============================================================================

class TestModelPersistence:
    """Test ModelPersistence class"""
    
    def test_serialize_model(self, sample_classification_data):
        """Test model serialization"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a simple model
        X = sample_classification_data.drop('approved', axis=1).select_dtypes(include=['number'])
        y = sample_classification_data['approved']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Serialize
        persistence = ModelPersistence()
        model_bytes = persistence.serialize_model(model)
        
        assert model_bytes is not None
        assert len(model_bytes) > 0
        assert isinstance(model_bytes, bytes)


class TestDataProfiler:
    """Test DataProfiler class"""
    
    def test_profile_dataframe(self, sample_classification_data):
        """Test dataframe profiling"""
        profile = DataProfiler.profile_dataframe(sample_classification_data)
        
        assert 'overview' in profile
        assert 'numeric_features' in profile
        assert 'categorical_features' in profile
        assert 'missing_values' in profile
        assert profile['overview']['rows'] == len(sample_classification_data)
    
    def test_detect_outliers(self, sample_classification_data):
        """Test outlier detection"""
        outliers = DataProfiler.detect_outliers(sample_classification_data)
        
        assert isinstance(outliers, dict)
        assert len(outliers) > 0
        
        for col_outliers in outliers.values():
            assert 'count' in col_outliers
            assert 'percentage' in col_outliers
    
    def test_suggest_transformations(self, sample_regression_data):
        """Test transformation suggestions"""
        suggestions = DataProfiler.suggest_transformations(sample_regression_data)
        
        assert isinstance(suggestions, dict)


class TestConfigManager:
    """Test ConfigManager class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ConfigManager()
        
        assert config.get('random_state') == 42
        assert config.get('test_size') == 0.2
    
    def test_set_config(self):
        """Test setting configuration"""
        config = ConfigManager()
        config.set('test_size', 0.3)
        
        assert config.get('test_size') == 0.3
    
    def test_update_config(self):
        """Test updating multiple configs"""
        config = ConfigManager()
        config.update({'test_size': 0.3, 'cv_folds': 10})
        
        assert config.get('test_size') == 0.3
        assert config.get('cv_folds') == 10


class TestPerformanceTracker:
    """Test PerformanceTracker class"""
    
    def test_time_tracking(self):
        """Test performance time tracking"""
        import time
        
        tracker = PerformanceTracker()
        tracker.start()
        time.sleep(0.1)
        tracker.stop()
        
        summary = tracker.get_summary()
        assert 'total_time_seconds' in summary
        assert summary['total_time_seconds'] >= 0.1
    
    def test_add_metric(self):
        """Test adding custom metrics"""
        tracker = PerformanceTracker()
        tracker.add_metric('test_metric', 100)
        
        summary = tracker.get_summary()
        assert 'test_metric' in summary
        assert summary['test_metric'] == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_end_to_end_classification(self, sample_classification_data):
        """Test complete end-to-end classification workflow"""
        # Step 1: Validate data
        validator = DataValidator()
        is_valid, message = validator.validate_dataframe(sample_classification_data)
        assert is_valid == True
        
        # Step 2: Run analysis
        engine = AIAnalysisEngine(sample_classification_data, 
                                 research_question="Predict approved")
        results = engine.run_full_pipeline(test_size=0.2, cv_folds=3)
        
        assert results['task_type'] == 'classification'
        assert len(results['model_results']) > 0
        
        # Step 3: Generate visualizations
        viz = VisualizationEngine()
        plots = viz.generate_all_eda_plots(sample_classification_data)
        
        assert isinstance(plots, dict)
        assert len(plots) > 0
        
        # Step 4: Generate report
        report_gen = ReportGenerator()
        report_bytes = report_gen.generate_full_report(
            research_question="Predict approved",
            task_type=results['task_type'],
            target_col=results['target_column'],
            model_results=results['model_results'],
            feature_importance=results['feature_importance'],
            insights=results['insights']
        )
        
        assert report_bytes is not None
        assert len(report_bytes) > 0
        
        # Step 5: Save model
        persistence = ModelPersistence()
        model_bytes = persistence.serialize_model(results['best_model_object'])
        
        assert model_bytes is not None
        assert len(model_bytes) > 0
    
    def test_end_to_end_regression(self, sample_regression_data):
        """Test complete end-to-end regression workflow"""
        # Validate and analyze
        validator = DataValidator()
        is_valid, _ = validator.validate_dataframe(sample_regression_data)
        assert is_valid == True
        
        # Run analysis
        engine = AIAnalysisEngine(sample_regression_data)
        results = engine.run_full_pipeline(test_size=0.2, cv_folds=3)
        
        assert results['task_type'] == 'regression'
        assert 'R² Score' in list(results['model_results'].values())[0]
        
        # Generate visualizations
        viz = VisualizationEngine()
        plots = viz.generate_all_eda_plots(sample_regression_data)
        
        assert len(plots) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    """Run all tests with pytest"""
    print("="*70)
    print("AI Data Analysis Agent - Test Suite")
    print("="*70)
    print("\nRunning comprehensive tests...\n")
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes',
        '-ra'
    ])
    
    if exit_code == 0:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n✨ Your AI Data Analysis Agent is production-ready!")
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\n⚠️  Please review the errors above and fix them.")
    
    sys.exit(exit_code)
