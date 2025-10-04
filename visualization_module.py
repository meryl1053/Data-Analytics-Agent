"""
AI Data Analysis Agent - Visualization Module
Comprehensive plotting system for EDA and model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

class VisualizationEngine:
    """Handles all visualization tasks"""
    
    def __init__(self, figsize=(12, 6), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 10)
    
    def plot_missing_values(self, df, max_cols=20):
        """Visualize missing values in dataset"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_data = pd.DataFrame({
            'Column': missing.index,
            'Missing_Percentage': missing_pct.values
        })
        missing_data = missing_data[missing_data['Missing_Percentage'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        ).head(max_cols)
        
        if len(missing_data) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Missing Values Found!', ha='center', va='center', fontsize=20, color='green')
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        bars = ax.barh(missing_data['Column'], missing_data['Missing_Percentage'], color=self.color_palette[0])
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Missing Percentage (%)', fontsize=12)
        ax.set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df, method='pearson', max_features=15):
        """Create correlation heatmap for numeric features"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Numeric Features Found', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        if len(numeric_df.columns) > max_features:
            target_corr = numeric_df.corr()[numeric_df.columns[-1]].abs().sort_values(ascending=False)
            top_features = target_corr.head(max_features).index
            numeric_df = numeric_df[top_features]
        
        corr_matrix = numeric_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, ax=ax)
        ax.set_title(f'Correlation Heatmap ({method.title()})', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def plot_distributions(self, df, max_features=6):
        """Plot distributions of numeric features"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:max_features]
        
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Numeric Features to Plot', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        n_cols = min(3, len(numeric_cols))
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), dpi=self.dpi)
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            data = df[col].dropna()
            axes[idx].hist(data, bins=30, alpha=0.7, color=self.color_palette[idx % len(self.color_palette)], edgecolor='black')
            axes[idx].set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_outliers(self, df, max_features=6):
        """Create box plots to identify outliers"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:max_features]
        
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Numeric Features to Analyze', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        n_cols = min(3, len(numeric_cols))
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), dpi=self.dpi)
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            data = df[col].dropna()
            axes[idx].boxplot(data, vert=True, patch_artist=True)
            axes[idx].set_title(f'Outlier Detection: {col}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(col, fontsize=10)
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_categorical_distributions(self, df, max_features=6):
        """Plot distributions of categorical features"""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:max_features]
        
        if len(cat_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Categorical Features Found', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        n_cols = min(2, len(cat_cols))
        n_rows = int(np.ceil(len(cat_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5), dpi=self.dpi)
        axes = axes.flatten() if len(cat_cols) > 1 else [axes]
        
        for idx, col in enumerate(cat_cols):
            value_counts = df[col].value_counts().head(10)
            axes[idx].bar(range(len(value_counts)), value_counts.values, color=self.color_palette[idx % len(self.color_palette)])
            axes[idx].set_xticks(range(len(value_counts)))
            axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Count', fontsize=10)
        
        for idx in range(len(cat_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=15):
        """Plot feature importance from model"""
        if feature_importance_df is None or len(feature_importance_df) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'Feature Importance Not Available', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        top_features = feature_importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        bars = ax.barh(top_features['Feature'], top_features['Importance'], color=self.color_palette[2])
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, model_results, task_type):
        """Compare performance of different models"""
        if not model_results:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Model Results Available', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        models = list(model_results.keys())
        
        if task_type == 'classification':
            metric1 = [model_results[m].get('Accuracy', 0) for m in models]
            metric2 = [model_results[m].get('F1 Score', 0) for m in models]
            labels = ['Accuracy', 'F1 Score']
        else:
            metric1 = [model_results[m].get('R² Score', 0) for m in models]
            metric2 = [model_results[m].get('MAE', 0) for m in models]
            labels = ['R² Score', 'MAE']
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, metric1, width, label=labels[0], color=self.color_palette[0])
        ax.bar(x + width/2, metric2, width, label=labels[1], color=self.color_palette[1])
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def generate_all_eda_plots(self, df):
        """Generate complete set of EDA plots"""
        plots = {}
        plots['Missing Values'] = self.plot_missing_values(df)
        plots['Correlation Heatmap'] = self.plot_correlation_heatmap(df)
        plots['Numeric Distributions'] = self.plot_distributions(df)
        plots['Outlier Detection'] = self.plot_outliers(df)
        plots['Categorical Distributions'] = self.plot_categorical_distributions(df)
        return plots
