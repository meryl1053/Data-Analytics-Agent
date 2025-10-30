"""
AI Data Analysis Agent - Enhanced Visualization Module v2.0
Comprehensive plotting system with 15+ visualization types
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

class EnhancedVisualizationEngine:
    """Advanced visualization with comprehensive plot types"""
    
    def __init__(self, figsize=(12, 6), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 12)
        self.qualitative_cmap = sns.color_palette("Set2", 10)
        
    def plot_missing_values(self, df, max_cols=25):
        """Enhanced missing values visualization"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_data = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_data = missing_data[missing_data['Missing_Percentage'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        ).head(max_cols)
        
        if len(missing_data) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, '✓ No Missing Values Found!', 
                   ha='center', va='center', fontsize=20, color='green', weight='bold')
            ax.axis('off')
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Bar plot
        bars = ax1.barh(missing_data['Column'], missing_data['Missing_Percentage'], 
                       color=self.color_palette[0], edgecolor='black', linewidth=0.5)
        
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        ax1.set_xlabel('Missing Percentage (%)', fontsize=12, weight='bold')
        ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Heatmap view
        missing_matrix = df.isnull().astype(int)
        sns.heatmap(missing_matrix.T, cmap=['lightblue', 'darkred'], 
                   cbar_kws={'label': 'Missing'}, ax=ax2, yticklabels=True)
        ax2.set_title('Missing Data Pattern', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sample Index', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df, method='pearson', max_features=20):
        """Enhanced correlation heatmap with annotations"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Numeric Features Found', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        if len(numeric_df.columns) > max_features:
            corr_with_last = numeric_df.corr()[numeric_df.columns[-1]].abs().sort_values(ascending=False)
            top_features = corr_with_last.head(max_features).index
            numeric_df = numeric_df[top_features]
        
        corr_matrix = numeric_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1, 
                   cbar_kws={'label': 'Correlation Coefficient'}, ax=ax,
                   vmin=-1, vmax=1)
        
        ax.set_title(f'Correlation Heatmap ({method.title()})', 
                    fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def plot_distributions(self, df, max_features=9):
        """Enhanced distribution plots with KDE and statistics"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:max_features]
        
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Numeric Features to Plot', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        n_cols = 3
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4), dpi=self.dpi)
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            data = df[col].dropna()
            
            # Histogram + KDE
            axes[idx].hist(data, bins=30, alpha=0.6, color=self.color_palette[idx % len(self.color_palette)], 
                          edgecolor='black', density=True, label='Histogram')
            
            # KDE overlay
            data.plot(kind='kde', ax=axes[idx], color='darkred', linewidth=2, label='KDE')
            
            # Statistics text
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            axes[idx].axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            axes[idx].set_title(f'{col}\nSkew: {data.skew():.2f} | Kurt: {data.kurtosis():.2f}', 
                              fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Density', fontsize=10)
            axes[idx].legend(fontsize=8, loc='upper right')
            axes[idx].grid(alpha=0.3)
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions with Statistical Overlays', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig
    
    def plot_boxplots(self, df, max_features=9):
        """Enhanced box plots with violin overlay"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:max_features]
        
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Numeric Features', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        n_cols = 3
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4), dpi=self.dpi)
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            data = df[[col]].dropna()
            
            # Violin + Box plot combination
            parts = axes[idx].violinplot([data[col]], positions=[0], showmeans=True, 
                                        showmedians=True, widths=0.7)
            
            for pc in parts['bodies']:
                pc.set_facecolor(self.color_palette[idx % len(self.color_palette)])
                pc.set_alpha(0.6)
            
            axes[idx].boxplot(data[col], positions=[0], widths=0.3, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
            
            # Outlier statistics
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = data[(data[col] < lower) | (data[col] > upper)]
            
            axes[idx].set_title(f'{col}\nOutliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)', 
                              fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(col, fontsize=10)
            axes[idx].set_xticks([])
            axes[idx].grid(axis='y', alpha=0.3)
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Outlier Detection: Box-Violin Plots', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig
    
    def plot_categorical_distributions(self, df, max_features=6):
        """Enhanced categorical distribution plots"""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:max_features]
        
        if len(cat_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Categorical Features Found', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        n_cols = 2
        n_rows = int(np.ceil(len(cat_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5), dpi=self.dpi)
        axes = axes.flatten() if len(cat_cols) > 1 else [axes]
        
        for idx, col in enumerate(cat_cols):
            value_counts = df[col].value_counts().head(15)
            
            bars = axes[idx].barh(range(len(value_counts)), value_counts.values, 
                                 color=self.qualitative_cmap[idx % len(self.qualitative_cmap)],
                                 edgecolor='black', linewidth=0.5)
            
            # Add percentage labels
            total = value_counts.sum()
            for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
                width = bar.get_width()
                pct = (count / total) * 100
                axes[idx].text(width + max(value_counts.values)*0.01, bar.get_y() + bar.get_height()/2, 
                             f'{count} ({pct:.1f}%)', ha='left', va='center', fontsize=9)
            
            axes[idx].set_yticks(range(len(value_counts)))
            axes[idx].set_yticklabels(value_counts.index, fontsize=9)
            axes[idx].set_xlabel('Count', fontsize=10, weight='bold')
            axes[idx].set_title(f'{col} Distribution\nCardinality: {df[col].nunique()}', 
                              fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
        
        for idx in range(len(cat_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_qq_plots(self, df, max_features=6):
        """Q-Q plots for normality testing"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:max_features]
        
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Numeric Features', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        n_cols = 3
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), dpi=self.dpi)
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            data = df[col].dropna()
            
            stats.probplot(data, dist="norm", plot=axes[idx])
            axes[idx].get_lines()[0].set_markerfacecolor(self.color_palette[idx % len(self.color_palette)])
            axes[idx].get_lines()[0].set_markeredgecolor('black')
            axes[idx].get_lines()[0].set_markersize(4)
            
            # Shapiro-Wilk test
            if len(data) < 5000:  # Shapiro-Wilk limitation
                stat, p_value = stats.shapiro(data)
                normality = "Normal" if p_value > 0.05 else "Non-Normal"
                axes[idx].set_title(f'{col}\n{normality} (p={p_value:.4f})', 
                                  fontsize=11, fontweight='bold')
            else:
                axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
            
            axes[idx].grid(alpha=0.3)
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig
    
    def plot_pairplot(self, df, max_features=6):
        """Pairwise scatter plots"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:max_features]
        
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'Need at least 2 numeric features', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        pairplot_fig = sns.pairplot(df[numeric_cols], diag_kind='kde', plot_kws={'alpha': 0.6},
                                   corner=True, height=2.5)
        pairplot_fig.fig.suptitle('Pairwise Feature Relationships', y=1.01, fontsize=16, fontweight='bold')
        
        return pairplot_fig.fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=20):
        """Enhanced feature importance visualization"""
        if feature_importance_df is None or len(feature_importance_df) == 0:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'Feature Importance Not Available', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        top_features = feature_importance_df.head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=self.dpi)
        
        # Horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = ax1.barh(top_features['Feature'], top_features['Importance'], 
                       color=colors, edgecolor='black', linewidth=0.5)
        
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + max(top_features['Importance'])*0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        ax1.set_xlabel('Importance Score', fontsize=12, weight='bold')
        ax1.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Cumulative importance
        cumsum = top_features['Importance'].cumsum() / top_features['Importance'].sum() * 100
        ax2.plot(range(len(cumsum)), cumsum, marker='o', color='darkblue', linewidth=2, markersize=6)
        ax2.fill_between(range(len(cumsum)), cumsum, alpha=0.3, color='lightblue')
        ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Threshold')
        ax2.set_xlabel('Number of Features', fontsize=12, weight='bold')
        ax2.set_ylabel('Cumulative Importance (%)', fontsize=12, weight='bold')
        ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, model_results, task_type):
        """Enhanced model comparison visualization"""
        if not model_results:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Model Results Available', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        models = list(model_results.keys())
        
        if task_type == 'classification':
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics):
                values = [model_results[m].get(metric, 0) for m in models]
                
                bars = axes[idx].bar(range(len(models)), values, 
                                    color=self.color_palette[:len(models)],
                                    edgecolor='black', linewidth=0.5)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
                axes[idx].set_xticks(range(len(models)))
                axes[idx].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
                axes[idx].set_ylabel(metric, fontsize=11, weight='bold')
                axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                axes[idx].set_ylim([0, max(values) * 1.15])
                axes[idx].grid(axis='y', alpha=0.3)
                
                # Highlight best
                best_idx = values.index(max(values))
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
        
        else:  # Regression
            metrics = ['R² Score', 'RMSE', 'MAE', 'MAPE']
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics):
                values = [model_results[m].get(metric, 0) for m in models]
                
                bars = axes[idx].bar(range(len(models)), values,
                                    color=self.color_palette[:len(models)],
                                    edgecolor='black', linewidth=0.5)
                
                for i, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
                axes[idx].set_xticks(range(len(models)))
                axes[idx].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
                axes[idx].set_ylabel(metric, fontsize=11, weight='bold')
                axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)
                
                # Highlight best (note: lower is better for RMSE, MAE, MAPE)
                if metric in ['RMSE', 'MAE', 'MAPE']:
                    best_idx = values.index(min([v for v in values if v > 0]))
                else:
                    best_idx = values.index(max(values))
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Count'}, linewidths=1, linecolor='black')
        
        ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
        ax.set_ylabel('True Label', fontsize=12, weight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """ROC curve for binary classification"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.3, color='orange')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_all_eda_plots(self, df):
        """Generate comprehensive set of EDA plots"""
        plots = {}
        
        plots['Missing Values'] = self.plot_missing_values(df)
        plots['Correlation Heatmap'] = self.plot_correlation_heatmap(df)
        plots['Feature Distributions'] = self.plot_distributions(df)
        plots['Box-Violin Plots'] = self.plot_boxplots(df)
        plots['Categorical Distributions'] = self.plot_categorical_distributions(df)
        plots['Q-Q Plots'] = self.plot_qq_plots(df)
        
        if len(df.select_dtypes(include=['number']).columns) >= 2:
            plots['Pairwise Relationships'] = self.plot_pairplot(df)
        
        return plots
