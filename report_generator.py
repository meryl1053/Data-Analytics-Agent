"""
AI Data Analysis Agent - Report Generator
Professional DOCX report creation with embedded visualizations
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime
import io
import pandas as pd

class ReportGenerator:
    """Handles professional report generation"""
    
    def __init__(self):
        self.doc = None
    
    def create_document(self):
        """Create new document with custom styles"""
        self.doc = Document()
        style = self.doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(11)
        return self.doc
    
    def add_title_page(self, research_question):
        """Add title page to report"""
        title = self.doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run('AI Data Analysis Report')
        run.font.size = Pt(32)
        run.font.bold = True
        run.font.color.rgb = RGBColor(102, 126, 234)
        
        subtitle = self.doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = subtitle.add_run('Automated Machine Learning Analysis')
        run.font.size = Pt(16)
        
        self.doc.add_paragraph()
        rq = self.doc.add_paragraph()
        rq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = rq.add_run(f'Research Question: "{research_question}"')
        run.font.size = Pt(14)
        run.font.italic = True
        
        self.doc.add_paragraph()
        date_para = self.doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = date_para.add_run(f'Generated on {datetime.now().strftime("%B %d, %Y")}')
        run.font.size = Pt(12)
        
        self.doc.add_page_break()
    
    def add_executive_summary(self, task_type, target_col, best_model, insights):
        """Add executive summary section"""
        heading = self.doc.add_paragraph('Executive Summary')
        heading.style = 'Heading 1'
        
        para = self.doc.add_paragraph()
        para.add_run(f'This analysis identified a {task_type} task with "{target_col}" as the target variable. ')
        para.add_run(f'The best performing model was {best_model}.')
        
        self.doc.add_paragraph()
        findings = self.doc.add_paragraph('Key Findings:')
        findings.style = 'Heading 2'
        
        for insight in insights:
            self.doc.add_paragraph(insight, style='List Bullet')
        
        self.doc.add_paragraph()
    
    def add_model_results(self, model_results, task_type):
        """Add model performance section"""
        heading = self.doc.add_paragraph('Model Performance')
        heading.style = 'Heading 1'
        
        results_df = pd.DataFrame(model_results).T
        
        table = self.doc.add_table(rows=len(results_df)+1, cols=len(results_df.columns)+1)
        table.style = 'Light Grid Accent 1'
        
        # Headers
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Model'
        for i, col in enumerate(results_df.columns):
            header_cells[i+1].text = str(col)
        
        # Data
        for i, (idx, row) in enumerate(results_df.iterrows()):
            cells = table.rows[i+1].cells
            cells[0].text = str(idx)
            for j, value in enumerate(row):
                cells[j+1].text = f'{value:.4f}' if isinstance(value, float) else str(value)
        
        self.doc.add_paragraph()
    
    def add_feature_importance(self, feature_importance_df):
        """Add feature importance section"""
        if feature_importance_df is None or len(feature_importance_df) == 0:
            return
        
        heading = self.doc.add_paragraph('Feature Importance')
        heading.style = 'Heading 1'
        
        top_features = feature_importance_df.head(10)
        
        table = self.doc.add_table(rows=len(top_features)+1, cols=2)
        table.style = 'Light Grid Accent 1'
        
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Feature'
        header_cells[1].text = 'Importance'
        
        for i, row in enumerate(top_features.itertuples(index=False)):
            cells = table.rows[i+1].cells
            cells[0].text = str(row.Feature)
            cells[1].text = f'{row.Importance:.4f}'
        
        self.doc.add_paragraph()
    
    def add_recommendations(self):
        """Add recommendations section"""
        heading = self.doc.add_paragraph('Recommendations')
        heading.style = 'Heading 1'
        
        recommendations = [
            'Deploy the best performing model for production predictions',
            'Monitor model performance regularly and retrain with new data',
            'Consider feature engineering to improve model accuracy',
            'Validate predictions on new, unseen data before deployment'
        ]
        
        for rec in recommendations:
            self.doc.add_paragraph(rec, style='List Bullet')
        
        self.doc.add_paragraph()
    
    def generate_full_report(self, research_question, task_type, target_col, 
                           model_results, feature_importance, insights,
                           eda_results=None, plots=None):
        """Generate complete analysis report"""
        self.create_document()
        
        self.add_title_page(research_question)
        self.add_executive_summary(task_type, target_col, 
                                  max(model_results.keys(), key=lambda x: list(model_results[x].values())[0]),
                                  insights)
        self.add_model_results(model_results, task_type)
        self.add_feature_importance(feature_importance)
        self.add_recommendations()
        
        doc_stream = io.BytesIO()
        self.doc.save(doc_stream)
        doc_stream.seek(0)
        
        return doc_stream.getvalue()
