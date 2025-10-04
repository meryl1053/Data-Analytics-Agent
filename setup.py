"""
AI Data Analysis Agent - Automated Setup Script
Handles project structure, dependencies, and sample data generation
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+ is installed"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_project_structure():
    """Create all necessary directories"""
    directories = ['data', 'models', 'reports', 'tests', '.streamlit']
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")

def install_dependencies():
    """Install all required packages"""
    print("\n📦 Installing dependencies...")
    
    packages = [
        'streamlit>=1.28.0',
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.17.0',
        'python-docx>=0.8.11',
        'imbalanced-learn>=0.11.0',
        'openpyxl>=3.1.0',
        'pytest>=7.4.0'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install: {package}")

def generate_sample_data():
    """Create sample loan approval dataset"""
    print("\n📊 Generating sample dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
        'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
        'loan_purpose': np.random.choice(['Business', 'Education', 'Home', 'Personal'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with realistic logic
    approval_prob = (
        (df['credit_score'] / 850) * 0.4 +
        (df['income'] / 150000) * 0.3 +
        (df['employment_years'] / 40) * 0.2 +
        np.random.uniform(0, 0.1, n_samples)
    )
    df['approved'] = (approval_prob > 0.5).astype(int)
    
    # Add some missing values
    for col in ['income', 'credit_score', 'employment_years']:
        missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Save to CSV
    df.to_csv('data/sample_loan_data.csv', index=False)
    print(f"✅ Created: data/sample_loan_data.csv ({n_samples} rows, {len(df.columns)} columns)")
    
    return df

def create_streamlit_config():
    """Create Streamlit configuration file"""
    print("\n⚙️  Creating Streamlit config...")
    
    config = """[server]
port = 8501
maxUploadSize = 200
enableCORS = false
headless = true

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false
"""
    
    with open('.streamlit/config.toml', 'w') as f:
        f.write(config)
    
    print("✅ Created: .streamlit/config.toml")

def create_requirements_file():
    """Generate requirements.txt"""
    print("\n📝 Creating requirements.txt...")
    
    requirements = """# Core Dependencies
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Document Generation
python-docx>=0.8.11

# ML Enhancement
imbalanced-learn>=0.11.0

# Data Handling
openpyxl>=3.1.0

# Testing
pytest>=7.4.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created: requirements.txt")

def create_readme():
    """Create basic README"""
    readme = """# AI Data Analysis Agent

Automated machine learning platform for instant insights.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run integrated_app.py
   ```

3. **Upload data and analyze!**

## Features

- ✨ Automated target detection
- 🧠 Multi-model training
- 📊 Comprehensive visualizations
- 📄 Professional DOCX reports
- 🚀 Production-ready architecture

## Supported Tasks

- Classification (binary/multi-class)
- Regression (continuous prediction)
- Clustering (unsupervised learning)

## Project Structure

```
ai-data-agent/
├── integrated_app.py          # Main application
├── analysis_engine.py         # ML pipeline
├── visualization_module.py    # Plotting system
├── report_generator.py        # Report creation
├── utils_module.py            # Utilities
├── data/                      # Datasets
├── models/                    # Saved models
└── reports/                   # Generated reports
```

## Support

For issues or questions, check the documentation or create an issue.

---
Built with ❤️ using Streamlit, scikit-learn, and pandas
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print("✅ Created: README.md")

def main():
    """Main setup workflow"""
    print("=" * 60)
    print("🤖 AI Data Analysis Agent - Setup")
    print("=" * 60)
    
    try:
        # Step 1: Check Python version
        print("\n📍 Step 1: Checking Python version...")
        check_python_version()
        
        # Step 2: Create directories
        print("\n📍 Step 2: Creating project structure...")
        create_project_structure()
        
        # Step 3: Install dependencies
        print("\n📍 Step 3: Installing dependencies...")
        install_dependencies()
        
        # Step 4: Generate sample data
        print("\n📍 Step 4: Generating sample data...")
        df = generate_sample_data()
        
        # Step 5: Create config files
        print("\n📍 Step 5: Creating configuration files...")
        create_streamlit_config()
        create_requirements_file()
        create_readme()
        
        # Success message
        print("\n" + "=" * 60)
        print("✅ Setup Complete!")
        print("=" * 60)
        print("\n🚀 Next Steps:")
        print("   1. Run: streamlit run integrated_app.py")
        print("   2. Upload: data/sample_loan_data.csv")
        print("   3. Question: 'Predict loan approval'")
        print("   4. Click: 'Start AI Analysis'")
        print("\n📊 Sample Dataset Preview:")
        print(df.head())
        print(f"\n📈 Dataset Shape: {df.shape}")
        print(f"🎯 Target Variable: 'approved' (Classification)")
        print(f"✨ Approval Rate: {df['approved'].mean():.1%}")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
