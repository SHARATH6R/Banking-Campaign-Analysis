Banking Campaign Analysis
Analyzing bank marketing data to understand why customers subscribe to term deposits.
Setup
bashpip install -r requirements.txt
Put your CSV file in data/raw/ and open the notebook.
What's Inside

config/ - Settings
src/ - Data loading, cleaning, and analysis code
notebooks/ - Main analysis notebook
utils/ - Helper functions

Requirements

Python 3.8+
pandas
matplotlib
seaborn

Usage
Open notebooks/banking_intelligence_analysis.ipynb in Jupyter and run it.
Or import modules:
pythonfrom src.data_processor.dataset_loader import load_banking_data
from src.analysis.statistical_analyzer import StatisticalAnalyzer

data = load_banking_data('data/raw/bank_data.csv')
analyzer = StatisticalAnalyzer(data)
Features

Clean messy data
Calculate stats
Create pivot tables
Make charts

Questions This Answers

Subscription rate
Customer demographics
Call duration impact
Job type patterns

Common Issues
File not found - Check path in config/analysis_config.py
Memory error - Sample the data first
No plots - Use Jupyter or add plt.show()
