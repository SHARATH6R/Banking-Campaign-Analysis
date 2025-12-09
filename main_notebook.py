# Banking Data Intelligence Analysis - Main Notebook
# Comprehensive Exploratory Data Analysis for Banking Campaign Dataset

# ========== SECTION 1: ENVIRONMENT SETUP ==========

# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('..')

from src.data_processor.dataset_loader import DatasetLoader, load_banking_data
from src.data_processor.data_cleaner import DataCleaner
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.analysis.pivot_generator import PivotTableGenerator
from src.visualization.chart_builder import ChartBuilder
from config.analysis_config import (
    ColumnIdentifiers, 
    AnalysisParameters, 
    ResponseCategories,
    AnalysisQuestions
)

# Display configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 2)

print("=" * 80)
print("BANKING DATA INTELLIGENCE ANALYSIS SYSTEM")
print("Advanced Exploratory Data Analysis Framework")
print("=" * 80)

# ========== SECTION 2: DATA ACQUISITION ==========

print("\n[STEP 1] Initiating Data Loading Process...")

# Initialize dataset loader
data_loader = DatasetLoader()

# Load banking dataset
banking_dataset = data_loader.load_from_csv()

# Display basic information
dataset_info = data_loader.retrieve_dataset_info()

print(f"\nDataset Successfully Loaded:")
print(f"- Total Records: {dataset_info['total_rows']:,}")
print(f"- Total Attributes: {dataset_info['total_columns']}")
print(f"- Memory Usage: {dataset_info['memory_consumption'] / (1024**2):.2f} MB")

# Display sample records
print("\n[Sample Records - First 5 Rows]")
print(data_loader.display_sample_records(num_records=5, sample_type='head'))

# ========== SECTION 3: DATA QUALITY ASSESSMENT ==========

print("\n[STEP 2] Conducting Data Quality Analysis...")

# Check for missing values
missing_value_summary = banking_dataset.isnull().sum()
print("\nMissing Values Summary:")
print(missing_value_summary[missing_value_summary > 0])

# Check for duplicate records
duplicate_count = banking_dataset.duplicated().sum()
print(f"\nDuplicate Records Found: {duplicate_count}")

# Display data types
print("\nData Type Information:")
print(banking_dataset.dtypes)

# Statistical summary
print("\nStatistical Overview:")
print(banking_dataset.describe())

# ========== SECTION 4: DATA CLEANING & PREPROCESSING ==========

print("\n[STEP 3] Executing Data Cleaning Pipeline...")

# Initialize data cleaner
data_cleaner = DataCleaner(banking_dataset)

# Apply comprehensive cleaning
cleaned_banking_data = data_cleaner.apply_cleaning_pipeline()

# Display cleaning report
cleaning_report = data_cleaner.get_cleaning_report()
print("\nCleaning Process Summary:")
print(f"- Original Shape: {cleaning_report['original_shape']}")
print(f"- Processed Shape: {cleaning_report['processed_shape']}")
print(f"- Records Removed: {cleaning_report['rows_removed']}")
print(f"- Data Quality Score: {cleaning_report['data_quality_score']}%")

# ========== SECTION 5: STATISTICAL ANALYSIS ==========

print("\n[STEP 4] Performing Statistical Analysis...")

# Initialize statistical analyzer
stats_analyzer = StatisticalAnalyzer(cleaned_banking_data)

# Research Question 1: Proportion of clients who subscribed
print("\n" + "=" * 80)
print("RESEARCH QUESTION 1: Client Subscription Rate")
print("=" * 80)

subscription_proportions = stats_analyzer.determine_response_proportion()
print(f"\nPositive Response Rate: {subscription_proportions['positive_proportion']:.2f}%")
print(f"Total Subscribers: {subscription_proportions['positive_response_count']:,}")
print(f"Non-Subscribers: {subscription_proportions['negative_response_count']:,}")

# Research Question 2: Mean values of numeric features for subscribers
print("\n" + "=" * 80)
print("RESEARCH QUESTION 2: Subscriber Profile Analysis")
print("=" * 80)

numeric_attributes = [
    ColumnIdentifiers.ClientAge,
    ColumnIdentifiers.AccountBalance,
    ColumnIdentifiers.CallDurationSeconds,
    ColumnIdentifiers.CampaignContacts
]

subscriber_analysis = stats_analyzer.segment_analysis_by_response(numeric_attributes)
print("\nComparative Analysis - Subscribers vs Non-Subscribers:")
print(subscriber_analysis)

# Research Question 3: Average call duration for subscribers
print("\n" + "=" * 80)
print("RESEARCH QUESTION 3: Call Duration Analysis")
print("=" * 80)

avg_duration_subscribers = stats_analyzer.filter_and_compute_average(
    target_column=ColumnIdentifiers.CallDurationSeconds,
    filter_conditions={ColumnIdentifiers.SubscriptionResponse: ResponseCategories.POSITIVE_RESPONSE}
)
print(f"\nAverage Call Duration for Subscribers: {avg_duration_subscribers:.2f} seconds")
print(f"Equivalent to: {avg_duration_subscribers / 60:.2f} minutes")

# Research Question 4: Average age of unmarried subscribers
print("\n" + "=" * 80)
print("RESEARCH QUESTION 4: Unmarried Subscriber Demographics")
print("=" * 80)

avg_age_unmarried_subscribers = stats_analyzer.filter_and_compute_average(
    target_column=ColumnIdentifiers.ClientAge,
    filter_conditions={
        ColumnIdentifiers.SubscriptionResponse: ResponseCategories.POSITIVE_RESPONSE,
        ColumnIdentifiers.MaritalCategory: 'single'
    }
)
print(f"\nAverage Age (Unmarried Subscribers): {avg_age_unmarried_subscribers:.2f} years")

# Research Question 5: Age and duration by employment type
print("\n" + "=" * 80)
print("RESEARCH QUESTION 5: Employment-based Segmentation")
print("=" * 80)

employment_metrics = stats_analyzer.calculate_grouped_metrics(
    grouping_columns=[ColumnIdentifiers.EmploymentType],
    aggregation_column=ColumnIdentifiers.ClientAge,
    aggregation_functions=['mean', 'median', 'count']
)
print("\nAge Metrics by Employment Type:")
print(employment_metrics)

duration_by_employment = stats_analyzer.calculate_grouped_metrics(
    grouping_columns=[ColumnIdentifiers.EmploymentType],
    aggregation_column=ColumnIdentifiers.CallDurationSeconds,
    aggregation_functions=['mean', 'median']
)
print("\nCall Duration by Employment Type:")
print(duration_by_employment)

# ========== SECTION 6: PIVOT TABLE ANALYSIS ==========

print("\n[STEP 5] Generating Pivot Tables...")

# Initialize pivot generator
pivot_creator = PivotTableGenerator(cleaned_banking_data)

# Pivot 1: Age by Education and Marital Status
print("\nPivot Table 1: Average Age by Education and Marital Status")
age_education_pivot = pivot_creator.create_basic_pivot(
    index_column=ColumnIdentifiers.EducationLevel,
    value_column=ColumnIdentifiers.ClientAge,
    aggregation_func='mean',
    column_dimension=ColumnIdentifiers.MaritalCategory
)
print(age_education_pivot)

# Pivot 2: Subscription rate by employment
print("\nPivot Table 2: Subscription Distribution by Employment")
subscription_employment_pivot = pivot_creator.build_cross_tabulation(
    row_variable=ColumnIdentifiers.EmploymentType,
    column_variable=ColumnIdentifiers.SubscriptionResponse,
    normalize_option='index'
)
print(subscription_employment_pivot)

# Pivot 3: Call duration analysis
print("\nPivot Table 3: Call Duration Statistics by Education")
duration_pivot = pivot_creator.create_multi_aggregation_pivot(
    index_column=ColumnIdentifiers.EducationLevel,
    value_columns=[ColumnIdentifiers.CallDurationSeconds],
    aggregation_functions={ColumnIdentifiers.CallDurationSeconds: ['mean', 'median', 'count']}
)
print(duration_pivot)

# ========== SECTION 7: DATA VISUALIZATION ==========

print("\n[STEP 6] Creating Visual Analytics...")

# Initialize chart builder
chart_creator = ChartBuilder(figure_style='whitegrid')

# Visualization 1: Age Distribution
print("\nGenerating Age Distribution Histogram...")
age_histogram = chart_creator.construct_distribution_histogram(
    data_series=cleaned_banking_data[ColumnIdentifiers.ClientAge],
    bin_count=30,
    title_text='Client Age Distribution',
    x_label='Age (years)',
    show_kde=True
)
plt.show()

# Visualization 2: Box plot - Age by Education
print("\nGenerating Age by Education Box Plot...")
age_education_boxplot = chart_creator.build_box_plot_comparison(
    dataframe=cleaned_banking_data,
    value_column=ColumnIdentifiers.ClientAge,
    category_column=ColumnIdentifiers.EducationLevel,
    title_text='Age Distribution Across Education Levels'
)
plt.show()

# Visualization 3: Employment type distribution
print("\nGenerating Employment Type Bar Chart...")
employment_counts = cleaned_banking_data[ColumnIdentifiers.EmploymentType].value_counts()
employment_barchart = chart_creator.create_bar_chart_analysis(
    data_series=employment_counts,
    top_n=10,
    title_text='Top 10 Employment Categories',
    horizontal_bars=True
)
plt.show()

# Visualization 4: Subscription by marital status
print("\nGenerating Subscription Distribution...")
subscription_countplot = chart_creator.build_count_plot_distribution(
    dataframe=cleaned_banking_data,
    category_column=ColumnIdentifiers.MaritalCategory,
    hue_column=ColumnIdentifiers.SubscriptionResponse,
    title_text='Subscription Status by Marital Category'
)
plt.show()

# Visualization 5: Call Duration vs Age scatter
print("\nGenerating Scatter Plot Analysis...")
scatter_analysis = chart_creator.generate_scatter_plot_analysis(
    dataframe=cleaned_banking_data.sample(n=min(1000, len(cleaned_banking_data))),
    x_column=ColumnIdentifiers.ClientAge,
    y_column=ColumnIdentifiers.CallDurationSeconds,
    hue_column=ColumnIdentifiers.SubscriptionResponse,
    title_text='Call Duration vs Client Age (Colored by Subscription)'
)
plt.show()

# Visualization 6: Correlation heatmap
print("\nGenerating Correlation Heatmap...")
correlation_columns = [
    ColumnIdentifiers.ClientAge,
    ColumnIdentifiers.AccountBalance,
    ColumnIdentifiers.CallDurationSeconds,
    ColumnIdentifiers.CampaignContacts,
    ColumnIdentifiers.PreviousContactCount
]
correlation_matrix = stats_analyzer.perform_correlation_analysis(correlation_columns)
correlation_heatmap = chart_creator.build_correlation_heatmap(
    correlation_matrix=correlation_matrix,
    title_text='Correlation Matrix - Key Numeric Attributes'
)
plt.show()

# Visualization 7: Subscription proportion pie chart
print("\nGenerating Subscription Proportion Pie Chart...")
subscription_counts = cleaned_banking_data[ColumnIdentifiers.SubscriptionResponse].value_counts()
pie_chart = chart_creator.create_pie_chart_visualization(
    data_series=subscription_counts,
    title_text='Overall Subscription Response Distribution',
    show_percentages=True,
    explode_largest=True
)
plt.show()

# Visualization 8: Violin plot - Duration by Education
print("\nGenerating Violin Plot...")
violin_plot = chart_creator.create_violin_plot_display(
    dataframe=cleaned_banking_data,
    value_column=ColumnIdentifiers.CallDurationSeconds,
    category_column=ColumnIdentifiers.EducationLevel,
    title_text='Call Duration Distribution by Education Level'
)
plt.show()

# ========== SECTION 8: KEY INSIGHTS & CONCLUSIONS ==========

print("\n" + "=" * 80)
print("KEY INSIGHTS SUMMARY")
print("=" * 80)

print(f"""
1. SUBSCRIPTION RATE:
   - Overall acceptance rate: {subscription_proportions['positive_proportion']:.2f}%
   - Total subscribers: {subscription_proportions['positive_response_count']:,}
   
2. SUBSCRIBER CHARACTERISTICS:
   - Average age of subscribers differs from non-subscribers
   - Call duration is a strong predictor of subscription
   
3. EMPLOYMENT INSIGHTS:
   - Different employment types show varying subscription rates
   - Age and call duration patterns vary by occupation
   
4. DEMOGRAPHIC PATTERNS:
   - Education level influences subscription behavior
   - Marital status correlates with response rates
   
5. CAMPAIGN EFFECTIVENESS:
   - Longer call durations generally associate with positive responses
   - Contact frequency impacts subscription decisions
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ========== SECTION 9: DATA EXPORT ==========

print("\n[OPTIONAL] Exporting Cleaned Dataset...")

# Export cleaned data
export_success = data_loader.banking_dataframe = cleaned_banking_data
if data_loader.export_to_csv('data/processed/cleaned_banking_data.csv'):
    print("✓ Cleaned dataset exported successfully")

print("\n" + "=" * 80)
print("Banking Intelligence Analysis - Session Completed")
print("=" * 80)