"""
Configuration module for Banking Data Intelligence Analysis
Centralized configuration management for analysis parameters
"""

from enum import Enum
from pathlib import Path


class DataPaths:
    """File path configuration for data assets"""
    ProjectRoot = Path(__file__).parent.parent
    DataDirectory = ProjectRoot / "data"
    RawDataFolder = DataDirectory / "raw"
    ProcessedDataFolder = DataDirectory / "processed"
    
    BankingDatasetFile = RawDataFolder / "bank_marketing_dataset.csv"
    CleanedOutputFile = ProcessedDataFolder / "cleaned_banking_data.csv"


class ColumnIdentifiers:
    """Column name identifiers for banking dataset"""
    ClientAge = "age"
    EmploymentType = "job"
    MaritalCategory = "marital"
    EducationLevel = "education"
    DefaultStatus = "default"
    AccountBalance = "balance"
    HousingLoanStatus = "housing"
    PersonalLoanStatus = "loan"
    ContactChannel = "contact"
    ContactDay = "day"
    ContactMonth = "month"
    CallDurationSeconds = "duration"
    CampaignContacts = "campaign"
    DaysSinceLastContact = "pdays"
    PreviousContactCount = "previous"
    PreviousCampaignOutcome = "poutcome"
    SubscriptionResponse = "y"


class AnalysisParameters:
    """Parameters for statistical analysis and visualization"""
    
    # Visual configuration
    DefaultFigureWidth = 15
    DefaultFigureHeight = 10
    StandardFontSize = 12
    TitleFontSize = 16
    
    # Statistical thresholds
    CorrelationThreshold = 0.7
    SignificanceLevel = 0.05
    OutlierZScore = 3.0
    
    # Data quality parameters
    MissingValueThreshold = 0.3  # 30% threshold
    MinimumSampleSize = 30
    
    # Visualization color schemes
    PrimaryColorPalette = "Set2"
    SecondaryColorPalette = "viridis"
    DivergingPalette = "RdYlGn"


class AggregationMethods(Enum):
    """Enumeration of aggregation methods for pivot tables"""
    MEAN_VALUE = "mean"
    MEDIAN_VALUE = "median"
    TOTAL_SUM = "sum"
    COUNT_RECORDS = "count"
    STANDARD_DEV = "std"
    MINIMUM_VAL = "min"
    MAXIMUM_VAL = "max"


class VisualizationTypes(Enum):
    """Types of visualizations supported"""
    HISTOGRAM_PLOT = "hist"
    BOX_PLOT = "box"
    SCATTER_PLOT = "scatter"
    BAR_CHART = "bar"
    LINE_GRAPH = "line"
    HEATMAP_PLOT = "heatmap"
    PIE_CHART = "pie"


class ResponseCategories:
    """Client response classification"""
    POSITIVE_RESPONSE = "yes"
    NEGATIVE_RESPONSE = "no"
    
    @classmethod
    def get_binary_mapping(cls):
        """Returns mapping for binary conversion"""
        return {
            cls.POSITIVE_RESPONSE: 1,
            cls.NEGATIVE_RESPONSE: 0
        }


class AnalysisQuestions:
    """Research questions for the banking analysis"""
    
    QUESTIONS = [
        "What is the proportion of clients who accepted the term deposit offer?",
        "What are the central tendency measures for numeric attributes among accepting clients?",
        "What is the mean duration of calls for clients who subscribed?",
        "What is the average age of unmarried clients who accepted?",
        "How do age and call duration vary across employment categories?"
    ]
    
    @classmethod
    def display_questions(cls):
        """Print all research questions"""
        print("=" * 80)
        print("BANKING DATA ANALYSIS - KEY RESEARCH QUESTIONS")
        print("=" * 80)
        for index, question in enumerate(cls.QUESTIONS, start=1):
            print(f"\n{index}. {question}")
        print("\n" + "=" * 80)


# Export configuration for easy import
__all__ = [
    'DataPaths',
    'ColumnIdentifiers',
    'AnalysisParameters',
    'AggregationMethods',
    'VisualizationTypes',
    'ResponseCategories',
    'AnalysisQuestions'
]