"""
Utility Helper Functions Module
Common utility functions for banking data analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


def execution_timer(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_timestamp = time.perf_counter()
        result = func(*args, **kwargs)
        end_timestamp = time.perf_counter()
        execution_duration = end_timestamp - start_timestamp
        
        logger.info(f"Function '{func.__name__}' executed in {execution_duration:.4f} seconds")
        return result
    
    return wrapper_timer


def safe_division(numerator: float, denominator: float, default_value: float = 0.0) -> float:
    """
    Perform division with zero-division protection
    
    Args:
        numerator: Top value
        denominator: Bottom value
        default_value: Return value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default_value
    except (TypeError, ZeroDivisionError):
        return default_value


def percentage_calculator(part_value: float, total_value: float, decimal_places: int = 2) -> float:
    """
    Calculate percentage with rounding
    
    Args:
        part_value: Partial value
        total_value: Total value
        decimal_places: Number of decimal places
        
    Returns:
        Percentage value
    """
    percentage = safe_division(part_value, total_value, 0.0) * 100
    return round(percentage, decimal_places)


def identify_outlier_indices(data_series: pd.Series, method: str = 'iqr', 
                            threshold: float = 1.5) -> List[int]:
    """
    Detect outlier indices in a series
    
    Args:
        data_series: Input data series
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold value
        
    Returns:
        List of outlier indices
    """
    if method == 'iqr':
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - threshold * IQR
        upper_fence = Q3 + threshold * IQR
        outlier_mask = (data_series < lower_fence) | (data_series > upper_fence)
        
    elif method == 'zscore':
        z_scores = np.abs((data_series - data_series.mean()) / data_series.std())
        outlier_mask = z_scores > threshold
    
    else:
        logger.warning(f"Unknown method: {method}. Using IQR.")
        return identify_outlier_indices(data_series, method='iqr', threshold=threshold)
    
    return data_series[outlier_mask].index.tolist()


def normalize_column_names(dataframe: pd.DataFrame, strategy: str = 'snake_case') -> pd.DataFrame:
    """
    Standardize column names across different formats
    
    Args:
        dataframe: Input DataFrame
        strategy: Naming strategy ('snake_case', 'camelCase', 'PascalCase', 'lowercase')
        
    Returns:
        DataFrame with normalized column names
    """
    normalized_df = dataframe.copy()
    
    if strategy == 'snake_case':
        normalized_df.columns = [
            col.lower().replace(' ', '_').replace('-', '_') 
            for col in normalized_df.columns
        ]
    elif strategy == 'camelCase':
        normalized_df.columns = [
            ''.join([word.capitalize() if i > 0 else word.lower() 
                    for i, word in enumerate(col.split())]) 
            for col in normalized_df.columns
        ]
    elif strategy == 'PascalCase':
        normalized_df.columns = [
            ''.join([word.capitalize() for word in col.split()]) 
            for col in normalized_df.columns
        ]
    elif strategy == 'lowercase':
        normalized_df.columns = [col.lower().replace(' ', '') for col in normalized_df.columns]
    
    return normalized_df


def filter_dataframe_by_conditions(dataframe: pd.DataFrame, 
                                  conditions_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply multiple filter conditions to DataFrame
    
    Args:
        dataframe: Source DataFrame
        conditions_dict: Dictionary of column-value conditions
        
    Returns:
        Filtered DataFrame
    """
    filtered_result = dataframe.copy()
    
    for column_name, filter_value in conditions_dict.items():
        if column_name not in filtered_result.columns:
            logger.warning(f"Column '{column_name}' not found in DataFrame")
            continue
        
        if isinstance(filter_value, list):
            filtered_result = filtered_result[filtered_result[column_name].isin(filter_value)]
        elif isinstance(filter_value, dict):
            # Support for range filters: {'min': 0, 'max': 100}
            if 'min' in filter_value:
                filtered_result = filtered_result[filtered_result[column_name] >= filter_value['min']]
            if 'max' in filter_value:
                filtered_result = filtered_result[filtered_result[column_name] <= filter_value['max']]
        else:
            filtered_result = filtered_result[filtered_result[column_name] == filter_value]
    
    return filtered_result


def compute_category_statistics(dataframe: pd.DataFrame, 
                               category_column: str,
                               value_column: str) -> Dict[str, Dict]:
    """
    Calculate statistics for each category
    
    Args:
        dataframe: Source data
        category_column: Grouping column
        value_column: Numeric column to analyze
        
    Returns:
        Dictionary with category statistics
    """
    category_stats = {}
    
    for category in dataframe[category_column].unique():
        category_data = dataframe[dataframe[category_column] == category][value_column]
        
        category_stats[category] = {
            'count': len(category_data),
            'mean': category_data.mean(),
            'median': category_data.median(),
            'std': category_data.std(),
            'min': category_data.min(),
            'max': category_data.max()
        }
    
    return category_stats


def create_age_groups(age_series: pd.Series, bin_edges: Optional[List] = None, 
                     labels: Optional[List] = None) -> pd.Series:
    """
    Categorize ages into groups
    
    Args:
        age_series: Series containing ages
        bin_edges: Custom bin boundaries
        labels: Custom labels for bins
        
    Returns:
        Categorical series with age groups
    """
    if bin_edges is None:
        bin_edges = [0, 25, 35, 45, 55, 65, 100]
    
    if labels is None:
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    
    age_categories = pd.cut(age_series, bins=bin_edges, labels=labels, include_lowest=True)
    return age_categories


def encode_binary_response(response_series: pd.Series, 
                          positive_value: str = 'yes',
                          negative_value: str = 'no') -> pd.Series:
    """
    Convert binary categorical response to numeric
    
    Args:
        response_series: Series with binary responses
        positive_value: Value representing positive response
        negative_value: Value representing negative response
        
    Returns:
        Numeric series (0 and 1)
    """
    encoding_map = {positive_value: 1, negative_value: 0}
    return response_series.map(encoding_map)


def calculate_response_rate_by_segment(dataframe: pd.DataFrame,
                                       segment_column: str,
                                       response_column: str,
                                       positive_response: str = 'yes') -> pd.DataFrame:
    """
    Calculate response rates for different segments
    
    Args:
        dataframe: Source data
        segment_column: Column to segment by
        response_column: Response indicator column
        positive_response: Value indicating positive response
        
    Returns:
        DataFrame with response rates
    """
    segment_analysis = dataframe.groupby(segment_column)[response_column].agg([
        ('total_count', 'count'),
        ('positive_count', lambda x: (x == positive_response).sum())
    ])
    
    segment_analysis['response_rate'] = percentage_calculator(
        segment_analysis['positive_count'],
        segment_analysis['total_count']
    )
    
    return segment_analysis.sort_values('response_rate', ascending=False)


def identify_top_features_by_correlation(dataframe: pd.DataFrame,
                                        target_column: str,
                                        top_n: int = 10) -> pd.Series:
    """
    Find features most correlated with target
    
    Args:
        dataframe: Source data
        target_column: Target column for correlation
        top_n: Number of top features to return
        
    Returns:
        Series with top correlated features
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])
    
    if target_column not in numeric_data.columns:
        logger.error(f"Target column '{target_column}' is not numeric")
        return pd.Series()
    
    correlations = numeric_data.corr()[target_column].abs().sort_values(ascending=False)
    # Exclude the target itself
    top_features = correlations.drop(target_column).head(top_n)
    
    return top_features


def generate_data_quality_report(dataframe: pd.DataFrame) -> Dict:
    """
    Create comprehensive data quality report
    
    Args:
        dataframe: DataFrame to analyze
        
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'missing_percentage': (dataframe.isnull().sum() / len(dataframe) * 100).to_dict(),
        'duplicate_rows': dataframe.duplicated().sum(),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object']).columns),
        'memory_usage_mb': dataframe.memory_usage(deep=True).sum() / (1024**2)
    }
    
    return report


def smart_sample_dataframe(dataframe: pd.DataFrame, 
                          sample_size: Optional[int] = None,
                          stratify_column: Optional[str] = None) -> pd.DataFrame:
    """
    Intelligently sample DataFrame with optional stratification
    
    Args:
        dataframe: Source DataFrame
        sample_size: Number of rows to sample
        stratify_column: Column to stratify by
        
    Returns:
        Sampled DataFrame
    """
    if sample_size is None:
        sample_size = min(1000, len(dataframe))
    
    sample_size = min(sample_size, len(dataframe))
    
    if stratify_column and stratify_column in dataframe.columns:
        # Stratified sampling
        sampled_df = dataframe.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // dataframe[stratify_column].nunique()))
        )
        return sampled_df
    else:
        # Random sampling
        return dataframe.sample(n=sample_size, random_state=42)


def format_large_number(number: float, precision: int = 2) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        number: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.{precision}f}B"
    elif abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{precision}f}M"
    elif abs(number) >= 1_000:
        return f"{number / 1_000:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def create_summary_table(dataframe: pd.DataFrame, 
                        group_column: str,
                        metric_columns: List[str]) -> pd.DataFrame:
    """
    Generate summary table with multiple metrics
    
    Args:
        dataframe: Source data
        group_column: Grouping column
        metric_columns: Columns to summarize
        
    Returns:
        Summary DataFrame
    """
    summary_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
    
    summary_table = dataframe.groupby(group_column)[metric_columns].agg(summary_funcs).round(2)
    
    return summary_table