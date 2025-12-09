"""
Chart Building Module for Banking Data Visualization
Comprehensive visualization components using matplotlib and seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import logging

from config.analysis_config import AnalysisParameters, VisualizationTypes

logger = logging.getLogger(__name__)


class ChartBuilder:
    """
    Builds various types of charts and visualizations
    Implements consistent styling and configuration
    """
    
    def __init__(self, figure_style: str = 'whitegrid'):
        """
        Initialize chart builder with styling preferences
        
        Args:
            figure_style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        """
        sns.set_style(figure_style)
        self.default_figsize = (AnalysisParameters.DefaultFigureWidth, 
                               AnalysisParameters.DefaultFigureHeight)
        self.color_palette = AnalysisParameters.PrimaryColorPalette
        plt.rcParams['font.size'] = AnalysisParameters.StandardFontSize
    
    def construct_distribution_histogram(self, data_series: pd.Series,
                                        bin_count: int = 30,
                                        title_text: str = '',
                                        x_label: str = '',
                                        show_kde: bool = True,
                                        figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Create histogram showing data distribution
        
        Args:
            data_series: Data to plot
            bin_count: Number of histogram bins
            title_text: Chart title
            x_label: X-axis label
            show_kde: Whether to overlay KDE curve
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or self.default_figsize
        figure, axis = plt.subplots(figsize=figsize)
        
        sns.histplot(data=data_series, bins=bin_count, kde=show_kde, 
                    color='skyblue', edgecolor='black', ax=axis)
        
        axis.set_title(title_text or f'Distribution of {data_series.name}',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        axis.set_xlabel(x_label or data_series.name)
        axis.set_ylabel('Frequency Count')
        axis.grid(True, alpha=0.3)
        
        plt.tight_layout()
        logger.info(f"Created histogram for {data_series.name}")
        
        return figure
    
    def build_box_plot_comparison(self, dataframe: pd.DataFrame,
                                 value_column: str,
                                 category_column: str,
                                 title_text: str = '',
                                 figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Generate box plot for comparing distributions across categories
        
        Args:
            dataframe: Source data
            value_column: Numeric column for y-axis
            category_column: Categorical column for x-axis
            title_text: Chart title
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or self.default_figsize
        figure, axis = plt.subplots(figsize=figsize)
        
        sns.boxplot(data=dataframe, x=category_column, y=value_column,
                   palette=self.color_palette, ax=axis)
        
        axis.set_title(title_text or f'{value_column} by {category_column}',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        axis.set_xlabel(category_column.replace('_', ' ').title())
        axis.set_ylabel(value_column.replace('_', ' ').title())
        axis.tick_params(axis='x', rotation=45)
        axis.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        logger.info(f"Created box plot: {value_column} by {category_column}")
        
        return figure
    
    def create_bar_chart_analysis(self, data_series: pd.Series,
                                 top_n: Optional[int] = None,
                                 horizontal_bars: bool = False,
                                 title_text: str = '',
                                 figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Construct bar chart for categorical data
        
        Args:
            data_series: Categorical data (value counts)
            top_n: Show only top N categories
            horizontal_bars: Use horizontal orientation
            title_text: Chart title
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or self.default_figsize
        figure, axis = plt.subplots(figsize=figsize)
        
        plot_data = data_series.nlargest(top_n) if top_n else data_series
        
        if horizontal_bars:
            plot_data.plot(kind='barh', ax=axis, color='coral', edgecolor='black')
            axis.set_xlabel('Count')
            axis.set_ylabel('Category')
        else:
            plot_data.plot(kind='bar', ax=axis, color='coral', edgecolor='black')
            axis.set_ylabel('Count')
            axis.set_xlabel('Category')
            axis.tick_params(axis='x', rotation=45)
        
        axis.set_title(title_text or f'Bar Chart: {data_series.name}',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        axis.grid(True, alpha=0.3)
        
        plt.tight_layout()
        logger.info(f"Created bar chart with {len(plot_data)} categories")
        
        return figure
    
    def generate_scatter_plot_analysis(self, dataframe: pd.DataFrame,
                                      x_column: str,
                                      y_column: str,
                                      hue_column: Optional[str] = None,
                                      title_text: str = '',
                                      figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Create scatter plot for relationship analysis
        
        Args:
            dataframe: Source data
            x_column: Column for x-axis
            y_column: Column for y-axis
            hue_column: Column for color coding
            title_text: Chart title
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or self.default_figsize
        figure, axis = plt.subplots(figsize=figsize)
        
        sns.scatterplot(data=dataframe, x=x_column, y=y_column, hue=hue_column,
                       palette=self.color_palette, alpha=0.6, s=100, ax=axis)
        
        axis.set_title(title_text or f'{y_column} vs {x_column}',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        axis.set_xlabel(x_column.replace('_', ' ').title())
        axis.set_ylabel(y_column.replace('_', ' ').title())
        axis.grid(True, alpha=0.3)
        
        if hue_column:
            axis.legend(title=hue_column.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        logger.info(f"Created scatter plot: {x_column} vs {y_column}")
        
        return figure
    
    def build_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                 title_text: str = 'Correlation Matrix',
                                 annotate_values: bool = True,
                                 figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Generate heatmap for correlation analysis
        
        Args:
            correlation_matrix: Correlation DataFrame
            title_text: Chart title
            annotate_values: Whether to display correlation values
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or (12, 10)
        figure, axis = plt.subplots(figsize=figsize)
        
        sns.heatmap(correlation_matrix, annot=annotate_values, fmt='.2f',
                   cmap=AnalysisParameters.DivergingPalette, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axis)
        
        axis.set_title(title_text, fontsize=AnalysisParameters.TitleFontSize, 
                      fontweight='bold', pad=20)
        
        plt.tight_layout()
        logger.info("Created correlation heatmap")
        
        return figure
    
    def create_pie_chart_visualization(self, data_series: pd.Series,
                                      title_text: str = '',
                                      show_percentages: bool = True,
                                      explode_largest: bool = False,
                                      figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Construct pie chart for proportion display
        
        Args:
            data_series: Categorical counts
            title_text: Chart title
            show_percentages: Whether to display percentages
            explode_largest: Separate largest slice
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or (10, 8)
        figure, axis = plt.subplots(figsize=figsize)
        
        explode_values = None
        if explode_largest:
            explode_values = [0.1 if i == data_series.idxmax() else 0 for i in data_series.index]
        
        axis.pie(data_series.values, labels=data_series.index, autopct='%1.1f%%' if show_percentages else None,
                startangle=90, colors=sns.color_palette(self.color_palette, len(data_series)),
                explode=explode_values, shadow=True)
        
        axis.set_title(title_text or f'Proportion: {data_series.name}',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        
        plt.tight_layout()
        logger.info(f"Created pie chart with {len(data_series)} segments")
        
        return figure
    
    def generate_line_plot_trend(self, dataframe: pd.DataFrame,
                                x_column: str,
                                y_columns: List[str],
                                title_text: str = '',
                                figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Create line plot for trend analysis
        
        Args:
            dataframe: Source data
            x_column: Column for x-axis
            y_columns: List of columns for y-axis lines
            title_text: Chart title
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or self.default_figsize
        figure, axis = plt.subplots(figsize=figsize)
        
        for y_col in y_columns:
            axis.plot(dataframe[x_column], dataframe[y_col], marker='o', label=y_col, linewidth=2)
        
        axis.set_title(title_text or 'Trend Analysis',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        axis.set_xlabel(x_column.replace('_', ' ').title())
        axis.set_ylabel('Values')
        axis.legend()
        axis.grid(True, alpha=0.3)
        
        plt.tight_layout()
        logger.info(f"Created line plot with {len(y_columns)} series")
        
        return figure
    
    def build_count_plot_distribution(self, dataframe: pd.DataFrame,
                                     category_column: str,
                                     hue_column: Optional[str] = None,
                                     title_text: str = '',
                                     figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Generate count plot for categorical distributions
        
        Args:
            dataframe: Source data
            category_column: Column to count
            hue_column: Secondary categorical variable
            title_text: Chart title
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or self.default_figsize
        figure, axis = plt.subplots(figsize=figsize)
        
        sns.countplot(data=dataframe, x=category_column, hue=hue_column,
                     palette=self.color_palette, ax=axis)
        
        axis.set_title(title_text or f'Count Distribution: {category_column}',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        axis.set_xlabel(category_column.replace('_', ' ').title())
        axis.set_ylabel('Count')
        axis.tick_params(axis='x', rotation=45)
        axis.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        logger.info(f"Created count plot for {category_column}")
        
        return figure
    
    def create_violin_plot_display(self, dataframe: pd.DataFrame,
                                   value_column: str,
                                   category_column: str,
                                   title_text: str = '',
                                   figure_dimensions: Optional[Tuple] = None) -> plt.Figure:
        """
        Build violin plot showing distribution shape
        
        Args:
            dataframe: Source data
            value_column: Numeric column
            category_column: Categorical column
            title_text: Chart title
            figure_dimensions: Custom figure size
            
        Returns:
            Matplotlib figure object
        """
        figsize = figure_dimensions or self.default_figsize
        figure, axis = plt.subplots(figsize=figsize)
        
        sns.violinplot(data=dataframe, x=category_column, y=value_column,
                      palette=self.color_palette, ax=axis)
        
        axis.set_title(title_text or f'{value_column} Distribution by {category_column}',
                      fontsize=AnalysisParameters.TitleFontSize, fontweight='bold')
        axis.set_xlabel(category_column.replace('_', ' ').title())
        axis.set_ylabel(value_column.replace('_', ' ').title())
        axis.tick_params(axis='x', rotation=45)
        axis.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        logger.info(f"Created violin plot: {value_column} by {category_column}")
        
        return figure
    
    def save_figure_to_file(self, figure: plt.Figure, 
                           output_path: str,
                           dpi_resolution: int = 300,
                           file_format: str = 'png') -> bool:
        """
        Export figure to file
        
        Args:
            figure: Matplotlib figure to save
            output_path: Destination file path
            dpi_resolution: Image resolution
            file_format: File format ('png', 'jpg', 'pdf', 'svg')
            
        Returns:
            Boolean indicating success
        """
        try:
            figure.savefig(output_path, dpi=dpi_resolution, format=file_format, bbox_inches='tight')
            logger.info(f"Figure saved to {output_path}")
            return True
        except Exception as error:
            logger.error(f"Failed to save figure: {str(error)}")
            return False