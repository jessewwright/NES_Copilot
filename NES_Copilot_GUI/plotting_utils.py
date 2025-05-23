"""
Plotting utilities for the NES Co-Pilot Mission Control GUI.

This module provides utilities for creating and customizing plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import io
import base64

def create_figure(figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a new matplotlib figure.
    
    Args:
        figsize: Figure size (width, height) in inches.
        
    Returns:
        Matplotlib figure.
    """
    return plt.figure(figsize=figsize)

def plot_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.
    
    Args:
        fig: Matplotlib figure.
        
    Returns:
        Base64-encoded string representation of the figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def plot_ks_test_results(ks_results: Dict[str, Any]) -> plt.Figure:
    """
    Create a bar plot of KS test p-values.
    
    Args:
        ks_results: Dictionary containing KS test results.
        
    Returns:
        Matplotlib figure.
    """
    fig = create_figure(figsize=(12, 8))
    
    # Extract data
    parameters = ks_results['parameters']
    p_values = ks_results['p_values']
    is_uniform = ks_results['is_uniform']
    
    # Create bar colors based on uniformity
    colors = ['green' if uniform else 'red' for uniform in is_uniform]
    
    # Create the bar plot
    plt.bar(parameters, p_values, color=colors)
    
    # Add a horizontal line at p=0.05
    plt.axhline(y=0.05, color='black', linestyle='--', label='p=0.05')
    
    # Add labels and title
    plt.xlabel('Parameter')
    plt.ylabel('p-value')
    plt.title('KS Test p-values for SBC')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_rank_histogram(ranks_df: pd.DataFrame, param_column: str, num_bins: int = 20) -> plt.Figure:
    """
    Create a histogram of ranks for a specific parameter.
    
    Args:
        ranks_df: DataFrame containing ranks.
        param_column: Column name for the parameter.
        num_bins: Number of bins for the histogram.
        
    Returns:
        Matplotlib figure.
    """
    fig = create_figure(figsize=(10, 6))
    
    # Extract the ranks for the parameter
    ranks = ranks_df[param_column].values
    
    # Calculate the expected count per bin for a uniform distribution
    expected_count = len(ranks) / num_bins
    
    # Create the histogram
    plt.hist(ranks, bins=num_bins, alpha=0.7, color='blue', edgecolor='black')
    
    # Add a horizontal line for the expected count
    plt.axhline(y=expected_count, color='red', linestyle='--', label='Expected (Uniform)')
    
    # Add labels and title
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title(f'Rank Histogram for {param_column}')
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_parameter_distributions(fit_results: pd.DataFrame, parameter: str) -> plt.Figure:
    """
    Create a histogram of parameter values across subjects.
    
    Args:
        fit_results: DataFrame containing empirical fitting results.
        parameter: Parameter name.
        
    Returns:
        Matplotlib figure.
    """
    fig = create_figure(figsize=(10, 6))
    
    # Find the column containing the parameter mean
    mean_col = next((col for col in fit_results.columns if col.startswith(f"{parameter}_") and col.endswith("_mean")), None)
    
    if mean_col is None:
        # Try to find the parameter column directly
        mean_col = parameter if parameter in fit_results.columns else None
        
    if mean_col is None:
        plt.text(0.5, 0.5, f"Parameter {parameter} not found in results", 
                 horizontalalignment='center', verticalalignment='center')
        return fig
        
    # Create the histogram with KDE
    sns.histplot(fit_results[mean_col], kde=True)
    
    # Add a vertical line for the mean
    plt.axvline(x=fit_results[mean_col].mean(), color='red', linestyle='-', label='Mean')
    
    # Add a vertical line for the median
    plt.axvline(x=fit_results[mean_col].median(), color='green', linestyle='--', label='Median')
    
    # Add labels and title
    plt.xlabel(parameter)
    plt.ylabel('Count')
    plt.title(f'Distribution of {parameter} Across Subjects')
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_parameter_correlation(fit_results: pd.DataFrame, param_x: str, param_y: str) -> plt.Figure:
    """
    Create a scatter plot of two parameters.
    
    Args:
        fit_results: DataFrame containing empirical fitting results.
        param_x: Parameter name for x-axis.
        param_y: Parameter name for y-axis.
        
    Returns:
        Matplotlib figure.
    """
    fig = create_figure(figsize=(10, 6))
    
    # Find the columns containing the parameter means
    x_col = next((col for col in fit_results.columns if col.startswith(f"{param_x}_") and col.endswith("_mean")), None)
    y_col = next((col for col in fit_results.columns if col.startswith(f"{param_y}_") and col.endswith("_mean")), None)
    
    if x_col is None:
        x_col = param_x if param_x in fit_results.columns else None
        
    if y_col is None:
        y_col = param_y if param_y in fit_results.columns else None
        
    if x_col is None or y_col is None:
        plt.text(0.5, 0.5, f"Parameters {param_x} and/or {param_y} not found in results", 
                 horizontalalignment='center', verticalalignment='center')
        return fig
        
    # Create the scatter plot
    plt.scatter(fit_results[x_col], fit_results[y_col])
    
    # Add a regression line
    sns.regplot(x=fit_results[x_col], y=fit_results[y_col], scatter=False, color='red')
    
    # Calculate correlation
    corr = fit_results[x_col].corr(fit_results[y_col])
    
    # Add labels and title
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title(f'Correlation between {param_x} and {param_y} (r={corr:.3f})')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_ppc_coverage(coverage_df: pd.DataFrame) -> plt.Figure:
    """
    Create a bar plot of PPC coverage percentages.
    
    Args:
        coverage_df: DataFrame containing PPC coverage summary.
        
    Returns:
        Matplotlib figure.
    """
    fig = create_figure(figsize=(12, 10))
    
    # Sort by coverage percentage
    if 'coverage_percentage' in coverage_df.columns and 'statistic' in coverage_df.columns:
        df_sorted = coverage_df.sort_values('coverage_percentage', ascending=False)
        
        # Create the bar plot
        plt.barh(df_sorted['statistic'], df_sorted['coverage_percentage'], color='skyblue')
        
        # Add a vertical line at 95%
        plt.axvline(x=95, color='red', linestyle='--', label='Target (95%)')
        
        # Add labels and title
        plt.xlabel('Coverage Percentage')
        plt.ylabel('Summary Statistic')
        plt.title('PPC Coverage Summary')
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Required columns not found in coverage data", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_subject_ppc_coverage(subject_coverage: pd.DataFrame, statistic: str) -> plt.Figure:
    """
    Create a plot showing PPC coverage for a specific subject and statistic.
    
    Args:
        subject_coverage: DataFrame containing subject's PPC coverage data.
        statistic: Statistic name.
        
    Returns:
        Matplotlib figure.
    """
    fig = create_figure(figsize=(10, 6))
    
    # Filter for the specific statistic
    if 'statistic' in subject_coverage.columns:
        stat_data = subject_coverage[subject_coverage['statistic'] == statistic]
        
        if len(stat_data) == 0:
            plt.text(0.5, 0.5, f"Statistic {statistic} not found for this subject", 
                     horizontalalignment='center', verticalalignment='center')
            return fig
            
        # Check if we have the necessary columns
        if all(col in stat_data.columns for col in ['empirical_value', 'simulated_mean', 'lower_bound', 'upper_bound']):
            # Create a histogram of simulated values (we'll fake this since we don't have the raw simulated values)
            x = np.linspace(
                stat_data['lower_bound'].values[0] - 2 * (stat_data['simulated_mean'].values[0] - stat_data['lower_bound'].values[0]),
                stat_data['upper_bound'].values[0] + 2 * (stat_data['upper_bound'].values[0] - stat_data['simulated_mean'].values[0]),
                1000
            )
            
            # Create a normal distribution based on the mean and implied std
            implied_std = (stat_data['upper_bound'].values[0] - stat_data['lower_bound'].values[0]) / 3.92  # 95% CI is approximately ±1.96 std
            y = np.exp(-0.5 * ((x - stat_data['simulated_mean'].values[0]) / implied_std) ** 2) / (implied_std * np.sqrt(2 * np.pi))
            
            # Plot the distribution
            plt.plot(x, y, color='blue')
            plt.fill_between(x, 0, y, alpha=0.3, color='blue')
            
            # Add vertical lines for the empirical value and CI bounds
            plt.axvline(x=stat_data['empirical_value'].values[0], color='red', linestyle='-', label='Empirical Value')
            plt.axvline(x=stat_data['lower_bound'].values[0], color='green', linestyle='--', label='95% CI')
            plt.axvline(x=stat_data['upper_bound'].values[0], color='green', linestyle='--')
            
            # Add labels and title
            plt.xlabel(statistic)
            plt.ylabel('Density')
            covered = stat_data['covered'].values[0]
            plt.title(f'PPC for {statistic} ({"Covered" if covered else "Not Covered"})')
            plt.legend()
        else:
            plt.text(0.5, 0.5, "Required columns not found in coverage data", 
                     horizontalalignment='center', verticalalignment='center')
    else:
        plt.text(0.5, 0.5, "Required columns not found in coverage data", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
