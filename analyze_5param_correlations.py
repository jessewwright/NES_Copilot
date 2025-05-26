#!/usr/bin/env python
# Filename: analyze_6param_correlations.py
# Purpose: Analyze parameters from the 6-parameter NES model fit and their correlations
#          with behavioral measures from the Roberts dataset.

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('correlation_analysis.log')
    ]
)

# File paths
fitted_params_file = "empirical_fit_results/empirical_fitting_results.csv"
original_data_file = "roberts_framing_data/ftp_osf_data.csv"
output_dir = "correlation_analysis"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parameters to analyze (6-parameter model)
PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain', 'beta_val']

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    logging.info(f"Loading fitted parameters from {fitted_params_file}")
    fitted_params_df = pd.read_csv(fitted_params_file)
    
    # Ensure consistent column naming and type for subject IDs
    if 'subject' in fitted_params_df.columns:
        fitted_params_df.rename(columns={'subject': 'subject_id'}, inplace=True)
    fitted_params_df['subject_id'] = fitted_params_df['subject_id'].astype(str)
    
    logging.info(f"Loading original Roberts data from {original_data_file}")
    original_data_df = pd.read_csv(original_data_file)
    
    # Log the column names in the original data for debugging
    logging.info("Original data columns:")
    for col in original_data_df.columns:
        logging.info(f"  - {col}")
    
    # Standardize column names if needed
    if 'subject' in original_data_df.columns and 'subject_id' not in original_data_df.columns:
        original_data_df.rename(columns={'subject': 'subject_id'}, inplace=True)
        logging.info("Renamed 'subject' column to 'subject_id'")
    
    # Convert subject_id to string for consistency
    original_data_df['subject_id'] = original_data_df['subject_id'].astype(str)
    
    # Log the unique subject IDs for debugging
    logging.info(f"Found {len(original_data_df['subject_id'].unique())} unique subjects in the original data")
    logging.info(f"Found {len(fitted_params_df['subject_id'].unique())} unique subjects in the fitted parameters")
    
    # Get behavioral metrics per subject
    behavioral_metrics = calculate_behavioral_metrics(original_data_df)
    
    # Merge behavioral metrics with fitted parameters
    if not behavioral_metrics.empty:
        logging.info(f"Merging behavioral metrics with fitted parameters")
        fitted_params_df = pd.merge(fitted_params_df, behavioral_metrics, on='subject_id', how='left')
    
    return fitted_params_df, original_data_df

def calculate_behavioral_metrics(df):
    """Calculate behavioral metrics for each subject from the raw data."""
    if 'subject_id' not in df.columns or 'frame' not in df.columns or 'cond' not in df.columns:
        logging.warning("Original data missing required columns (subject_id, frame, cond)")
        return pd.DataFrame()
    
    # Determine which columns to use for trial type and choice
    trial_type_col = 'trialType' if 'trialType' in df.columns else None
    choice_col = 'choice' if 'choice' in df.columns else ('outcome' if 'outcome' in df.columns else None)
    
    if trial_type_col is None or choice_col is None:
        logging.warning(f"Missing required columns. Available: {list(df.columns)}")
        return pd.DataFrame()
    
    # Filter to only include target trials if possible
    if trial_type_col and 'target' in df[trial_type_col].unique():
        df = df[df[trial_type_col] == 'target'].copy()
        logging.info(f"Filtered to {len(df)} target trials")
    
    # Examine values in the choice/outcome column
    choice_values = df[choice_col].unique()
    logging.info(f"Choice values: {choice_values}")
    
    # Determine how to categorize gambles based on available data
    if choice_col == 'choice':
        # Assuming 1 = gamble, 0 = sure option (common coding)
        df['did_gamble'] = df[choice_col] == 1
    else:  # outcome column
        risky_values = [val for val in choice_values if 'risk' in str(val).lower()]
        df['did_gamble'] = df[choice_col].isin(risky_values)
    
    # Create condition flags
    df['is_tc'] = df['cond'] == 'tc'
    df['is_gain'] = df['frame'] == 'gain'
    
    # Calculate metrics per subject and condition
    metrics_list = []
    
    for subject_id, subject_data in df.groupby('subject_id'):
        subject_metrics = {'subject_id': subject_id}
        
        # Overall gambling proportion
        subject_metrics['prop_gamble_overall_obs'] = subject_data['did_gamble'].mean()
        
        # Gambling proportion by condition
        for frame in ['gain', 'loss']:
            for time_constraint in ['tc', 'ntc']:
                condition_mask = (subject_data['frame'] == frame) & (subject_data['cond'] == time_constraint)
                condition_data = subject_data[condition_mask]
                
                if len(condition_data) > 0:
                    cond_name = f"{frame.capitalize()}_{time_constraint.upper()}"
                    subject_metrics[f"prop_gamble_{cond_name}_obs"] = condition_data['did_gamble'].mean()
                    
                    # RT metrics if available
                    # Check various RT column naming conventions
                    rt_col = None
                    for possible_rt_col in ['rt', 'RT', 'response_time', 'responseTime']:
                        if possible_rt_col in condition_data.columns:
                            rt_col = possible_rt_col
                            break
                    
                    if rt_col:
                        subject_metrics[f"mean_rt_{cond_name}_obs"] = condition_data[rt_col].mean()
        
        # Calculate framing effects
        if all(f"prop_gamble_{c}_obs" in subject_metrics for c in ['Gain_TC', 'Loss_TC']):
            subject_metrics['framing_effect_tc_obs'] = subject_metrics['prop_gamble_Loss_TC_obs'] - subject_metrics['prop_gamble_Gain_TC_obs']
        
        if all(f"prop_gamble_{c}_obs" in subject_metrics for c in ['Gain_NTC', 'Loss_NTC']):
            subject_metrics['framing_effect_ntc_obs'] = subject_metrics['prop_gamble_Loss_NTC_obs'] - subject_metrics['prop_gamble_Gain_NTC_obs']
        
        # RT contrasts
        if all(f"mean_rt_{c}_obs" in subject_metrics for c in ['Gain_TC', 'Loss_TC']):
            subject_metrics['rt_GvL_TC_obs'] = subject_metrics['mean_rt_Gain_TC_obs'] - subject_metrics['mean_rt_Loss_TC_obs']
        
        if all(f"mean_rt_{c}_obs" in subject_metrics for c in ['Gain_NTC', 'Loss_NTC']):
            subject_metrics['rt_GvL_NTC_obs'] = subject_metrics['mean_rt_Gain_NTC_obs'] - subject_metrics['mean_rt_Loss_NTC_obs']
        
        if all(f"mean_rt_{c}_obs" in subject_metrics for c in ['Gain_NTC', 'Gain_TC']):
            subject_metrics['gain_rt_speedup_obs'] = subject_metrics['mean_rt_Gain_NTC_obs'] - subject_metrics['mean_rt_Gain_TC_obs']
        
        # Average framing effect
        if 'framing_effect_tc_obs' in subject_metrics and 'framing_effect_ntc_obs' in subject_metrics:
            subject_metrics['framing_effect_avg_obs'] = (subject_metrics['framing_effect_tc_obs'] + subject_metrics['framing_effect_ntc_obs']) / 2
            subject_metrics['delta_framing_obs'] = subject_metrics['framing_effect_tc_obs'] - subject_metrics['framing_effect_ntc_obs']
        
        metrics_list.append(subject_metrics)
    
    if not metrics_list:
        logging.warning("No behavioral metrics could be calculated")
        return pd.DataFrame()
    
    return pd.DataFrame(metrics_list)

def calculate_derived_measures(df):
    """Calculate derived behavioral measures if they're not already in the DataFrame."""
    # Delta Framing (difference between framing effects in TC vs NTC)
    if 'framing_effect_tc_obs' in df.columns and 'framing_effect_ntc_obs' in df.columns:
        df['delta_framing_obs'] = df['framing_effect_tc_obs'] - df['framing_effect_ntc_obs']
        logging.info("Calculated delta_framing_obs")
    else:
        logging.warning("Couldn't calculate delta_framing_obs (missing required columns)")
    
    # Gain RT Speedup (difference in RT between Gain NTC and TC)
    if 'mean_rt_Gain_NTC_obs' in df.columns and 'mean_rt_Gain_TC_obs' in df.columns:
        df['gain_rt_speedup_obs'] = df['mean_rt_Gain_NTC_obs'] - df['mean_rt_Gain_TC_obs']
        logging.info("Calculated gain_rt_speedup_obs")
    else:
        logging.warning("Couldn't calculate gain_rt_speedup_obs (missing required columns)")
    
    # RT Gain vs Loss in TC condition
    if 'mean_rt_Gain_TC_obs' in df.columns and 'mean_rt_Loss_TC_obs' in df.columns:
        df['rt_GvL_TC_obs'] = df['mean_rt_Gain_TC_obs'] - df['mean_rt_Loss_TC_obs']
        logging.info("Calculated rt_GvL_TC_obs")
    else:
        logging.warning("Couldn't calculate rt_GvL_TC_obs (missing required columns)")
    
    # RT Gain vs Loss in NTC condition
    if 'mean_rt_Gain_NTC_obs' in df.columns and 'mean_rt_Loss_NTC_obs' in df.columns:
        df['rt_GvL_NTC_obs'] = df['mean_rt_Gain_NTC_obs'] - df['mean_rt_Loss_NTC_obs']
        logging.info("Calculated rt_GvL_NTC_obs")
    else:
        logging.warning("Couldn't calculate rt_GvL_NTC_obs (missing required columns)")
    
    return df

def calculate_and_print_correlation(df, param_col, metric_col):
    """Calculate and print correlation between parameter and behavioral measure."""
    # Drop rows with NaNs in the specific columns being correlated
    cleaned_df = df[[param_col, metric_col]].dropna()
    if len(cleaned_df) < 3:  # Need at least 3 data points for a meaningful correlation
        logging.warning(f"Correlation between {param_col} and {metric_col}: Not enough data points ({len(cleaned_df)})")
        return None, None, None
    
    r, p_value = pearsonr(cleaned_df[param_col], cleaned_df[metric_col])
    logging.info(f"Correlation between {param_col} and {metric_col}: r = {r:.3f}, p = {p_value:.4f} (N={len(cleaned_df)})")
    return r, p_value, len(cleaned_df)

def analyze_v_norm_correlations(df):
    """Analyze correlations between v_norm and behavioral measures."""
    logging.info("\n--- v_norm Correlations ---")
    v_norm_col = 'v_norm_mean' # Correct column name in the dataset
    
    if v_norm_col not in df.columns:
        logging.error(f"Column '{v_norm_col}' not found in DataFrame.")
        return None
    
    # Collect results in a dictionary
    results = {}
    
    # Analyze correlations with various measures
    behavioral_measures = [
        'framing_effect_avg_obs', 'delta_framing_obs',
        'prop_gamble_Gain_NTC_obs', 'prop_gamble_Gain_TC_obs',
        'prop_gamble_Loss_NTC_obs', 'prop_gamble_Loss_TC_obs'
    ]
    
    for measure in behavioral_measures:
        if measure in df.columns:
            r, p, n = calculate_and_print_correlation(df, v_norm_col, measure)
            if r is not None:
                results[measure] = {'r': r, 'p': p, 'n': n}
        else:
            logging.warning(f"Behavioral measure '{measure}' not found in DataFrame.")
    
    # Create plots for significant correlations
    for measure, stats in results.items():
        if stats['p'] < 0.05:  # Significant correlation
            plot_correlation(df, v_norm_col, measure, stats, output_dir)
    
    return results

def analyze_alpha_gain_correlations(df):
    """Analyze correlations between alpha_gain and behavioral measures."""
    logging.info("\n--- alpha_gain Correlations ---")
    alpha_gain_col = 'alpha_gain_mean' # Correct column name in the dataset
    
    if alpha_gain_col not in df.columns:
        logging.error(f"Column '{alpha_gain_col}' not found in DataFrame.")
        return None
    
    # Collect results in a dictionary
    results = {}
    
    # Analyze correlations with various measures
    behavioral_measures = [
        'gain_rt_speedup_obs', 'rt_GvL_TC_obs', 'rt_GvL_NTC_obs',
        'mean_rt_Gain_TC_obs', 'mean_rt_Gain_NTC_obs',
        'mean_rt_Loss_TC_obs', 'mean_rt_Loss_NTC_obs'
    ]
    
    for measure in behavioral_measures:
        if measure in df.columns:
            r, p, n = calculate_and_print_correlation(df, alpha_gain_col, measure)
            if r is not None:
                results[measure] = {'r': r, 'p': p, 'n': n}
        else:
            logging.warning(f"Behavioral measure '{measure}' not found in DataFrame.")
    
    # Create plots for significant correlations
    for measure, stats in results.items():
        if stats['p'] < 0.05:  # Significant correlation
            plot_correlation(df, alpha_gain_col, measure, stats, output_dir)

def plot_parameter_distributions(df, output_dir):
    """Plot distributions of fitted parameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots in a 2x3 grid for 6 parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()  # Flatten the axes array for easier iteration
    
    for i, param in enumerate(PARAM_NAMES):
        ax = axes[i]
        sns.histplot(df[f'{param}_mean'], kde=True, ax=ax, bins=15)
        ax.set_title(f'Distribution of {param}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Frequency')
        
        # Add mean and std to the plot
        mean_val = df[f'{param}_mean'].mean()
        std_val = df[f'{param}_std'].mean()
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='g', linestyle=':', label=f'±1 std')
        ax.axvline(mean_val - std_val, color='g', linestyle=':')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_correlations(df, output_dir):
    """Plot correlation matrix between all parameters."""
    # Extract parameter means
    param_cols = [f'{p}_mean' for p in PARAM_NAMES]
    param_df = df[param_cols].copy()
    
    # Calculate correlation matrix
    corr = param_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, 
                fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Model Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    logging.info("Loading and preparing data...")
    df, _ = load_and_prepare_data()
    
    if df is None or df.empty:
        logging.error("Failed to load or prepare data. Exiting.")
        return
    
    # Calculate derived behavioral measures
    logging.info("Calculating derived behavioral measures...")
    df = calculate_derived_measures(df)
    
    # Save the combined dataframe for reference
    combined_output_path = os.path.join(output_dir, 'combined_analysis_data.csv')
    df.to_csv(combined_output_path, index=False)
    logging.info(f"Saved combined analysis data to: {combined_output_path}")
    
    # Plot parameter distributions
    logging.info("Generating parameter distribution plots...")
    plot_parameter_distributions(df, output_dir)
    
    # Plot parameter correlations
    logging.info("Generating parameter correlation matrix...")
    plot_parameter_correlations(df, output_dir)
    
    # Analyze correlations for each parameter
    logging.info("Analyzing parameter-behavior correlations...")
    analyze_v_norm_correlations(df)
    analyze_alpha_gain_correlations(df)
    
    # Generate a summary of the analysis
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== NES 6-Parameter Model Analysis Summary ===\n\n")
        f.write(f"Number of subjects: {len(df)}\n")
        f.write(f"Parameters analyzed: {', '.join(PARAM_NAMES)}\n\n")
        
        # Add parameter statistics
        f.write("Parameter Statistics (mean ± std):\n")
        for param in PARAM_NAMES:
            mean_val = df[f'{param}_mean'].mean()
            std_val = df[f'{param}_std'].mean()
            f.write(f"- {param}: {mean_val:.3f} ± {std_val:.3f}\n")
    logging.info("\n=== Analysis Complete ===")
    logging.info(f"Results saved to: {os.path.abspath(output_dir)}")
    logging.info(f"- Parameter distributions: {os.path.join(output_dir, 'parameter_distributions.png')}")
    logging.info(f"- Parameter correlations: {os.path.join(output_dir, 'parameter_correlations.png')}")
    logging.info(f"- Combined data: {combined_output_path}")
    logging.info(f"- Analysis summary: {summary_path}")

def plot_correlation(df, param_col, metric_col, stats, output_dir):
    """Create a scatter plot for a correlation between parameter and behavioral measure."""
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot with regression line
    sns.regplot(data=df, x=param_col, y=metric_col, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
    
    # Add correlation information to title
    plt.title(f"Correlation: {param_col} vs {metric_col}\nr = {stats['r']:.3f}, p = {stats['p']:.4f}, N = {stats['n']}")
    
    # Clean up axes labels
    plt.xlabel(param_col.replace('_mean', '').replace('_', ' ').title())
    plt.ylabel(metric_col.replace('_obs', '').replace('_', ' ').title())
    
    # Add grid for readability
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure
    param_name = param_col.replace('_mean', '')
    metric_name = metric_col.replace('_obs', '')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'corr_{param_name}_vs_{metric_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved correlation plot to {plot_path}")

if __name__ == "__main__":
    main()
