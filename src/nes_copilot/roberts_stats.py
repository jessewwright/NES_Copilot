"""
Standalone functions for calculating Roberts et al. summary statistics.

This module provides a simplified interface for calculating the 60 summary statistics
used in the NPE SBC pipeline without requiring the full SummaryStatsModule.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

# Define the canonical list of 60 summary statistics
ROBERTS_SUMMARY_STAT_KEYS = [
    # 1. Basic choice proportions (9)
    "p_gamble_Gain_TC",
    "p_gamble_Gain_NTC",
    "p_gamble_Loss_TC",
    "p_gamble_Loss_NTC",
    "p_gamble_Gain",
    "p_gamble_Loss",
    "p_gamble_TC",
    "p_gamble_NTC",
    "p_gamble_All",
    
    # 2. Mean RTs by condition (17 total)
    "mean_rt_Gain_TC",
    "mean_rt_Gain_NTC",
    "mean_rt_Loss_TC",
    "mean_rt_Loss_NTC",
    "mean_rt_Gain",
    "mean_rt_Loss",
    "mean_rt_TC",
    "mean_rt_NTC",
    "mean_rt_All",
    
    # 3. Standard deviation of RTs (8)
    "std_rt_Gain_TC",
    "std_rt_Gain_NTC",
    "std_rt_Loss_TC",
    "std_rt_Loss_NTC",
    "std_rt_Gain",
    "std_rt_Loss",
    "std_rt_TC",
    "std_rt_NTC",
    "std_rt_All",
    
    # 4. Mean RTs for Gamble choices (9)
    "mean_rt_Gamble_Gain_TC",
    "mean_rt_Gamble_Gain_NTC",
    "mean_rt_Gamble_Loss_TC",
    "mean_rt_Gamble_Loss_NTC",
    "mean_rt_Gamble_Gain",
    "mean_rt_Gamble_Loss",
    "mean_rt_Gamble_TC",
    "mean_rt_Gamble_NTC",
    "mean_rt_Gamble_All",
    
    # 5. Mean RTs for Sure choices (9)
    "mean_rt_Sure_Gain_TC",
    "mean_rt_Sure_Gain_NTC",
    "mean_rt_Sure_Loss_TC",
    "mean_rt_Sure_Loss_NTC",
    "mean_rt_Sure_Gain",
    "mean_rt_Sure_Loss",
    "mean_rt_Sure_TC",
    "mean_rt_Sure_NTC",
    "mean_rt_Sure_All",
    
    # 6. Indices and derived measures (8)
    "framing_index",
    "time_pressure_index",
    "framing_index_TC",
    "framing_index_NTC",
    "time_pressure_index_Gain",
    "time_pressure_index_Loss",
    "rt_q90_Gain",
    "rt_q90_Loss",
    "rt_q10_Gain",
    "rt_q90_TC",
    "rt_q90_NTC",
    "rt_q10_TC",
    "mean_rt_Gamble_vs_Sure_Gain",
    "mean_rt_Gamble_vs_Sure_Loss",
    "rt_bimodality_ratio_overall"
]

def safe_mean(series, min_samples=1):
    """Calculate mean with minimum sample size check."""
    if len(series) >= min_samples:
        return series.mean()
    return np.nan

def safe_std(series, min_samples=2):
    """Calculate standard deviation with minimum sample size check."""
    if len(series) >= min_samples:
        return series.std(ddof=1)  # Sample standard deviation
    return np.nan

def safe_quantile(series, q, min_samples=1):
    """Calculate quantile with minimum sample size check."""
    if len(series) >= min_samples:
        return series.quantile(q)
    return np.nan

def calculate_summary_stats_roberts(
    df_trials: pd.DataFrame,
    stat_keys: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate summary statistics for trial data in the format used by Roberts et al.
    
    Args:
        df_trials: DataFrame containing trial data with columns:
                  - choice: 1 for gamble, 0 for sure
                  - rt: reaction time in seconds
                  - frame: 'gain' or 'loss'
                  - cond: 'tc' or 'ntc'
        stat_keys: List of statistics to calculate. If None, calculates all 60 statistics.
                
    Returns:
        Dictionary mapping statistic names to their values.
    """
    # Make a copy to avoid modifying the input
    df = df_trials.copy()
    
    # Use all stats if none specified
    if stat_keys is None:
        stat_keys = list(ROBERTS_SUMMARY_STAT_KEYS)
    
    # Validate input columns
    required_columns = {'choice', 'rt', 'frame', 'cond'}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
        
    # Convert to numeric and handle potential issues
    df['choice'] = pd.to_numeric(df['choice'], errors='coerce')
    df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
    
    # Drop rows with missing values in required columns
    df = df.dropna(subset=['choice', 'rt', 'frame', 'cond'])
    
    # Ensure frame and cond are lowercase for consistency
    df['frame'] = df['frame'].str.lower()
    df['cond'] = df['cond'].str.lower()
    
    # Create combined condition column
    df['condition'] = df['frame'].str.capitalize() + '_' + df['cond'].str.upper()
    
    # Initialize results with NaN for all requested stats
    stats = {k: np.nan for k in stat_keys}
    
    # Calculate basic statistics for each condition
    conditions = ['Gain_TC', 'Gain_NTC', 'Loss_TC', 'Loss_NTC']
    
    # Calculate statistics for each condition
    for cond in conditions:
        mask = df['condition'] == cond
        df_cond = df[mask]
        
        # Choice proportions
        if f'p_gamble_{cond}' in stat_keys:
            stats[f'p_gamble_{cond}'] = safe_mean(df_cond['choice'])
            
        # Mean RTs
        if f'mean_rt_{cond}' in stat_keys:
            stats[f'mean_rt_{cond}'] = safe_mean(df_cond['rt'])
            
        # Standard deviation of RTs
        if f'std_rt_{cond}' in stat_keys:
            stats[f'std_rt_{cond}'] = safe_std(df_cond['rt'])
            
        # Mean RTs for Gamble choices
        if f'mean_rt_Gamble_{cond}' in stat_keys:
            stats[f'mean_rt_Gamble_{cond}'] = safe_mean(df_cond[df_cond['choice'] == 1]['rt'])
            
        # Mean RTs for Sure choices
        if f'mean_rt_Sure_{cond}' in stat_keys:
            stats[f'mean_rt_Sure_{cond}'] = safe_mean(df_cond[df_cond['choice'] == 0]['rt'])
    
    # Calculate aggregate statistics
    for frame in ['Gain', 'Loss']:
        mask = df['frame'] == frame.lower()
        df_frame = df[mask]
        
        # Overall choice proportions by frame
        if f'p_gamble_{frame}' in stat_keys:
            stats[f'p_gamble_{frame}'] = safe_mean(df_frame['choice'])
            
        # Overall mean RTs by frame
        if f'mean_rt_{frame}' in stat_keys:
            stats[f'mean_rt_{frame}'] = safe_mean(df_frame['rt'])
            
        # Overall std RTs by frame
        if f'std_rt_{frame}' in stat_keys:
            stats[f'std_rt_{frame}'] = safe_std(df_frame['rt'])
            
        # Quantiles for RTs by frame
        if f'rt_q90_{frame}' in stat_keys:
            stats[f'rt_q90_{frame}'] = safe_quantile(df_frame['rt'], 0.9)
            
        if f'rt_q10_{frame}' in stat_keys:
            stats[f'rt_q10_{frame}'] = safe_quantile(df_frame['rt'], 0.1)
            
        # Gamble vs Sure RT difference
        if f'mean_rt_Gamble_vs_Sure_{frame}' in stat_keys:
            rt_gamble = safe_mean(df_frame[df_frame['choice'] == 1]['rt'])
            rt_sure = safe_mean(df_frame[df_frame['choice'] == 0]['rt'])
            stats[f'mean_rt_Gamble_vs_Sure_{frame}'] = rt_gamble - rt_sure
    
    # Calculate statistics by time constraint
    for tc in ['TC', 'NTC']:
        mask = df['cond'] == tc.lower()
        df_tc = df[mask]
        
        # Choice proportions by time constraint
        if f'p_gamble_{tc}' in stat_keys:
            stats[f'p_gamble_{tc}'] = safe_mean(df_tc['choice'])
            
        # Mean RTs by time constraint
        if f'mean_rt_{tc}' in stat_keys:
            stats[f'mean_rt_{tc}'] = safe_mean(df_tc['rt'])
            
        # Std RTs by time constraint
        if f'std_rt_{tc}' in stat_keys:
            stats[f'std_rt_{tc}'] = safe_std(df_tc['rt'])
            
        # Quantiles for RTs by time constraint
        if f'rt_q90_{tc}' in stat_keys:
            stats[f'rt_q90_{tc}'] = safe_quantile(df_tc['rt'], 0.9)
            
        if f'rt_q10_{tc}' in stat_keys:
            stats[f'rt_q10_{tc}'] = safe_quantile(df_tc['rt'], 0.1)
    
    # Overall statistics
    if 'p_gamble_All' in stat_keys:
        stats['p_gamble_All'] = safe_mean(df['choice'])
        
    if 'mean_rt_All' in stat_keys:
        stats['mean_rt_All'] = safe_mean(df['rt'])
        
    if 'std_rt_All' in stat_keys:
        stats['std_rt_All'] = safe_std(df['rt'])
        
    if 'mean_rt_Gamble_All' in stat_keys:
        stats['mean_rt_Gamble_All'] = safe_mean(df[df['choice'] == 1]['rt'])
        
    if 'mean_rt_Sure_All' in stat_keys:
        stats['mean_rt_Sure_All'] = safe_mean(df[df['choice'] == 0]['rt'])
    
    # Calculate framing index (Loss - Gain)
    if 'framing_index' in stat_keys:
        p_gain = stats.get('p_gamble_Gain', np.nan)
        p_loss = stats.get('p_gamble_Loss', np.nan)
        stats['framing_index'] = p_loss - p_gain
    
    # Calculate time pressure index (NTC - TC)
    if 'time_pressure_index' in stat_keys:
        p_tc = stats.get('p_gamble_TC', np.nan)
        p_ntc = stats.get('p_gamble_NTC', np.nan)
        stats['time_pressure_index'] = p_ntc - p_tc
    
    # Calculate framing index by time constraint
    if 'framing_index_TC' in stat_keys:
        p_gain_tc = stats.get('p_gamble_Gain_TC', np.nan)
        p_loss_tc = stats.get('p_gamble_Loss_TC', np.nan)
        stats['framing_index_TC'] = p_loss_tc - p_gain_tc
        
    if 'framing_index_NTC' in stat_keys:
        p_gain_ntc = stats.get('p_gamble_Gain_NTC', np.nan)
        p_loss_ntc = stats.get('p_gamble_Loss_NTC', np.nan)
        stats['framing_index_NTC'] = p_loss_ntc - p_gain_ntc
    
    # Calculate time pressure index by frame
    if 'time_pressure_index_Gain' in stat_keys:
        p_gain_tc = stats.get('p_gamble_Gain_TC', np.nan)
        p_gain_ntc = stats.get('p_gamble_Gain_NTC', np.nan)
        stats['time_pressure_index_Gain'] = p_gain_ntc - p_gain_tc
        
    if 'time_pressure_index_Loss' in stat_keys:
        p_loss_tc = stats.get('p_gamble_Loss_TC', np.nan)
        p_loss_ntc = stats.get('p_gamble_Loss_NTC', np.nan)
        stats['time_pressure_index_Loss'] = p_loss_ntc - p_loss_tc
    
    # Calculate bimodality ratio (q90/q10) for RTs
    if 'rt_bimodality_ratio_overall' in stat_keys:
        q90 = safe_quantile(df['rt'], 0.9)
        q10 = safe_quantile(df['rt'], 0.1)
        if not np.isnan(q90) and not np.isnan(q10) and q10 > 0:
            stats['rt_bimodality_ratio_overall'] = q90 / q10
    
    return stats

def validate_summary_stats(stats: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate that the summary statistics dictionary contains all expected statistics
    and that their values are within expected ranges.
    
    Args:
        stats: Dictionary of summary statistics.
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean indicating if the
              statistics are valid, and message provides details about any validation issues.
    """
    # Check if all expected statistics are present
    missing_stats = [key for key in ROBERTS_SUMMARY_STAT_KEYS if key not in stats]
    if missing_stats:
        return False, f"Missing required statistics: {', '.join(missing_stats[:5])}{'...' if len(missing_stats) > 5 else ''} (total missing: {len(missing_stats)})"
    
    # Check if there are any unexpected statistics
    extra_stats = [key for key in stats if key not in ROBERTS_SUMMARY_STAT_KEYS]
    if extra_stats:
        return False, f"Unexpected statistics found: {', '.join(extra_stats[:5])}{'...' if len(extra_stats) > 5 else ''} (total extra: {len(extra_stats)})"
    
    # Check value ranges and validity
    for key, value in stats.items():
        if not np.isfinite(value):
            return False, f"Non-finite value for {key}: {value}"
            
        # Check value ranges based on metadata
        if key.startswith(('p_gamble_', 'framing_index', 'time_pressure_index')):
            if not (0 <= value <= 1):
                return False, f"Value {value:.3f} for {key} is outside expected range (0, 1)"
        elif key.startswith(('mean_rt_', 'std_rt_', 'rt_q')) and value < 0:
            return False, f"Value {value:.3f} for {key} is negative"
    
    return True, "All statistics are valid"
