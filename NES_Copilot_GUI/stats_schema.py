"""
Stats Schema for NES Copilot

This module defines the canonical list of summary statistics and validation logic.
"""

from typing import List, Dict, Any

# Define the canonical list of summary statistics
ROBERTS_SUMMARY_STAT_KEYS = [
    # Probability of gambling statistics
    'p_gamble_Gain_TC',
    'p_gamble_Gain_NTC',
    'p_gamble_Loss_TC',
    'p_gamble_Loss_NTC',
    'p_gamble_Gain',
    'p_gamble_Loss',
    'p_gamble_TC',
    'p_gamble_NTC',
    'p_gamble_All',
    
    # Mean reaction time statistics
    'mean_rt_Gain_TC',
    'mean_rt_Gain_NTC',
    'mean_rt_Loss_TC',
    'mean_rt_Loss_NTC',
    'mean_rt_Gain',
    'mean_rt_Loss',
    'mean_rt_TC',
    'mean_rt_NTC',
    'mean_rt_All',
    
    # Standard deviation of reaction time statistics
    'std_rt_Gain_TC',
    'std_rt_Gain_NTC',
    'std_rt_Loss_TC',
    'std_rt_Loss_NTC',
    'std_rt_Gain',
    'std_rt_Loss',
    'std_rt_TC',
    'std_rt_NTC',
    'std_rt_All',
    
    # Mean reaction time for gamble choices
    'mean_rt_Gamble_Gain_TC',
    'mean_rt_Gamble_Gain_NTC',
    'mean_rt_Gamble_Loss_TC',
    'mean_rt_Gamble_Loss_NTC',
    'mean_rt_Gamble_Gain',
    'mean_rt_Gamble_Loss',
    'mean_rt_Gamble_TC',
    'mean_rt_Gamble_NTC',
    'mean_rt_Gamble_All',
    
    # Mean reaction time for sure choices
    'mean_rt_Sure_Gain_TC',
    'mean_rt_Sure_Gain_NTC',
    'mean_rt_Sure_Loss_TC',
    'mean_rt_Sure_Loss_NTC',
    'mean_rt_Sure_Gain',
    'mean_rt_Sure_Loss',
    'mean_rt_Sure_TC',
    'mean_rt_Sure_NTC',
    'mean_rt_Sure_All',
    
    # Composite indices
    'framing_index',
    'time_pressure_index',
    'framing_index_TC',
    'framing_index_NTC',
    'time_pressure_index_Gain',
    'time_pressure_index_Loss',
    
    # RT Quantiles by Frame
    'rt_q90_Gain',
    'rt_q90_Loss',
    'rt_q10_Gain',
    
    # RT Quantiles by Time Constraint
    'rt_q90_TC',
    'rt_q90_NTC',
    'rt_q10_TC',
    
    # Choice-Conditional RT Difference by Frame
    'mean_rt_Gamble_vs_Sure_Gain',
    'mean_rt_Gamble_vs_Sure_Loss',
    
    # RT Bimodality Proxy
    'rt_bimodality_ratio_overall'
]

# Expected number of statistics
EXPECTED_NUM_STATS = len(ROBERTS_SUMMARY_STAT_KEYS)

def validate_summary_stats(stats: Dict[str, float]) -> bool:
    """
    Validate that the summary statistics dictionary contains all expected statistics.
    
    Args:
        stats: Dictionary of summary statistics.
        
    Returns:
        True if the statistics are valid, False otherwise.
    """
    # Check if all expected statistics are present
    for key in ROBERTS_SUMMARY_STAT_KEYS:
        if key not in stats:
            return False
            
    # Check if there are any unexpected statistics
    for key in stats:
        if key not in ROBERTS_SUMMARY_STAT_KEYS:
            return False
            
    return True
