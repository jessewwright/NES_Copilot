"""
Stats Schema for NES Copilot

This module defines the canonical list of 60 summary statistics and validation logic
for the Normative Executive System (NES) model, matching the NPE SBC implementation.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

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

# Expected number of statistics (should be 60)
EXPECTED_NUM_STATS = len(ROBERTS_SUMMARY_STAT_KEYS)

def _get_stat_metadata(stat_name: str) -> Dict[str, Any]:
    """Generate metadata for a statistic based on its name pattern."""
    if stat_name == 'p_gamble_All':
        return {
            'description': 'Overall proportion of gamble choices',
            'range': (0, 1),
            'type': 'probability'
        }
    elif stat_name == 'mean_rt_All':
        return {
            'description': 'Overall mean reaction time (s)',
            'range': (0, None),
            'type': 'time'
        }
    elif stat_name == 'std_rt_All':
        return {
            'description': 'Overall standard deviation of reaction times (s)',
            'range': (0, None),
            'type': 'time'
        }
    elif stat_name == 'mean_rt_Gamble_All':
        return {
            'description': 'Mean reaction time for gamble choices (all conditions)',
            'range': (0, None),
            'type': 'time'
        }
    elif stat_name == 'mean_rt_Sure_All':
        return {
            'description': 'Mean reaction time for sure choices (all conditions)',
            'range': (0, None),
            'type': 'time'
        }
    elif stat_name.startswith(('p_gamble_', 'framing_index', 'time_pressure_index')):
        return {
            'description': f"{stat_name.replace('_', ' ').title()}",
            'range': (0, 1) if stat_name.startswith('p_gamble_') else (None, None),
            'type': 'probability' if stat_name.startswith('p_gamble_') else 'index'
        }
    elif stat_name.startswith(('mean_rt_', 'std_rt_', 'rt_q')):
        return {
            'description': f"{stat_name.replace('_', ' ').title()}",
            'range': (0, None),
            'type': 'time'
        }
    elif stat_name in ['rt_bimodality_ratio_overall', 'mean_rt_Gamble_vs_Sure_Gain', 'mean_rt_Gamble_vs_Sure_Loss']:
        return {
            'description': f"{stat_name.replace('_', ' ').title()}",
            'range': (None, None),
            'type': 'ratio'
        }
    else:
        return {
            'description': f"{stat_name.replace('_', ' ').title()}",
            'range': (None, None),
            'type': 'other'
        }

# Define metadata for each statistic
STATS_METADATA = {stat: _get_stat_metadata(stat) for stat in ROBERTS_SUMMARY_STAT_KEYS}

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
        if key in STATS_METADATA:
            min_val, max_val = STATS_METADATA[key]['range']
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                return False, f"Value {value:.3f} for {key} is outside expected range ({min_val}, {max_val})"
    
    return True, "All statistics are valid"

def get_stat_description(stat_name: str) -> str:
    """Get a human-readable description of a statistic."""
    return STATS_METADATA.get(stat_name, {}).get('description', f"{stat_name.replace('_', ' ').title()}")

def get_stat_type(stat_name: str) -> str:
    """Get the type of a statistic (e.g., 'probability', 'time')."""
    return STATS_METADATA.get(stat_name, {}).get('type', 'unknown')

def get_stat_range(stat_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Get the expected range of a statistic as (min, max)."""
    return STATS_METADATA.get(stat_name, {}).get('range', (None, None))
