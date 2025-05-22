"""
Stats Schema for NES Copilot

This module defines the canonical list of 25 summary statistics and validation logic
for the Normative Executive System (NES) model, matching the NPE SBC implementation.
"""

from typing import List, Dict, Any, Optional
import numpy as np

# Define the canonical list of 25 summary statistics
NES_SUMMARY_STAT_KEYS = [
    # 1. Overall statistics (5)
    'prop_gamble_overall',    # Overall proportion of gamble choices
    'mean_rt_overall',        # Overall mean reaction time
    'rt_q10_overall',         # 10th percentile of RT
    'rt_q50_overall',         # 50th percentile (median) of RT
    'rt_q90_overall',         # 90th percentile of RT
    
    # 2-5. Condition-specific statistics (4 conditions × 5 stats = 20)
    # Format: {stat_prefix}_{condition}
    # Conditions: Gain_NTC, Gain_TC, Loss_NTC, Loss_TC
    # Stats: prop_gamble, mean_rt, rt_q10, rt_q50, rt_q90
    *[f'prop_gamble_{cond}' for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']],
    *[f'mean_rt_{cond}' for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']],
    *[f'rt_q10_{cond}' for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']],
    *[f'rt_q50_{cond}' for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']],
    *[f'rt_q90_{cond}' for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']],
]

# Expected number of statistics (should be 25: 5 overall + (4 conditions × 5 stats))
EXPECTED_NUM_STATS = len(NES_SUMMARY_STAT_KEYS)

# Define metadata for each statistic for documentation and validation
STATS_METADATA = {
    # Overall statistics
    'prop_gamble_overall': {
        'description': 'Overall proportion of gamble choices',
        'range': (0, 1),
        'type': 'probability'
    },
    'mean_rt_overall': {
        'description': 'Overall mean reaction time (s)',
        'range': (0, None),
        'type': 'time'
    },
    'rt_q10_overall': {
        'description': '10th percentile of reaction times (s)',
        'range': (0, None),
        'type': 'time'
    },
    'rt_q50_overall': {
        'description': '50th percentile (median) of reaction times (s)',
        'range': (0, None),
        'type': 'time'
    },
    'rt_q90_overall': {
        'description': '90th percentile of reaction times (s)',
        'range': (0, None),
        'type': 'time'
    },
    
    # Condition-specific statistics (dynamically generated for each condition)
    **{
        f'{stat}_{cond}': {
            'description': f"{stat.replace('_', ' ')} in {cond.replace('_', ' ')}",
            'range': (0, 1) if stat.startswith('prop_gamble') else (0, None),
            'type': 'probability' if stat.startswith('prop_gamble') else 'time'
        }
        for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']
        for stat in ['prop_gamble', 'mean_rt', 'rt_q10', 'rt_q50', 'rt_q90']
    }
}

def validate_summary_stats(stats: Dict[str, float]) -> bool:
    """
    Validate that the summary statistics dictionary contains all expected statistics
    and that their values are within expected ranges.
    
    Args:
        stats: Dictionary of summary statistics.
        
    Returns:
        bool: True if the statistics are valid, False otherwise.
    """
    # Check if all expected statistics are present
    for key in NES_SUMMARY_STAT_KEYS:
        if key not in stats:
            return False
            
    # Check if there are any unexpected statistics
    for key in stats:
        if key not in NES_SUMMARY_STAT_KEYS:
            return False
            
    # Check value ranges
    for key, value in stats.items():
        if not np.isfinite(value):
            return False
            
        # Get metadata for this statistic
        meta = STATS_METADATA.get(key, {})
        if not meta:
            continue
            
        # Check if value is within expected range
        val_range = meta.get('range')
        if val_range and not (val_range[0] <= value <= val_range[1]):
            return False
            
    return True

def get_stat_description(stat_name: str) -> str:
    """Get a human-readable description of a statistic."""
    return STATS_METADATA.get(stat_name, {}).get('description', 'No description available')

def get_stat_type(stat_name: str) -> str:
    """Get the type of a statistic (e.g., 'probability', 'time')."""
    return STATS_METADATA.get(stat_name, {}).get('type', 'unknown')
