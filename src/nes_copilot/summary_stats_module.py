"""
Summary Statistics Module for NES Copilot

This module handles calculation of summary statistics for trial data,
implementing the 25 statistics used in the NPE SBC pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

from nes_copilot.module_base import ModuleBase
from nes_copilot.stats_schema import NES_SUMMARY_STAT_KEYS, STATS_METADATA


class SummaryStatsModule(ModuleBase):
    """
    Summary statistics module for the NES Copilot system.
    
    Implements the 25 summary statistics used in the NPE SBC pipeline,
    including overall statistics and condition-specific statistics for
    each combination of frame (Gain/Loss) and time constraint (TC/NTC).
    """
    
    # Define the four experimental conditions
    CONDITIONS = {
        'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'},
        'Gain_TC':  {'frame': 'gain', 'cond': 'tc'},
        'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'},
        'Loss_TC':  {'frame': 'loss', 'cond': 'tc'},
    }
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the summary statistics module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        super().__init__(config_manager, data_manager, logging_manager)
        
        # Get module configuration
        self.config = self.config_manager.get_module_config('summary_stats')
        
        # Use the 25 NES summary statistics as defaults
        self.default_stat_keys = NES_SUMMARY_STAT_KEYS
        
    def run(self, trials_df: pd.DataFrame, stat_keys: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate summary statistics for the given trials.
        
        Args:
            trials_df: DataFrame of trials (simulated or empirical).
            stat_keys: Optional list of specific statistics to calculate.
                      If None, uses the default set of 25 statistics.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the calculated summary statistics and metadata.
        """
        self.logger.info("Starting summary statistics calculation")
        
        # Validate inputs
        self.validate_inputs(trials_df=trials_df)
        
        # Use default statistics if none specified
        if stat_keys is None:
            stat_keys = self.default_stat_keys
            
        # Add any custom statistics from config
        custom_stats = self.config.get('custom_stats', [])
        for stat in custom_stats:
            if stat not in stat_keys:
                stat_keys.append(stat)
        
        # Calculate summary statistics
        summary_stats = self.calculate_summary_stats(trials_df, stat_keys)
        self.logger.info(f"Calculated {len(summary_stats)} summary statistics")
        
        # Save outputs
        output_paths = self.save_outputs({'summary_stats': summary_stats})
        
        # Return results
        results = {
            'summary_stats': summary_stats,
            'stat_keys': stat_keys,
            'output_paths': output_paths
        }
        
        self.logger.info("Summary statistics calculation completed successfully")
        return results
        
    def validate_inputs(self, trials_df: pd.DataFrame, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            trials_df: DataFrame of trials.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, raises ValueError otherwise.
            
        Raises:
            ValueError: If the input data is invalid or missing required columns.
        """
        # Check if trials DataFrame is provided
        if trials_df is None or len(trials_df) == 0:
            raise ValueError("Trials DataFrame not provided or empty")
            
        # Check if trials DataFrame has required columns
        required_columns = ['choice', 'rt', 'frame', 'cond']
        for col in required_columns:
            if col not in trials_df.columns:
                raise ValueError(f"Trials DataFrame missing required column: {col}")
                
        # Check that frame and cond have expected values
        valid_frames = {'gain', 'loss'}
        valid_conds = {'tc', 'ntc'}
        
        if not set(trials_df['frame'].unique()).issubset(valid_frames):
            raise ValueError(f"Invalid frame values. Expected {valid_frames}, got {set(trials_df['frame'].unique())}")
            
        if not set(trials_df['cond'].unique()).issubset(valid_conds):
            raise ValueError(f"Invalid cond values. Expected {valid_conds}, got {set(trials_df['cond'].unique())}")
                
        return True
        
    def _get_condition_mask(self, df: pd.DataFrame, condition: str) -> pd.Series:
        """
        Get a boolean mask for trials matching the specified condition.
        
        Args:
            df: DataFrame of trials.
            condition: Condition name (e.g., 'Gain_TC').
            
        Returns:
            Boolean mask for the specified condition.
        """
        if condition not in self.CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
            
        cond = self.CONDITIONS[condition]
        return (df['frame'] == cond['frame']) & (df['cond'] == cond['cond'])
    
    def _calculate_condition_stats(self, df: pd.DataFrame, condition: str) -> Dict[str, float]:
        """
        Calculate statistics for a specific condition.
        
        Args:
            df: DataFrame of trials.
            condition: Condition name (e.g., 'Gain_TC').
            
        Returns:
            Dictionary of statistics for the condition.
        """
        mask = self._get_condition_mask(df, condition)
        cond_df = df[mask]
        
        if len(cond_df) == 0:
            return {
                f'prop_gamble_{condition}': np.nan,
                f'mean_rt_{condition}': np.nan,
                f'rt_q10_{condition}': np.nan,
                f'rt_q50_{condition}': np.nan,
                f'rt_q90_{condition}': np.nan
            }
        
        # Calculate statistics
        stats = {
            f'prop_gamble_{condition}': cond_df['choice'].mean(),
            f'mean_rt_{condition}': cond_df['rt'].mean()
        }
        
        # Calculate RT percentiles if we have enough data
        if len(cond_df) >= 2:
            try:
                q10, q50, q90 = cond_df['rt'].quantile([0.1, 0.5, 0.9])
                stats.update({
                    f'rt_q10_{condition}': q10,
                    f'rt_q50_{condition}': q50,
                    f'rt_q90_{condition}': q90
                })
            except Exception as e:
                self.logger.warning(f"Error calculating RT percentiles for {condition}: {e}")
                stats.update({
                    f'rt_q10_{condition}': np.nan,
                    f'rt_q50_{condition}': np.nan,
                    f'rt_q90_{condition}': np.nan
                })
        else:
            stats.update({
                f'rt_q10_{condition}': np.nan,
                f'rt_q50_{condition}': np.nan,
                f'rt_q90_{condition}': np.nan
            })
            
        return stats
    
    def calculate_summary_stats(self, trials_df: pd.DataFrame, stat_keys: List[str]) -> Dict[str, float]:
        """
        Calculate summary statistics for the given trials.
        
        Args:
            trials_df: DataFrame of trials.
            stat_keys: List of statistics to calculate.
            
        Returns:
            Dict mapping statistic names to their values.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df = trials_df.copy()
        
        # Ensure required columns are in the correct format
        df['frame'] = df['frame'].str.lower()
        df['cond'] = df['cond'].str.lower()
        
        # Initialize results dictionary with NaN values
        stats = {key: np.nan for key in stat_keys}
        
        try:
            # Calculate overall statistics if requested
            if 'prop_gamble_overall' in stat_keys:
                stats['prop_gamble_overall'] = df['choice'].mean()
                
            if any(stat.startswith('mean_rt_') or stat.startswith('rt_q') for stat in stat_keys):
                if not df['rt'].empty:
                    if 'mean_rt_overall' in stat_keys:
                        stats['mean_rt_overall'] = df['rt'].mean()
                    
                    # Calculate RT percentiles if requested
                    if any(stat.startswith('rt_q') for stat in stat_keys):
                        try:
                            q10, q50, q90 = df['rt'].quantile([0.1, 0.5, 0.9])
                            if 'rt_q10_overall' in stat_keys:
                                stats['rt_q10_overall'] = q10
                            if 'rt_q50_overall' in stat_keys:
                                stats['rt_q50_overall'] = q50
                            if 'rt_q90_overall' in stat_keys:
                                stats['rt_q90_overall'] = q90
                        except Exception as e:
                            self.logger.warning(f"Error calculating overall RT percentiles: {e}")
            
            # Calculate condition-specific statistics
            for condition in self.CONDITIONS:
                # Only calculate if any of this condition's stats are requested
                if any(stat.endswith(f'_{condition}') for stat in stat_keys):
                    cond_stats = self._calculate_condition_stats(df, condition)
                    for stat_name, value in cond_stats.items():
                        if stat_name in stat_keys:
                            stats[stat_name] = value
            
        except Exception as e:
            self.logger.error(f"Error in calculate_summary_stats: {str(e)}")
            raise
                
        return stats
        
    def calculate_composite_indices(self, stats: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate composite indices from the basic statistics.
        
        Args:
            stats: Dictionary of basic statistics.
            
        Returns:
            Dictionary containing composite indices.
        """
        composites = {}
        
        try:
            # Framing index: p_gamble_Loss - p_gamble_Gain (across all conditions)
            if all(k in stats for k in ['prop_gamble_Loss_TC', 'prop_gamble_Gain_TC',
                                      'prop_gamble_Loss_NTC', 'prop_gamble_Gain_NTC']):
                p_gain = (stats['prop_gamble_Gain_TC'] + stats['prop_gamble_Gain_NTC']) / 2
                p_loss = (stats['prop_gamble_Loss_TC'] + stats['prop_gamble_Loss_NTC']) / 2
                composites['framing_index'] = p_loss - p_gain
                
                # Framing index by time constraint
                composites['framing_index_TC'] = stats['prop_gamble_Loss_TC'] - stats['prop_gamble_Gain_TC']
                composites['framing_index_NTC'] = stats['prop_gamble_Loss_NTC'] - stats['prop_gamble_Gain_NTC']
            
            # Time pressure index: p_gamble_TC - p_gamble_NTC (across all frames)
            if all(k in stats for k in ['prop_gamble_Gain_TC', 'prop_gamble_Gain_NTC',
                                      'prop_gamble_Loss_TC', 'prop_gamble_Loss_NTC']):
                p_tc = (stats['prop_gamble_Gain_TC'] + stats['prop_gamble_Loss_TC']) / 2
                p_ntc = (stats['prop_gamble_Gain_NTC'] + stats['prop_gamble_Loss_NTC']) / 2
                composites['time_pressure_index'] = p_tc - p_ntc
                
                # Time pressure by frame
                composites['time_pressure_index_Gain'] = stats['prop_gamble_Gain_TC'] - stats['prop_gamble_Gain_NTC']
                composites['time_pressure_index_Loss'] = stats['prop_gamble_Loss_TC'] - stats['prop_gamble_Loss_NTC']
            
            # Interaction index: (Loss_TC - Gain_TC) - (Loss_NTC - Gain_NTC)
            if all(k in stats for k in ['prop_gamble_Gain_TC', 'prop_gamble_Gain_NTC',
                                      'prop_gamble_Loss_TC', 'prop_gamble_Loss_NTC']):
                framing_tc = stats['prop_gamble_Loss_TC'] - stats['prop_gamble_Gain_TC']
                framing_ntc = stats['prop_gamble_Loss_NTC'] - stats['prop_gamble_Gain_NTC']
                composites['interaction_index'] = framing_tc - framing_ntc
                
        except Exception as e:
            self.logger.warning(f"Error calculating composite indices: {e}")
        
        return composites
        
    def save_outputs(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save module outputs to disk using the data manager.
        
        Args:
            results: Results to save.
            
        Returns:
            Dict mapping output names to their paths.
        """
        output_paths = {}
        
        # Save summary statistics
        summary_stats_path = self.data_manager.get_output_path('summary_stats', 'summary_stats.json')
        self.data_manager.save_json(results['summary_stats'], 'summary_stats', 'summary_stats.json')
        output_paths['summary_stats'] = summary_stats_path
        
        return output_paths
