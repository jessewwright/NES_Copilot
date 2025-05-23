"""
Summary Statistics Module for NES Copilot

This module handles calculation of summary statistics for trial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

from nes_copilot.module_base import ModuleBase


class SummaryStatsModule(ModuleBase):
    """
    Summary statistics module for the NES Copilot system.
    
    Handles calculation of summary statistics for trial data.
    """
    
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
        
        # Import stats schema
        from nes_copilot.summary_stats.stats_schema import ROBERTS_SUMMARY_STAT_KEYS
        self.default_stat_keys = ROBERTS_SUMMARY_STAT_KEYS
        
    def run(self, trials_df: pd.DataFrame, stat_keys: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate summary statistics for the given trials.
        
        Args:
            trials_df: DataFrame of trials (simulated or empirical).
            stat_keys: Optional list of specific statistics to calculate.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the calculated summary statistics.
        """
        self.logger.info("Starting summary statistics calculation")
        
        # Validate inputs
        self.validate_inputs(trials_df=trials_df)
        
        # Determine which statistics to calculate
        if stat_keys is None:
            stat_set = self.config.get('stat_set', 'full_60')
            if stat_set == 'full_60':
                stat_keys = self.default_stat_keys
            elif stat_set == 'lean_25':
                # Subset of the most important statistics
                stat_keys = self.default_stat_keys[:25]
            else:
                self.logger.warning(f"Unknown stat_set '{stat_set}', using full set")
                stat_keys = self.default_stat_keys
                
            # Add any custom statistics
            custom_stats = self.config.get('custom_stats', [])
            stat_keys.extend(custom_stats)
            
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
            True if inputs are valid, False otherwise.
        """
        # Check if trials DataFrame is provided
        if trials_df is None or len(trials_df) == 0:
            raise ValueError("Trials DataFrame not provided or empty")
            
        # Check if trials DataFrame has required columns
        required_columns = ['choice', 'rt', 'frame', 'cond']
        for col in required_columns:
            if col not in trials_df.columns:
                raise ValueError(f"Trials DataFrame missing required column: {col}")
                
        return True
        
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
        
        # Ensure required columns exist
        if 'is_gain_frame' not in df.columns:
            df['is_gain_frame'] = df['frame'] == 'gain'
        if 'time_constrained' not in df.columns:
            df['time_constrained'] = df['cond'] == 'tc'
            
        # Initialize results dictionary
        stats = {}
        
        # Calculate each requested statistic
        for key in stat_keys:
            try:
                # Dispatch to the appropriate calculation method
                if key.startswith('p_gamble_'):
                    stats[key] = self._calculate_p_gamble(df, key)
                elif key.startswith('mean_rt_'):
                    stats[key] = self._calculate_mean_rt(df, key)
                elif key.startswith('std_rt_'):
                    stats[key] = self._calculate_std_rt(df, key)
                elif key == 'framing_index':
                    stats[key] = self._calculate_framing_index(df)
                elif key == 'time_pressure_index':
                    stats[key] = self._calculate_time_pressure_index(df)
                else:
                    self.logger.warning(f"Unknown statistic: {key}")
                    stats[key] = np.nan
            except Exception as e:
                self.logger.error(f"Error calculating statistic {key}: {str(e)}")
                stats[key] = np.nan
                
        return stats
        
    def _calculate_p_gamble(self, df: pd.DataFrame, key: str) -> float:
        """
        Calculate probability of gambling for a specific condition.
        
        Args:
            df: DataFrame of trials.
            key: Statistic key (e.g., 'p_gamble_Gain_TC').
            
        Returns:
            Probability of gambling.
        """
        # Parse the key to determine the condition
        parts = key.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid p_gamble key format: {key}")
            
        frame = parts[2] if parts[2] in ['Gain', 'Loss'] else None
        cond = parts[3] if len(parts) > 3 and parts[3] in ['TC', 'NTC'] else None
        
        # Filter the DataFrame based on the condition
        mask = pd.Series(True, index=df.index)
        if frame == 'Gain':
            mask &= df['is_gain_frame']
        elif frame == 'Loss':
            mask &= ~df['is_gain_frame']
            
        if cond == 'TC':
            mask &= df['time_constrained']
        elif cond == 'NTC':
            mask &= ~df['time_constrained']
            
        # Calculate the probability of gambling
        filtered_df = df[mask]
        if len(filtered_df) == 0:
            return np.nan
            
        return filtered_df['choice'].mean()
        
    def _calculate_mean_rt(self, df: pd.DataFrame, key: str) -> float:
        """
        Calculate mean reaction time for a specific condition.
        
        Args:
            df: DataFrame of trials.
            key: Statistic key (e.g., 'mean_rt_Gain_TC').
            
        Returns:
            Mean reaction time.
        """
        # Parse the key to determine the condition
        parts = key.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid mean_rt key format: {key}")
            
        frame = parts[2] if parts[2] in ['Gain', 'Loss'] else None
        cond = parts[3] if len(parts) > 3 and parts[3] in ['TC', 'NTC'] else None
        
        # Filter the DataFrame based on the condition
        mask = pd.Series(True, index=df.index)
        if frame == 'Gain':
            mask &= df['is_gain_frame']
        elif frame == 'Loss':
            mask &= ~df['is_gain_frame']
            
        if cond == 'TC':
            mask &= df['time_constrained']
        elif cond == 'NTC':
            mask &= ~df['time_constrained']
            
        # Calculate the mean reaction time
        filtered_df = df[mask]
        if len(filtered_df) == 0:
            return np.nan
            
        return filtered_df['rt'].mean()
        
    def _calculate_std_rt(self, df: pd.DataFrame, key: str) -> float:
        """
        Calculate standard deviation of reaction time for a specific condition.
        
        Args:
            df: DataFrame of trials.
            key: Statistic key (e.g., 'std_rt_Gain_TC').
            
        Returns:
            Standard deviation of reaction time.
        """
        # Parse the key to determine the condition
        parts = key.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid std_rt key format: {key}")
            
        frame = parts[2] if parts[2] in ['Gain', 'Loss'] else None
        cond = parts[3] if len(parts) > 3 and parts[3] in ['TC', 'NTC'] else None
        
        # Filter the DataFrame based on the condition
        mask = pd.Series(True, index=df.index)
        if frame == 'Gain':
            mask &= df['is_gain_frame']
        elif frame == 'Loss':
            mask &= ~df['is_gain_frame']
            
        if cond == 'TC':
            mask &= df['time_constrained']
        elif cond == 'NTC':
            mask &= ~df['time_constrained']
            
        # Calculate the standard deviation of reaction time
        filtered_df = df[mask]
        if len(filtered_df) == 0:
            return np.nan
            
        return filtered_df['rt'].std()
        
    def _calculate_framing_index(self, df: pd.DataFrame) -> float:
        """
        Calculate the framing index.
        
        Args:
            df: DataFrame of trials.
            
        Returns:
            Framing index.
        """
        # Calculate p_gamble for gain and loss frames
        p_gamble_gain = df[df['is_gain_frame']]['choice'].mean()
        p_gamble_loss = df[~df['is_gain_frame']]['choice'].mean()
        
        # Calculate framing index
        return p_gamble_loss - p_gamble_gain
        
    def _calculate_time_pressure_index(self, df: pd.DataFrame) -> float:
        """
        Calculate the time pressure index.
        
        Args:
            df: DataFrame of trials.
            
        Returns:
            Time pressure index.
        """
        # Calculate p_gamble for time-constrained and non-time-constrained conditions
        p_gamble_tc = df[df['time_constrained']]['choice'].mean()
        p_gamble_ntc = df[~df['time_constrained']]['choice'].mean()
        
        # Calculate time pressure index
        return p_gamble_tc - p_gamble_ntc
        
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
