"""
Unit tests for Summary Statistics Module

This module contains unit tests for the SummaryStatsModule class.
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
from nes_copilot.summary_stats import SummaryStatsModule


class MockConfigManager:
    """
    Mock ConfigManager for testing.
    """
    
    def __init__(self):
        self.module_configs = {
            'summary_stats': {
                'stat_set': 'full_60',
                'custom_stats': []
            }
        }
        
    def get_module_config(self, module_name):
        return self.module_configs.get(module_name, {})
        
    def get_param(self, param_path, default=None):
        return default


class MockDataManager:
    """
    Mock DataManager for testing.
    """
    
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def get_output_path(self, module, filename):
        os.makedirs(os.path.join(self.temp_dir.name, module), exist_ok=True)
        return os.path.join(self.temp_dir.name, module, filename)
        
    def save_json(self, data, module, filename):
        path = self.get_output_path(module, filename)
        return path
        
    def cleanup(self):
        self.temp_dir.cleanup()


class MockLoggingManager:
    """
    Mock LoggingManager for testing.
    """
    
    def __init__(self):
        pass
        
    def get_logger(self):
        return self
        
    def info(self, message):
        pass
        
    def warning(self, message):
        pass
        
    def error(self, message):
        pass


class TestSummaryStatsModule(unittest.TestCase):
    """
    Unit tests for the SummaryStatsModule class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create mock managers
        self.config_manager = MockConfigManager()
        self.data_manager = MockDataManager()
        self.logging_manager = MockLoggingManager()
        
        # Initialize SummaryStatsModule
        self.summary_stats_module = SummaryStatsModule(
            self.config_manager,
            self.data_manager,
            self.logging_manager
        )
        
        # Create test data
        self.create_test_data()
        
    def tearDown(self):
        """
        Tear down test fixtures.
        """
        self.data_manager.cleanup()
        
    def create_test_data(self):
        """
        Create test data for summary statistics calculation.
        """
        # Create a DataFrame with trial data
        data = []
        
        # Add gain frame, time-constrained trials
        for _ in range(50):
            data.append({
                'choice': 1 if np.random.random() < 0.3 else 0,  # 30% gamble rate
                'rt': np.random.normal(0.8, 0.2),
                'frame': 'gain',
                'cond': 'tc',
                'is_gain_frame': True,
                'time_constrained': True
            })
            
        # Add gain frame, non-time-constrained trials
        for _ in range(50):
            data.append({
                'choice': 1 if np.random.random() < 0.4 else 0,  # 40% gamble rate
                'rt': np.random.normal(1.2, 0.3),
                'frame': 'gain',
                'cond': 'ntc',
                'is_gain_frame': True,
                'time_constrained': False
            })
            
        # Add loss frame, time-constrained trials
        for _ in range(50):
            data.append({
                'choice': 1 if np.random.random() < 0.6 else 0,  # 60% gamble rate
                'rt': np.random.normal(0.7, 0.2),
                'frame': 'loss',
                'cond': 'tc',
                'is_gain_frame': False,
                'time_constrained': True
            })
            
        # Add loss frame, non-time-constrained trials
        for _ in range(50):
            data.append({
                'choice': 1 if np.random.random() < 0.5 else 0,  # 50% gamble rate
                'rt': np.random.normal(1.1, 0.3),
                'frame': 'loss',
                'cond': 'ntc',
                'is_gain_frame': False,
                'time_constrained': False
            })
            
        self.trials_df = pd.DataFrame(data)
        
    def test_validate_inputs(self):
        """
        Test input validation.
        """
        # Valid input should pass validation
        self.assertTrue(self.summary_stats_module.validate_inputs(self.trials_df))
        
        # Test with missing DataFrame
        with self.assertRaises(ValueError):
            self.summary_stats_module.validate_inputs(None)
            
        # Test with empty DataFrame
        with self.assertRaises(ValueError):
            self.summary_stats_module.validate_inputs(pd.DataFrame())
            
        # Test with missing required columns
        invalid_df = self.trials_df.drop(columns=['choice'])
        with self.assertRaises(ValueError):
            self.summary_stats_module.validate_inputs(invalid_df)
            
    def test_calculate_summary_stats(self):
        """
        Test summary statistics calculation.
        """
        # Calculate summary statistics
        stat_keys = [
            'p_gamble_Gain_TC',
            'p_gamble_Gain_NTC',
            'p_gamble_Loss_TC',
            'p_gamble_Loss_NTC',
            'mean_rt_Gain_TC',
            'mean_rt_Gain_NTC',
            'mean_rt_Loss_TC',
            'mean_rt_Loss_NTC',
            'framing_index',
            'time_pressure_index'
        ]
        
        stats = self.summary_stats_module.calculate_summary_stats(self.trials_df, stat_keys)
        
        # Check that all requested statistics were calculated
        for key in stat_keys:
            self.assertIn(key, stats)
            self.assertFalse(np.isnan(stats[key]))
            
        # Check specific statistics
        # p_gamble values should be between 0 and 1
        self.assertGreaterEqual(stats['p_gamble_Gain_TC'], 0)
        self.assertLessEqual(stats['p_gamble_Gain_TC'], 1)
        
        # mean_rt values should be positive
        self.assertGreater(stats['mean_rt_Gain_TC'], 0)
        
        # framing_index should be the difference between loss and gain p_gamble
        p_gamble_gain = self.trials_df[self.trials_df['is_gain_frame']]['choice'].mean()
        p_gamble_loss = self.trials_df[~self.trials_df['is_gain_frame']]['choice'].mean()
        expected_framing_index = p_gamble_loss - p_gamble_gain
        self.assertAlmostEqual(stats['framing_index'], expected_framing_index, places=6)
        
    def test_run(self):
        """
        Test the run method.
        """
        # Run the module
        results = self.summary_stats_module.run(self.trials_df)
        
        # Check that results contain expected keys
        self.assertIn('summary_stats', results)
        self.assertIn('stat_keys', results)
        self.assertIn('output_paths', results)
        
        # Check that summary_stats contains all expected statistics
        summary_stats = results['summary_stats']
        for key in self.summary_stats_module.default_stat_keys:
            self.assertIn(key, summary_stats)
            
        # Check that output_paths contains the expected path
        self.assertIn('summary_stats', results['output_paths'])
        
    def test_calculate_p_gamble(self):
        """
        Test p_gamble calculation.
        """
        # Calculate p_gamble for gain frame, time-constrained condition
        p_gamble = self.summary_stats_module._calculate_p_gamble(self.trials_df, 'p_gamble_Gain_TC')
        
        # Calculate expected value
        mask = (self.trials_df['is_gain_frame']) & (self.trials_df['time_constrained'])
        expected_p_gamble = self.trials_df[mask]['choice'].mean()
        
        # Check that calculated value matches expected value
        self.assertAlmostEqual(p_gamble, expected_p_gamble, places=6)
        
    def test_calculate_mean_rt(self):
        """
        Test mean_rt calculation.
        """
        # Calculate mean_rt for gain frame, time-constrained condition
        mean_rt = self.summary_stats_module._calculate_mean_rt(self.trials_df, 'mean_rt_Gain_TC')
        
        # Calculate expected value
        mask = (self.trials_df['is_gain_frame']) & (self.trials_df['time_constrained'])
        expected_mean_rt = self.trials_df[mask]['rt'].mean()
        
        # Check that calculated value matches expected value
        self.assertAlmostEqual(mean_rt, expected_mean_rt, places=6)
        
    def test_calculate_framing_index(self):
        """
        Test framing_index calculation.
        """
        # Calculate framing index
        framing_index = self.summary_stats_module._calculate_framing_index(self.trials_df)
        
        # Calculate expected value
        p_gamble_gain = self.trials_df[self.trials_df['is_gain_frame']]['choice'].mean()
        p_gamble_loss = self.trials_df[~self.trials_df['is_gain_frame']]['choice'].mean()
        expected_framing_index = p_gamble_loss - p_gamble_gain
        
        # Check that calculated value matches expected value
        self.assertAlmostEqual(framing_index, expected_framing_index, places=6)


if __name__ == '__main__':
    unittest.main()
