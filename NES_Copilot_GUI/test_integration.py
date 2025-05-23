"""
Integration test for NES Copilot pipeline

This module contains integration tests for the NES Copilot pipeline.
"""

import os
import unittest
import tempfile
import yaml
import pandas as pd
import numpy as np
from nes_copilot.config_manager import ConfigManager
from nes_copilot.data_manager import DataManager
from nes_copilot.logging_manager import LoggingManager
from nes_copilot.workflow_manager import WorkflowManager
from nes_copilot.data_prep import DataPrepModule
from nes_copilot.simulation import SimulationModule
from nes_copilot.summary_stats import SummaryStatsModule


class TestNESCopilotIntegration(unittest.TestCase):
    """
    Integration tests for the NES Copilot pipeline.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data directory
        self.data_dir = os.path.join(self.temp_dir.name, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create test empirical data
        self.create_test_empirical_data()
        
        # Create test master config
        self.create_test_config()
        
        # Initialize managers
        self.config_manager = ConfigManager(self.master_config_path)
        self.data_manager = DataManager(self.config_manager)
        self.logging_manager = LoggingManager(self.data_manager)
        
    def tearDown(self):
        """
        Tear down test fixtures.
        """
        self.temp_dir.cleanup()
        
    def create_test_empirical_data(self):
        """
        Create test empirical data.
        """
        # Create a DataFrame with empirical data
        data = []
        
        # Create data for 5 subjects
        for subject in range(1, 6):
            # Add gain frame, time-constrained trials
            for _ in range(20):
                data.append({
                    'subject': subject,
                    'choice': 1 if np.random.random() < 0.3 else 0,  # 30% gamble rate
                    'rt': np.random.normal(0.8, 0.2),
                    'frame': 'gain',
                    'cond': 'tc',
                    'valence': np.random.normal(0.5, 0.2)
                })
                
            # Add gain frame, non-time-constrained trials
            for _ in range(20):
                data.append({
                    'subject': subject,
                    'choice': 1 if np.random.random() < 0.4 else 0,  # 40% gamble rate
                    'rt': np.random.normal(1.2, 0.3),
                    'frame': 'gain',
                    'cond': 'ntc',
                    'valence': np.random.normal(0.5, 0.2)
                })
                
            # Add loss frame, time-constrained trials
            for _ in range(20):
                data.append({
                    'subject': subject,
                    'choice': 1 if np.random.random() < 0.6 else 0,  # 60% gamble rate
                    'rt': np.random.normal(0.7, 0.2),
                    'frame': 'loss',
                    'cond': 'tc',
                    'valence': np.random.normal(-0.5, 0.2)
                })
                
            # Add loss frame, non-time-constrained trials
            for _ in range(20):
                data.append({
                    'subject': subject,
                    'choice': 1 if np.random.random() < 0.5 else 0,  # 50% gamble rate
                    'rt': np.random.normal(1.1, 0.3),
                    'frame': 'loss',
                    'cond': 'ntc',
                    'valence': np.random.normal(-0.5, 0.2)
                })
                
        # Create DataFrame
        empirical_df = pd.DataFrame(data)
        
        # Save to CSV
        self.empirical_data_path = os.path.join(self.data_dir, 'roberts_data.csv')
        empirical_df.to_csv(self.empirical_data_path, index=False)
        
    def create_test_config(self):
        """
        Create test configuration files.
        """
        # Create master config
        self.master_config = {
            'output_dir': os.path.join(self.temp_dir.name, 'output'),
            'run_id': 'test_run',
            'seed': 12345,
            'device': 'cpu',
            'data': {
                'roberts_data': self.empirical_data_path
            },
            'run_modules': {
                'data_prep': True,
                'simulation': True,
                'summary_stats': True,
                'npe_training': False,
                'sbc': False,
                'empirical_fit': False,
                'ppc': False,
                'analysis': False
            },
            'module_configs': {}
        }
        
        # Create data_prep config
        self.data_prep_config = {
            'filtering': {
                'trial_type': 'all',
                'exclude_subjects': []
            },
            'valence_processor': {
                'use_existing_valence': True,
                'valence_column': 'valence',
                'rescaling': {
                    'min_value': -1.0,
                    'max_value': 1.0
                }
            },
            'trial_generator': {
                'num_trials': 80,
                'frame_distribution': {
                    'gain': 0.5,
                    'loss': 0.5
                },
                'condition_distribution': {
                    'tc': 0.5,
                    'ntc': 0.5
                }
            }
        }
        
        # Create simulation config
        self.simulation_config = {
            'mvnes_agent': {
                'fixed_params': {
                    'beta': 1.0,
                    'tau': 0.1
                },
                'variable_params': {
                    'v_norm': 2.0,
                    'w_gain': 0.5,
                    'w_loss': 0.5,
                    'theta': 0.7
                }
            },
            'num_simulations': 1
        }
        
        # Create summary_stats config
        self.summary_stats_config = {
            'stat_set': 'full_60',
            'custom_stats': []
        }
        
        # Save configs to files
        self.master_config_path = os.path.join(self.temp_dir.name, 'master_config.yaml')
        with open(self.master_config_path, 'w') as f:
            yaml.dump(self.master_config, f)
            
        # Add module configs to master config
        self.master_config['module_configs']['data_prep'] = os.path.join(self.temp_dir.name, 'data_prep_config.yaml')
        self.master_config['module_configs']['simulation'] = os.path.join(self.temp_dir.name, 'simulation_config.yaml')
        self.master_config['module_configs']['summary_stats'] = os.path.join(self.temp_dir.name, 'summary_stats_config.yaml')
        
        # Save updated master config
        with open(self.master_config_path, 'w') as f:
            yaml.dump(self.master_config, f)
            
        # Save module configs
        with open(self.master_config['module_configs']['data_prep'], 'w') as f:
            yaml.dump(self.data_prep_config, f)
            
        with open(self.master_config['module_configs']['simulation'], 'w') as f:
            yaml.dump(self.simulation_config, f)
            
        with open(self.master_config['module_configs']['summary_stats'], 'w') as f:
            yaml.dump(self.summary_stats_config, f)
            
    def test_data_prep_to_summary_stats_integration(self):
        """
        Test integration from data preparation to summary statistics.
        """
        # Initialize modules
        data_prep_module = DataPrepModule(self.config_manager, self.data_manager, self.logging_manager)
        simulation_module = SimulationModule(self.config_manager, self.data_manager, self.logging_manager)
        summary_stats_module = SummaryStatsModule(self.config_manager, self.data_manager, self.logging_manager)
        
        # Run data preparation
        data_prep_results = data_prep_module.run()
        
        # Check that data preparation produced expected outputs
        self.assertIn('empirical_data', data_prep_results)
        self.assertIn('trial_template', data_prep_results)
        self.assertIn('output_paths', data_prep_results)
        
        # Get trial template
        trial_template = data_prep_results['trial_template']
        
        # Run simulation
        simulation_params = self.simulation_config['mvnes_agent']['variable_params']
        simulation_results = simulation_module.run(simulation_params, trial_template)
        
        # Check that simulation produced expected outputs
        self.assertIn('simulated_data', simulation_results)
        self.assertIn('output_paths', simulation_results)
        
        # Get simulated data
        simulated_data = simulation_results['simulated_data']
        
        # Run summary statistics calculation
        stats_results = summary_stats_module.run(simulated_data)
        
        # Check that summary statistics calculation produced expected outputs
        self.assertIn('summary_stats', stats_results)
        self.assertIn('stat_keys', stats_results)
        self.assertIn('output_paths', stats_results)
        
        # Check that summary statistics contain expected keys
        summary_stats = stats_results['summary_stats']
        expected_keys = ['p_gamble_Gain', 'p_gamble_Loss', 'framing_index']
        for key in expected_keys:
            self.assertIn(key, summary_stats)
            
    def test_workflow_manager_integration(self):
        """
        Test integration using the WorkflowManager.
        """
        # Initialize workflow manager
        workflow_manager = WorkflowManager(self.config_manager, self.data_manager, self.logging_manager)
        
        # Run pipeline
        results = workflow_manager.run_pipeline()
        
        # Check that pipeline produced expected outputs
        self.assertIn('data_prep', results)
        self.assertIn('simulation', results)
        self.assertIn('summary_stats', results)
        
        # Check that each module produced expected outputs
        self.assertIn('empirical_data', results['data_prep'])
        self.assertIn('trial_template', results['data_prep'])
        
        self.assertIn('simulated_data', results['simulation'])
        
        self.assertIn('summary_stats', results['summary_stats'])
        self.assertIn('stat_keys', results['summary_stats'])


if __name__ == '__main__':
    unittest.main()
