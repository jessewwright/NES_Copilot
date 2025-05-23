"""
Unit tests for Configuration Manager

This module contains unit tests for the ConfigManager class.
"""

import os
import unittest
import tempfile
import yaml
from nes_copilot.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """
    Unit tests for the ConfigManager class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test master config
        self.master_config = {
            'output_dir': '/path/to/output',
            'run_id': 'test_run',
            'seed': 12345,
            'device': 'cpu',
            'data': {
                'roberts_data': '/path/to/roberts_data.csv'
            },
            'run_modules': {
                'data_prep': True,
                'npe_training': True,
                'sbc': False
            },
            'module_configs': {
                'data_prep': os.path.join(self.temp_dir.name, 'data_prep_config.yaml'),
                'npe': os.path.join(self.temp_dir.name, 'npe_config.yaml')
            }
        }
        
        # Create test module configs
        self.data_prep_config = {
            'filtering': {
                'trial_type': 'target',
                'exclude_subjects': [1, 2, 3]
            },
            'valence_processor': {
                'model_name': 'test_model',
                'rescaling': {
                    'min_value': -1.0,
                    'max_value': 1.0
                }
            }
        }
        
        self.npe_config = {
            'num_training_sims': 1000,
            'batch_size': 50,
            'prior': {
                'v_norm': {
                    'distribution': 'uniform',
                    'low': 0.0,
                    'high': 5.0
                }
            }
        }
        
        # Write configs to temporary files
        self.master_config_path = os.path.join(self.temp_dir.name, 'master_config.yaml')
        with open(self.master_config_path, 'w') as f:
            yaml.dump(self.master_config, f)
            
        with open(self.master_config['module_configs']['data_prep'], 'w') as f:
            yaml.dump(self.data_prep_config, f)
            
        with open(self.master_config['module_configs']['npe'], 'w') as f:
            yaml.dump(self.npe_config, f)
            
        # Initialize ConfigManager
        self.config_manager = ConfigManager(self.master_config_path)
        
    def tearDown(self):
        """
        Tear down test fixtures.
        """
        self.temp_dir.cleanup()
        
    def test_get_master_config(self):
        """
        Test getting the master configuration.
        """
        master_config = self.config_manager.get_master_config()
        self.assertEqual(master_config['run_id'], 'test_run')
        self.assertEqual(master_config['seed'], 12345)
        self.assertEqual(master_config['device'], 'cpu')
        
    def test_get_module_config(self):
        """
        Test getting module configurations.
        """
        data_prep_config = self.config_manager.get_module_config('data_prep')
        self.assertEqual(data_prep_config['filtering']['trial_type'], 'target')
        self.assertEqual(data_prep_config['valence_processor']['model_name'], 'test_model')
        
        npe_config = self.config_manager.get_module_config('npe')
        self.assertEqual(npe_config['num_training_sims'], 1000)
        self.assertEqual(npe_config['batch_size'], 50)
        
    def test_get_param(self):
        """
        Test getting parameters using dot notation.
        """
        # Test getting master config params
        self.assertEqual(self.config_manager.get_param('run_id'), 'test_run')
        self.assertEqual(self.config_manager.get_param('seed'), 12345)
        
        # Test getting module config params
        self.assertEqual(self.config_manager.get_param('data_prep.filtering.trial_type'), 'target')
        self.assertEqual(self.config_manager.get_param('npe.num_training_sims'), 1000)
        
        # Test getting nested params
        self.assertEqual(self.config_manager.get_param('data_prep.valence_processor.rescaling.min_value'), -1.0)
        
        # Test default value for non-existent param
        self.assertEqual(self.config_manager.get_param('non_existent_param', 'default'), 'default')
        
    def test_validate_config(self):
        """
        Test configuration validation.
        """
        # Valid configuration should pass validation
        self.assertTrue(self.config_manager.validate_config())
        
        # Test with missing required parameter
        invalid_config = {
            'run_id': 'test_run',
            'seed': 12345,
            # Missing 'output_dir'
            'run_modules': {}
        }
        
        invalid_config_path = os.path.join(self.temp_dir.name, 'invalid_config.yaml')
        with open(invalid_config_path, 'w') as f:
            yaml.dump(invalid_config, f)
            
        invalid_config_manager = ConfigManager(invalid_config_path)
        with self.assertRaises(ValueError):
            invalid_config_manager.validate_config()
            
    def test_save_config_snapshot(self):
        """
        Test saving a configuration snapshot.
        """
        snapshot_dir = os.path.join(self.temp_dir.name, 'snapshot')
        os.makedirs(snapshot_dir, exist_ok=True)
        
        saved_paths = self.config_manager.save_config_snapshot(snapshot_dir)
        
        # Check that files were saved
        self.assertTrue(os.path.exists(saved_paths['master']))
        self.assertTrue(os.path.exists(saved_paths['data_prep']))
        self.assertTrue(os.path.exists(saved_paths['npe']))
        
        # Check content of saved files
        with open(saved_paths['master'], 'r') as f:
            saved_master_config = yaml.safe_load(f)
            self.assertEqual(saved_master_config['run_id'], 'test_run')
            
        with open(saved_paths['data_prep'], 'r') as f:
            saved_data_prep_config = yaml.safe_load(f)
            self.assertEqual(saved_data_prep_config['filtering']['trial_type'], 'target')


if __name__ == '__main__':
    unittest.main()
