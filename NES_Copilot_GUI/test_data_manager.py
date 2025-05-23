"""
Unit tests for Data Manager

This module contains unit tests for the DataManager class.
"""

import os
import unittest
import tempfile
import json
from nes_copilot.config_manager import ConfigManager
from nes_copilot.data_manager import DataManager


class MockConfigManager:
    """
    Mock ConfigManager for testing.
    """
    
    def __init__(self, output_dir, run_id=None):
        self.output_dir = output_dir
        self.run_id = run_id
        
    def get_param(self, param_path, default=None):
        if param_path == 'output_dir':
            return self.output_dir
        elif param_path == 'run_id':
            return self.run_id
        elif param_path == 'data.roberts_data':
            return '/path/to/roberts_data.csv'
        return default


class TestDataManager(unittest.TestCase):
    """
    Unit tests for the DataManager class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock config manager
        self.config_manager = MockConfigManager(self.temp_dir.name, 'test_run')
        
        # Initialize DataManager
        self.data_manager = DataManager(self.config_manager)
        
    def tearDown(self):
        """
        Tear down test fixtures.
        """
        self.temp_dir.cleanup()
        
    def test_directory_creation(self):
        """
        Test that output directories are created correctly.
        """
        # Check that run output directory was created
        run_output_dir = self.data_manager.get_run_output_dir()
        self.assertTrue(os.path.exists(run_output_dir))
        
        # Check that module directories were created
        modules = [
            'data_prep',
            'simulation',
            'summary_stats',
            'npe',
            'sbc',
            'empirical_fit',
            'ppc',
            'analysis',
            'config',
            'logs',
        ]
        
        for module in modules:
            module_dir = self.data_manager.get_module_dir(module)
            self.assertTrue(os.path.exists(module_dir))
            
    def test_get_run_output_dir(self):
        """
        Test getting the run output directory.
        """
        run_output_dir = self.data_manager.get_run_output_dir()
        expected_dir = os.path.join(self.temp_dir.name, 'test_run')
        self.assertEqual(run_output_dir, expected_dir)
        
    def test_get_module_dir(self):
        """
        Test getting module directories.
        """
        data_prep_dir = self.data_manager.get_module_dir('data_prep')
        expected_dir = os.path.join(self.temp_dir.name, 'test_run', 'data_prep')
        self.assertEqual(data_prep_dir, expected_dir)
        
        # Test with non-existent module
        with self.assertRaises(KeyError):
            self.data_manager.get_module_dir('non_existent_module')
            
    def test_get_input_path(self):
        """
        Test getting input paths.
        """
        roberts_data_path = self.data_manager.get_input_path('roberts_data')
        self.assertEqual(roberts_data_path, '/path/to/roberts_data.csv')
        
        # Test with non-existent input
        with self.assertRaises(ValueError):
            self.data_manager.get_input_path('non_existent_input')
            
    def test_get_output_path(self):
        """
        Test getting output paths.
        """
        output_path = self.data_manager.get_output_path('data_prep', 'test_output.csv')
        expected_path = os.path.join(self.temp_dir.name, 'test_run', 'data_prep', 'test_output.csv')
        self.assertEqual(output_path, expected_path)
        
    def test_save_metadata(self):
        """
        Test saving metadata.
        """
        # Save metadata for a module
        metadata = {'test_key': 'test_value'}
        self.data_manager.save_metadata('data_prep', metadata)
        
        # Check that metadata was saved
        metadata_path = os.path.join(self.temp_dir.name, 'test_run', 'metadata.json')
        self.assertTrue(os.path.exists(metadata_path))
        
        # Check content of metadata file
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
            self.assertEqual(saved_metadata['modules']['data_prep'], metadata)
            
    def test_save_and_load_json(self):
        """
        Test saving and loading JSON data.
        """
        # Save JSON data
        data = {'test_key': 'test_value'}
        json_path = self.data_manager.save_json(data, 'data_prep', 'test_data.json')
        
        # Check that file was saved
        self.assertTrue(os.path.exists(json_path))
        
        # Load JSON data
        loaded_data = self.data_manager.load_json(json_path)
        self.assertEqual(loaded_data, data)
        
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            self.data_manager.load_json('/non/existent/file.json')
            
    def test_copy_file(self):
        """
        Test copying files.
        """
        # Create a test file
        source_path = os.path.join(self.temp_dir.name, 'test_file.txt')
        with open(source_path, 'w') as f:
            f.write('test content')
            
        # Copy the file
        dest_path = self.data_manager.copy_file(source_path, 'data_prep')
        
        # Check that file was copied
        self.assertTrue(os.path.exists(dest_path))
        
        # Check content of copied file
        with open(dest_path, 'r') as f:
            content = f.read()
            self.assertEqual(content, 'test content')
            
        # Test copying non-existent file
        with self.assertRaises(FileNotFoundError):
            self.data_manager.copy_file('/non/existent/file.txt', 'data_prep')


if __name__ == '__main__':
    unittest.main()
