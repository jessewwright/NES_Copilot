"""
Data Manager for NES Copilot

This module handles managing input and output paths, creating output directories,
and tracking metadata for each run.
"""

import os
import json
import datetime
import shutil
from typing import Dict, Any, List, Optional, Union


class DataManager:
    """
    Data manager for the NES Copilot system.
    
    Handles input and output paths, creates output directories, and tracks
    metadata for each run.
    """
    
    def __init__(self, config_manager, logger=None):
        """
        Initialize the data manager.
        
        Args:
            config_manager: Configuration manager instance.
            logger: Logger instance (optional).
        """
        self.config_manager = config_manager
        self.logger = logger
        
        # Get base output directory and run ID from config
        self.base_output_dir = self.config_manager.get_param('output_dir')
        self.run_id = self.config_manager.get_param('run_id')
        
        # If run_id is not specified, generate one based on timestamp
        if not self.run_id:
            self.run_id = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
            
        # Create the run output directory
        self.run_output_dir = os.path.join(self.base_output_dir, self.run_id)
        os.makedirs(self.run_output_dir, exist_ok=True)
        
        # Create subdirectories for each module
        self.module_dirs = {}
        self._create_module_directories()
        
        # Initialize metadata
        self.metadata = {
            'run_id': self.run_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'modules': {},
        }
    
    def _create_module_directories(self):
        """
        Create output directories for each module.
        """
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
            module_dir = os.path.join(self.run_output_dir, module)
            os.makedirs(module_dir, exist_ok=True)
            self.module_dirs[module] = module_dir
    
    def get_run_output_dir(self) -> str:
        """
        Get the output directory for the current run.
        
        Returns:
            Path to the run output directory.
        """
        return self.run_output_dir
    
    def get_module_dir(self, module_name: str) -> str:
        """
        Get the output directory for a specific module.
        
        Args:
            module_name: Name of the module.
            
        Returns:
            Path to the module output directory.
        """
        if module_name not in self.module_dirs:
            raise KeyError(f"Output directory for module '{module_name}' not found")
        
        return self.module_dirs[module_name]
    
    def get_input_path(self, input_name: str) -> str:
        """
        Get the path to an input file specified in the configuration.
        
        Args:
            input_name: Name of the input file (e.g., 'roberts_data').
            
        Returns:
            Path to the input file.
        """
        input_path = self.config_manager.get_param(f'data.{input_name}')
        if not input_path:
            raise ValueError(f"Input path '{input_name}' not found in configuration")
        
        return input_path
    
    def get_output_path(self, module_name: str, filename: str) -> str:
        """
        Get the path to an output file for a specific module.
        
        Args:
            module_name: Name of the module.
            filename: Name of the output file.
            
        Returns:
            Path to the output file.
        """
        module_dir = self.get_module_dir(module_name)
        return os.path.join(module_dir, filename)
    
    def save_metadata(self, module_name: str, metadata: Dict[str, Any]):
        """
        Save metadata for a specific module.
        
        Args:
            module_name: Name of the module.
            metadata: Dictionary of metadata to save.
        """
        # Update the module metadata
        self.metadata['modules'][module_name] = metadata
        
        # Save the updated metadata to disk
        metadata_path = os.path.join(self.run_output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def copy_file(self, source_path: str, module_name: str, dest_filename: Optional[str] = None) -> str:
        """
        Copy a file to a module's output directory.
        
        Args:
            source_path: Path to the source file.
            module_name: Name of the module.
            dest_filename: Name of the destination file (optional, defaults to source filename).
            
        Returns:
            Path to the copied file.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if not dest_filename:
            dest_filename = os.path.basename(source_path)
            
        dest_path = self.get_output_path(module_name, dest_filename)
        shutil.copy2(source_path, dest_path)
        
        return dest_path
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            Dictionary containing the JSON data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return data
    
    def save_json(self, data: Dict[str, Any], module_name: str, filename: str) -> str:
        """
        Save data to a JSON file in a module's output directory.
        
        Args:
            data: Data to save.
            module_name: Name of the module.
            filename: Name of the output file.
            
        Returns:
            Path to the saved file.
        """
        output_path = self.get_output_path(module_name, filename)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return output_path
