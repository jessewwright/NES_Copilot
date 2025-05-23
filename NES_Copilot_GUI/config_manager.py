"""
Configuration Manager for NES Copilot

This module handles loading, validating, and providing access to configuration parameters
for the NES Copilot system.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union


class ConfigManager:
    """
    Configuration manager for the NES Copilot system.
    
    Handles loading and validating configuration files, and provides a unified
    interface for accessing configuration parameters.
    """
    
    def __init__(self, master_config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            master_config_path: Path to the master configuration file.
        """
        self.master_config_path = master_config_path
        self.master_config = self._load_yaml(master_config_path)
        self.module_configs = {}
        self._load_module_configs()
        
    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            Dict containing the configuration parameters.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def _load_module_configs(self):
        """
        Load all module-specific configuration files specified in the master config.
        """
        if 'module_configs' not in self.master_config:
            return
        
        for module_name, config_path in self.master_config['module_configs'].items():
            # Handle relative paths
            if not os.path.isabs(config_path):
                base_dir = os.path.dirname(self.master_config_path)
                config_path = os.path.join(base_dir, config_path)
                
            self.module_configs[module_name] = self._load_yaml(config_path)
    
    def get_master_config(self) -> Dict[str, Any]:
        """
        Get the complete master configuration.
        
        Returns:
            Dict containing the master configuration.
        """
        return self.master_config
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific module.
        
        Args:
            module_name: Name of the module (e.g., 'data_prep', 'npe', etc.).
            
        Returns:
            Dict containing the module configuration.
        """
        if module_name not in self.module_configs:
            raise KeyError(f"Configuration for module '{module_name}' not found")
        
        return self.module_configs[module_name]
    
    def get_param(self, param_path: str, default: Any = None) -> Any:
        """
        Get a parameter from the configuration using a dot-notation path.
        
        Args:
            param_path: Path to the parameter (e.g., 'data_prep.filtering.trial_type').
            default: Default value to return if the parameter is not found.
            
        Returns:
            The parameter value, or the default if not found.
        """
        parts = param_path.split('.')
        
        if len(parts) == 1:
            # Parameter is in the master config
            return self.master_config.get(parts[0], default)
        
        # First part is the module name
        module_name = parts[0]
        if module_name not in self.module_configs:
            return default
        
        # Navigate through the nested dictionary
        config = self.module_configs[module_name]
        for part in parts[1:]:
            if not isinstance(config, dict) or part not in config:
                return default
            config = config[part]
            
        return config
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for consistency and required parameters.
        
        Returns:
            True if the configuration is valid, False otherwise.
        """
        # Basic validation of master config
        required_master_params = ['output_dir', 'run_modules']
        for param in required_master_params:
            if param not in self.master_config:
                raise ValueError(f"Required parameter '{param}' not found in master config")
        
        # Validate module configs based on which modules are enabled
        run_modules = self.master_config.get('run_modules', {})
        for module_name, enabled in run_modules.items():
            if enabled and module_name not in self.module_configs:
                raise ValueError(f"Module '{module_name}' is enabled but its configuration is missing")
        
        # Module-specific validation could be added here
        
        return True
    
    def save_config_snapshot(self, output_dir: str) -> Dict[str, str]:
        """
        Save a snapshot of all configuration files to the output directory.
        
        Args:
            output_dir: Directory to save the configuration snapshot.
            
        Returns:
            Dict mapping config names to their saved paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = {}
        
        # Save master config
        master_config_name = os.path.basename(self.master_config_path)
        master_config_out_path = os.path.join(output_dir, master_config_name)
        with open(master_config_out_path, 'w') as f:
            yaml.dump(self.master_config, f, default_flow_style=False)
        saved_paths['master'] = master_config_out_path
        
        # Save module configs
        for module_name, config in self.module_configs.items():
            module_config_name = f"{module_name}_config.yaml"
            module_config_out_path = os.path.join(output_dir, module_config_name)
            with open(module_config_out_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            saved_paths[module_name] = module_config_out_path
            
        return saved_paths
