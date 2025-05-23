"""
Configuration Manager for NES Copilot

This module handles loading, validating, and providing access to configuration parameters
for the NES Copilot system, specifically for the NES model with SBI integration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class PriorConfig:
    """Dataclass to hold prior distribution configuration."""
    low: np.ndarray
    high: np.ndarray
    parameter_names: List[str]
    
    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert bounds to PyTorch tensors."""
        return (
            torch.as_tensor(self.low, dtype=torch.float32),
            torch.as_tensor(self.high, dtype=torch.float32)
        )


class ConfigManager:
    """
    Configuration manager for the NES Copilot system with SBI integration.
    
    Handles loading and validating configuration files, and provides a unified
    interface for accessing configuration parameters.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self.prior = self._create_prior_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and return the configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def _validate_config(self):
        """Validate the configuration structure and values."""
        required_sections = [
            'experiment', 'data', 'model', 'npe', 'summary_stats', 'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section in config: {section}")
        
        # Validate model parameters
        model_params = self.model_parameters
        if not all(isinstance(v, (int, float)) for v in model_params.values()):
            raise ValueError("Model parameters must be numeric values")
    
    def _create_prior_config(self) -> PriorConfig:
        """Create a PriorConfig instance from the model parameters."""
        param_config = self.config['model']['parameters']
        
        # Get parameter names and bounds in a consistent order
        param_names = list(param_config.keys())
        low_bounds = []
        high_bounds = []
        
        for param in param_names:
            bounds = param_config[param]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"Parameter {param} must have [low, high] bounds")
                
            low_bounds.append(float(bounds[0]))
            high_bounds.append(float(bounds[1]))
        
        return PriorConfig(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            parameter_names=param_names
        )
    
    @property
    def experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.config['experiment']
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config['data']
    
    @property
    def model_parameters(self) -> Dict[str, List[float]]:
        """Get model parameters configuration."""
        return self.config['model']['parameters']
    
    @property
    def fixed_parameters(self) -> Dict[str, float]:
        """Get fixed model parameters."""
        return self.config['model'].get('fixed_parameters', {})
    
    @property
    def npe_config(self) -> Dict[str, Any]:
        """Get NPE training and inference configuration."""
        return self.config['npe']
    
    @property
    def summary_stats_config(self) -> Dict[str, Any]:
        """Get summary statistics configuration."""
        return self.config['summary_stats']
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config['logging']
    
    def get_output_dir(self) -> Path:
        """Get the output directory, creating it if it doesn't exist."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_run_id(self) -> str:
        """Get the run ID, generating one if not specified."""
        if 'run_id' in self.config and self.config['run_id']:
            return self.config['run_id']
        
        # Generate a run ID based on timestamp and experiment name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_config['name']}_{timestamp}"
    
    def get_prior_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the prior bounds as PyTorch tensors."""
        return self.prior.to_tensor()
    
    def get_parameter_names(self) -> List[str]:
        """Get the list of parameter names in consistent order."""
        return self.prior.parameter_names
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
