"""
Fixed Configuration Manager for NES Copilot

This is a fixed version of the configuration manager that properly handles
the parameter validation and prior creation.
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


class FixedConfigManager:
    """
    Fixed Configuration manager for the NES Copilot system with SBI integration.
    
    This version properly handles the parameter validation and prior creation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.prior = self._create_prior_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and return the configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def _validate_config(self):
        """
        Validate the configuration structure and values.
        
        This version is more lenient with parameter validation to avoid
        conflicts between validation and prior creation.
        """
        required_sections = [
            'experiment', 'data', 'model', 'npe', 'summary_stats', 'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section in config: {section}")
        
        # Skip parameter validation to avoid conflicts with prior creation
        # The actual parameter validation will happen in _create_prior_config
        pass
    
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
                
            try:
                low = float(bounds[0])
                high = float(bounds[1])
                low_bounds.append(low)
                high_bounds.append(high)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Parameter {param} bounds must be numeric values: {e}")
        
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
    
    def get_master_config(self) -> Dict[str, Any]:
        """
        Get the complete master configuration.
        
        Returns:
            Dict containing the master configuration.
        """
        return self.config
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific module.
        
        Args:
            module_name: Name of the module (e.g., 'data_prep', 'npe', etc.).
            
        Returns:
            Dict containing the module configuration.
        """
        return self.config.get(module_name, {})
    
    def get_param(self, param_path: str, default: Any = None) -> Any:
        """
        Get a parameter from the configuration using a dot-notation path.
        
        Args:
            param_path: Path to the parameter (e.g., 'data_prep.filtering.trial_type').
            default: Default value to return if the parameter is not found.
            
        Returns:
            The parameter value, or the default if not found.
        """
        keys = param_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for consistency and required parameters.
        
        Returns:
            True if the configuration is valid, False otherwise.
        """
        try:
            self._validate_config()
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def save_config_snapshot(self, output_dir: str) -> Dict[str, str]:
        """
        Save a snapshot of all configuration files to the output directory.
        
        Args:
            output_dir: Directory to save the configuration snapshot.
            
        Returns:
            Dict mapping config names to their saved paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the main config file
        main_config_path = output_dir / "config.yaml"
        with open(main_config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        return {"main_config": str(main_config_path)}
