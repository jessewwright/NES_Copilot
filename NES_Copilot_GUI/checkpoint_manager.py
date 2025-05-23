"""
Checkpoint Manager for NES Copilot

This module handles saving and loading NPE checkpoints.
"""

import os
import torch
import json
from typing import Dict, Any, Optional, Union

import sbi
from sbi.inference import SNPE
from sbi.utils import BoxUniform


class CheckpointManager:
    """
    Checkpoint manager for the NES Copilot system.
    
    Handles saving and loading NPE checkpoints.
    """
    
    def __init__(self, logger):
        """
        Initialize the checkpoint manager.
        
        Args:
            logger: Logger instance.
        """
        self.logger = logger
        
    def save_checkpoint(self, density_estimator, prior, summary_stat_keys, param_names, 
                        checkpoint_dir, training_args=None):
        """
        Save an NPE checkpoint.
        
        Args:
            density_estimator: Trained density estimator.
            prior: Prior distribution.
            summary_stat_keys: List of summary statistic keys.
            param_names: List of parameter names.
            checkpoint_dir: Directory to save the checkpoint.
            training_args: Optional training arguments.
            
        Returns:
            Dict mapping output names to their paths.
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save density estimator
        density_estimator_path = os.path.join(checkpoint_dir, 'density_estimator.pt')
        torch.save(density_estimator.state_dict(), density_estimator_path)
        
        # Save metadata
        metadata = {
            'sbi_version': sbi.__version__,
            'num_summary_stats': len(summary_stat_keys),
            'summary_stat_keys': summary_stat_keys,
            'param_names': param_names,
            'prior_low': prior.low.tolist(),
            'prior_high': prior.high.tolist(),
            'training_args': training_args or {}
        }
        
        metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Saved NPE checkpoint to {checkpoint_dir}")
        
        return {
            'density_estimator': density_estimator_path,
            'metadata': metadata_path
        }
        
    def load_checkpoint(self, checkpoint_dir):
        """
        Load an NPE checkpoint.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint.
            
        Returns:
            Dict containing the loaded checkpoint components.
        """
        # Check if checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        # Load metadata
        metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Create prior
        prior_low = torch.tensor(metadata['prior_low'], dtype=torch.float32)
        prior_high = torch.tensor(metadata['prior_high'], dtype=torch.float32)
        prior = BoxUniform(low=prior_low, high=prior_high)
        
        # Load density estimator
        density_estimator_path = os.path.join(checkpoint_dir, 'density_estimator.pt')
        if not os.path.exists(density_estimator_path):
            raise FileNotFoundError(f"Density estimator file not found: {density_estimator_path}")
            
        # Create SNPE instance
        snpe = SNPE(prior=prior)
        
        # Build posterior
        posterior = snpe.build_posterior()
        
        # Load state dict
        density_estimator_state_dict = torch.load(density_estimator_path)
        posterior.net.load_state_dict(density_estimator_state_dict)
        
        self.logger.info(f"Loaded NPE checkpoint from {checkpoint_dir}")
        
        return {
            'posterior': posterior,
            'prior': prior,
            'metadata': metadata,
            'param_names': metadata['param_names'],
            'summary_stat_keys': metadata['summary_stat_keys']
        }
