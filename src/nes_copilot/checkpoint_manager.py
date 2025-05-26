"""
Checkpoint Manager for NES Copilot with sbi Integration

This module handles saving and loading NPE training checkpoints with metadata.
"""

import os
import json
import torch
import logging
import sbi
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from sbi.inference import SNPE_C as SNPE
from sbi.utils import BoxUniform, process_prior
from sbi.utils.torchutils import atleast_2d_float32_tensor
from torch.distributions import Distribution

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Checkpoint manager for the NES Copilot system with sbi integration.
    
    Handles saving and loading of NPE training checkpoints with metadata.
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory where checkpoints will be stored.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized CheckpointManager with directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:
        """
        Save a checkpoint to disk.
        
        Args:
            checkpoint_data: Dictionary containing the checkpoint data.
            filename: Optional filename. If None, generates one with a timestamp.
            
        Returns:
            Path to the saved checkpoint file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save the checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save metadata as JSON for easier inspection
        metadata = {
            k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
            for k, v in checkpoint_data.items() 
            if k != 'state_dict'
        }
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return checkpoint_path
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint.
        
        Returns:
            Dictionary with checkpoint data, or None if no checkpoints exist.
        """
        checkpoints = sorted(self.checkpoint_dir.glob('*.pt'))
        if not checkpoints:
            return None
            
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        logger.info(f"Loading latest checkpoint: {latest_checkpoint}")
        
        try:
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
            checkpoint_data['checkpoint_path'] = latest_checkpoint
            return checkpoint_data
        except Exception as e:
            logger.error(f"Error loading checkpoint {latest_checkpoint}: {e}")
            return None
    
    def get_all_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Get all available checkpoints, sorted by modification time (newest first).
        
        Returns:
            List of checkpoint data dictionaries.
        """
        checkpoints = []
        for checkpoint_file in sorted(
            self.checkpoint_dir.glob('*.pt'),
            key=os.path.getmtime,
            reverse=True
        ):
            try:
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                checkpoint_data['checkpoint_path'] = checkpoint_file
                checkpoints.append(checkpoint_data)
            except Exception as e:
                logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Delete old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of most recent checkpoints to keep.
        """
        checkpoints = sorted(self.checkpoint_dir.glob('*.pt'), key=os.path.getmtime)
        
        if len(checkpoints) <= keep_last_n:
            return
            
        for checkpoint_file in checkpoints[:-keep_last_n]:
            try:
                # Also remove corresponding JSON metadata if it exists
                json_file = checkpoint_file.with_suffix('.json')
                if json_file.exists():
                    os.remove(json_file)
                os.remove(checkpoint_file)
                logger.info(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.error(f"Error removing checkpoint {checkpoint_file}: {e}")
        
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
        
    def load_npe_posterior_object_from_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        device: str = 'cpu',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load a trained NPE (Neural Posterior Estimation) model from a checkpoint directory.
        
        This function loads the necessary components to perform inference with a trained
        NPE model, including the posterior distribution, prior, and metadata.
        
        Args:
            checkpoint_dir: Path to the directory containing the checkpoint files.
            device: Device to load the model onto ('cpu' or 'cuda').
            **kwargs: Additional arguments to pass to the posterior's `set_default_x` method.
            
        Returns:
            A dictionary containing:
                - 'posterior': The loaded NPE posterior object
                - 'prior': The prior distribution
                - 'metadata': Dictionary of metadata from training
                - 'param_names': List of parameter names
                - 'summary_stat_keys': List of summary statistic keys
                
        Raises:
            FileNotFoundError: If required checkpoint files are missing
            RuntimeError: If there's an error loading the model
        """
        logger.info(f"Loading NPE posterior from checkpoint: {checkpoint_dir}")
        checkpoint_dir = Path(checkpoint_dir)
        
        # Check if checkpoint directory exists
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        # Load metadata
        metadata_path = checkpoint_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            logger.info(f"Loaded metadata from {metadata_path}")
            
            # Extract parameter information
            param_names = metadata.get('param_names', [f'param_{i}' for i in range(len(metadata['prior_low']))])
            prior_low = torch.tensor(metadata['prior_low'], dtype=torch.float32, device=device)
            prior_high = torch.tensor(metadata['prior_high'], dtype=torch.float32, device=device)
            
            # Create prior distribution
            prior = BoxUniform(low=prior_low, high=prior_high, device=device)
            logger.info(f"Created prior with bounds: {prior_low} to {prior_high}")
            
            # Load density estimator state dict
            density_estimator_path = checkpoint_dir / 'density_estimator.pt'
            if not density_estimator_path.exists():
                # Try alternative naming convention
                density_estimator_path = next(checkpoint_dir.glob('*density_estimator*.pt'), None)
                if density_estimator_path is None:
                    raise FileNotFoundError(
                        f"Density estimator file not found in {checkpoint_dir}. "
                        "Expected a file named 'density_estimator.pt' or similar."
                    )
            
            # Initialize SNPE
            logger.info("Initializing SNPE with prior...")
            snpe = SNPE(prior=prior, device=device)
            
            # Load state dict
            logger.info(f"Loading density estimator from {density_estimator_path}")
            state_dict = torch.load(density_estimator_path, map_location=device)
            
            # Build posterior
            logger.info("Building posterior...")
            posterior = snpe.build_posterior(
                prior=prior,
                sample_with_mcmc=kwargs.pop('sample_with_mcmc', False),
                mcmc_method=kwargs.pop('mcmc_method', 'slice_np'),
                mcmc_parameters=kwargs.pop('mcmc_parameters', {})
            )
            
            # Load state dict into the posterior's neural network
            if hasattr(posterior, 'net'):
                posterior.net.load_state_dict(state_dict)
                posterior.net.to(device)
            elif hasattr(posterior, '_posterior'):
                # Handle different sbi versions
                if hasattr(posterior._posterior, 'net'):
                    posterior._posterior.net.load_state_dict(state_dict)
                    posterior._posterior.net.to(device)
                else:
                    # Try direct loading for older sbi versions
                    posterior.load_state_dict(state_dict)
            else:
                # Last resort: try direct loading
                posterior.load_state_dict(state_dict)
            
            # Set device for the posterior
            if hasattr(posterior, 'set_default_x'):
                posterior.set_default_x(None, **kwargs)
            
            logger.info("Successfully loaded NPE posterior")
            
            # Return all relevant components
            return {
                'posterior': posterior,
                'prior': prior,
                'metadata': metadata,
                'param_names': param_names,
                'summary_stat_keys': metadata.get('summary_stat_keys', [])
            }
            
        except Exception as e:
            error_msg = f"Error loading NPE posterior from {checkpoint_dir}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
