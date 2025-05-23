"""
Checkpoint Manager for NES Copilot with sbi Integration

This module handles saving and loading NPE training checkpoints with metadata.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

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
