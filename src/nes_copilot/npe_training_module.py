"""
NPE Training Module for NES Copilot with sbi Integration

This module handles training of Neural Posterior Estimation models using sbi.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

import sbi
from sbi.inference import SNPE_C as SNPE
from sbi.utils import BoxUniform, process_prior

from nes_copilot.simulation_module import SimulationModule
from nes_copilot.data_prep_module import DataPrepModule
from nes_copilot.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class NPETrainingModule:
    """
    NPE Training module for the NES Copilot system with sbi integration.
    
    Handles training of Neural Posterior Estimation models using sbi.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the NPE training module.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(self.config.get_output_dir())
        
        # Initialize modules
        self.data_prep = DataPrepModule(config_manager)
        self.simulator = SimulationModule(config_manager)
        
        # Setup checkpointing
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        
        logger.info(f"Initialized NPETrainingModule with output directory: {self.output_dir}")
    
    def train_npe(
        self,
        trial_template: Optional[pd.DataFrame] = None,
        n_simulations: Optional[int] = None,
        resume_from_checkpoint: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a Neural Posterior Estimator (NPE) using sbi.
        
        Args:
            trial_template: Optional trial template DataFrame. If None, will be loaded from config.
            n_simulations: Number of simulations to run. If None, uses value from config.
            resume_from_checkpoint: Whether to resume training from a checkpoint if available.
            
        Returns:
            Tuple of (density_estimator, training_metadata)
        """
        # Load or generate trial template if not provided
        if trial_template is None:
            trial_template = self.data_prep.prepare_trial_template()
        
        # Get number of simulations from config if not provided
        if n_simulations is None:
            n_simulations = self.config.experiment_config['n_train_simulations']
        
        # Setup prior distribution
        prior = self._create_prior()
        
        # Check for existing checkpoint
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if checkpoint is not None:
                logger.info(f"Resuming from checkpoint: {checkpoint['checkpoint_path']}")
                return self._resume_training(checkpoint, trial_template, n_simulations)
        
        # Initialize SNPE
        density_estimator = SNPE(prior=prior, device=self.device)
        
        # Run simulations and train NPE
        logger.info(f"Starting NPE training with {n_simulations} simulations")
        
        # Get simulator function
        simulator = self.simulator.get_simulator(trial_template)
        
        # Run simulations and train NPE in batches to allow for checkpointing
        batch_size = min(1000, n_simulations)  # Process in batches of 1000
        n_batches = (n_simulations + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            current_batch = min(batch_size, n_simulations - batch_idx * batch_size)
            logger.info(f"Running batch {batch_idx + 1}/{n_batches} with {current_batch} simulations")
            
            # Sample parameters from prior
            theta = prior.sample((current_batch,))
            
            # Run simulations
            x = simulator(theta)
            
            # Append simulations to NPE
            density_estimator = density_estimator.append_simulations(
                theta, x, data_device=self.device
            )
            
            # Save checkpoint after each batch
            if (batch_idx + 1) % 5 == 0:  # Save checkpoint every 5 batches
                self._save_checkpoint(density_estimator, batch_idx + 1, n_simulations)
        
        # Train the density estimator
        logger.info("Training density estimator...")
        density_estimator = density_estimator.train()
        
        # Save final checkpoint
        training_metadata = self._save_checkpoint(density_estimator, n_batches, n_simulations)
        
        return density_estimator, training_metadata
    
    def _create_prior(self) -> BoxUniform:
        """Create a BoxUniform prior from the configuration."""
        low, high = self.config.get_prior_bounds()
        return BoxUniform(low=low, high=high, device=self.device)
    
    def _save_checkpoint(
        self, 
        density_estimator: Any, 
        batch_idx: int, 
        total_batches: int
    ) -> Dict[str, Any]:
        """
        Save a training checkpoint.
        
        Args:
            density_estimator: The current density estimator
            batch_idx: Current batch index
            total_batches: Total number of batches
            
        Returns:
            Dictionary with metadata about the checkpoint
        """
        checkpoint_data = {
            'state_dict': density_estimator.state_dict(),
            'batch_idx': batch_idx,
            'total_batches': total_batches,
            'timestamp': datetime.now().isoformat(),
            'config': dict(self.config.config)  # Save a copy of the config
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            f"npe_checkpoint_batch_{batch_idx:04d}.pt"
        )
        
        logger.info(f"Saved checkpoint to {checkpoint_path} (batch {batch_idx}/{total_batches})")
        return checkpoint_data
    
    def _resume_training(
        self, 
        checkpoint: Dict[str, Any],
        trial_template: pd.DataFrame,
        n_simulations: int
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint: Checkpoint data
            trial_template: Trial template DataFrame
            n_simulations: Total number of simulations to run
            
        Returns:
            Tuple of (density_estimator, training_metadata)
        """
        # Initialize SNPE with the same prior
        prior = self._create_prior()
        density_estimator = SNPE(prior=prior, device=self.device)
        
        # Load state dict
        density_estimator.load_state_dict(checkpoint['state_dict'])
        
        # Get number of remaining simulations
        completed_sims = checkpoint.get('batch_idx', 0) * (
            n_simulations // checkpoint.get('total_batches', 1)
        )
        remaining_sims = n_simulations - completed_sims
        
        if remaining_sims <= 0:
            logger.info("Checkpoint already has the requested number of simulations. Returning...")
            return density_estimator, checkpoint
        
        logger.info(f"Resuming training with {remaining_sims} remaining simulations")
        
        # Continue training with remaining simulations
        return self.train_npe(
            trial_template=trial_template,
            n_simulations=remaining_sims,
            resume_from_checkpoint=False  # Don't resume from checkpoint again
        )
        """
        Train an NPE model using the specified configuration.
        
        Args:
            trial_template: Optional override for the trial template.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the results of the NPE training.
        """
        self.logger.info("Starting NPE training")
        
        # Load trial template if not provided
        if trial_template is None:
            trial_template_path = self.data_manager.get_output_path('data_prep', 'trial_template.csv')
            if os.path.exists(trial_template_path):
                trial_template = pd.read_csv(trial_template_path)
            else:
                raise FileNotFoundError(f"Trial template not found: {trial_template_path}")
                
        # Validate inputs
        self.validate_inputs(trial_template=trial_template)
        
        # Define prior
        prior = self._define_prior()
        self.logger.info(f"Defined prior with {len(prior.low)} parameters")
        
        # Generate training data
        theta, x = self._generate_training_data(prior, trial_template)
        self.logger.info(f"Generated {len(theta)} training samples")
        
        # Train NPE
        density_estimator = self._train_npe(theta, x)
        self.logger.info("NPE training completed")
        
        # Save NPE checkpoint
        output_paths = self.save_outputs({
            'density_estimator': density_estimator,
            'prior': prior,
            'summary_stat_keys': self.summary_stats_module.default_stat_keys
        })
        
        # Return results
        results = {
            'num_training_samples': len(theta),
            'num_parameters': len(prior.low),
            'num_summary_stats': x.shape[1],
            'output_paths': output_paths
        }
        
        self.logger.info("NPE training completed successfully")
        return results
        
    def validate_inputs(self, trial_template: pd.DataFrame, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            trial_template: Trial template DataFrame.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        # Check if trial template is provided
        if trial_template is None or len(trial_template) == 0:
            raise ValueError("Trial template not provided or empty")
            
        # Check if trial template has required columns
        required_columns = ['frame', 'cond', 'is_gain_frame', 'time_constrained', 'valence_score', 'norm_category_for_trial']
        for col in required_columns:
            if col not in trial_template.columns:
                raise ValueError(f"Trial template missing required column: {col}")
                
        return True
        
    def _define_prior(self) -> BoxUniform:
        """
        Define the prior distribution for the parameters.
        
        Returns:
            BoxUniform prior distribution.
        """
        # Get prior configuration
        prior_config = self.config.get('prior', {})
        
        # Initialize lists for parameter bounds
        param_names = []
        lower_bounds = []
        upper_bounds = []
        
        # Process each parameter
        for param_name, param_config in prior_config.items():
            # Check if this is a valid parameter configuration
            if not isinstance(param_config, dict) or 'distribution' not in param_config:
                continue
                
            # Currently only uniform distributions are supported
            if param_config['distribution'] != 'uniform':
                self.logger.warning(f"Unsupported distribution for parameter {param_name}: {param_config['distribution']}")
                continue
                
            # Get parameter bounds
            low = param_config.get('low', 0.0)
            high = param_config.get('high', 1.0)
            
            # Add to lists
            param_names.append(param_name)
            lower_bounds.append(low)
            upper_bounds.append(high)
            
        # Create BoxUniform prior
        prior = BoxUniform(
            low=torch.tensor(lower_bounds, dtype=torch.float32),
            high=torch.tensor(upper_bounds, dtype=torch.float32)
        )
        
        # Store parameter names for reference
        self.param_names = param_names
        
        return prior
        
    def _generate_training_data(self, prior: BoxUniform, trial_template: pd.DataFrame) -> tuple:
        """
        Generate training data for NPE.
        
        Args:
            prior: Prior distribution.
            trial_template: Trial template DataFrame.
            
        Returns:
            Tuple of (theta, x) where theta are parameter sets and x are summary statistics.
        """
        # Get number of training simulations
        num_training_sims = self.config.get('num_training_sims', 1000)
        
        # Get fixed parameters for simulation
        fixed_params = self.config_manager.get_module_config('simulator').get('mvnes_agent', {}).get('fixed_params', {})
        
        # Get batch size
        batch_size = self.config.get('batch_size', 100)
        
        # Sample parameters from prior
        theta_raw = prior.sample((num_training_sims,))
        
        # Convert to numpy for easier handling
        theta_np = theta_raw.numpy()
        
        # Initialize arrays for summary statistics
        x_list = []
        
        # Generate simulations in batches
        for i in range(0, num_training_sims, batch_size):
            self.logger.info(f"Generating simulations {i+1}-{min(i+batch_size, num_training_sims)} of {num_training_sims}")
            
            # Get batch of parameters
            batch_end = min(i + batch_size, num_training_sims)
            theta_batch = theta_np[i:batch_end]
            
            # Process each parameter set in the batch
            for j, theta_single in enumerate(theta_batch):
                # Convert parameter array to dictionary
                param_dict = {name: float(value) for name, value in zip(self.param_names, theta_single)}
                
                # Run simulation
                sim_results = self.simulation_module.run(param_dict, trial_template, fixed_params)
                simulated_data = sim_results['simulated_data']
                
                # Calculate summary statistics
                stats_results = self.summary_stats_module.run(simulated_data)
                summary_stats = stats_results['summary_stats']
                
                # Convert summary stats dictionary to array
                stat_keys = self.summary_stats_module.default_stat_keys
                stats_array = np.array([summary_stats.get(key, np.nan) for key in stat_keys])
                
                # Add to list
                x_list.append(stats_array)
                
        # Convert lists to arrays
        x = np.array(x_list)
        
        # Convert to torch tensors
        theta = torch.tensor(theta_np, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        
        return theta, x
        
    def _train_npe(self, theta: torch.Tensor, x: torch.Tensor) -> Any:
        """
        Train the NPE model.
        
        Args:
            theta: Parameter sets.
            x: Summary statistics.
            
        Returns:
            Trained density estimator.
        """
        # Get NPE architecture configuration
        npe_config = self.config.get('npe_architecture', {})
        density_estimator = npe_config.get('density_estimator', 'maf')
        hidden_features = npe_config.get('hidden_features', 50)
        num_transforms = npe_config.get('num_transforms', 5)
        
        # Create SNPE instance
        snpe = SNPE(prior=None)
        
        # Train the model
        density_estimator = snpe.append_simulations(theta, x).train(
            training_batch_size=min(len(theta), 50),
            learning_rate=5e-4,
            max_num_epochs=1000,
            show_train_summary=False,
            device=self.device
        )
        
        return density_estimator
        
    def save_outputs(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save module outputs to disk using the data manager.
        
        Args:
            results: Results to save.
            
        Returns:
            Dict mapping output names to their paths.
        """
        output_paths = {}
        
        # Create checkpoint directory
        checkpoint_dir = self.data_manager.get_output_path('npe', 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save density estimator
        density_estimator = results['density_estimator']
        density_estimator_path = os.path.join(checkpoint_dir, 'density_estimator.pt')
        torch.save(density_estimator.state_dict(), density_estimator_path)
        output_paths['density_estimator'] = density_estimator_path
        
        # Save metadata
        metadata = {
            'sbi_version': sbi.__version__,
            'num_summary_stats': len(results['summary_stat_keys']),
            'summary_stat_keys': results['summary_stat_keys'],
            'param_names': self.param_names,
            'prior_low': results['prior'].low.tolist(),
            'prior_high': results['prior'].high.tolist(),
            'training_args': {
                'num_training_sims': self.config.get('num_training_sims', 1000),
                'batch_size': self.config.get('batch_size', 100),
                'npe_architecture': self.config.get('npe_architecture', {})
            }
        }
        
        metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        self.data_manager.save_json(metadata, 'npe', 'checkpoint/metadata.json')
        output_paths['metadata'] = metadata_path
        
        return output_paths
