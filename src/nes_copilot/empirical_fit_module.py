"""
Empirical Fitting Module for NES Copilot

This module handles fitting trained NPE models to empirical data.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from nes_copilot.module_base import ModuleBase


class EmpiricalFittingModule(ModuleBase):
    """
    Empirical Fitting module for the NES Copilot system.
    
    Handles fitting trained NPE models to empirical data.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the empirical fitting module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        super().__init__(config_manager, data_manager, logging_manager)
        
        # Get module configuration
        self.config = self.config_manager.get_module_config('empirical_fit')
        
        # Get device from master config
        self.device = self.config_manager.get_param('device', 'cpu')
        
        # Import required modules
        from nes_copilot.summary_stats import SummaryStatsModule
        from nes_copilot.npe import CheckpointManager
        
        # Initialize required modules
        self.summary_stats_module = SummaryStatsModule(config_manager, data_manager, logging_manager)
        self.checkpoint_manager = CheckpointManager(self.logger)
        
    def run(self, npe_checkpoint: Optional[str] = None, empirical_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Fit a trained NPE to empirical data.
        
        Args:
            npe_checkpoint: Optional override for the NPE checkpoint path.
            empirical_data: Optional override for the empirical data path.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the results of the empirical fitting.
        """
        self.logger.info("Starting empirical fitting")
        
        # Get NPE checkpoint path
        if npe_checkpoint is None:
            npe_checkpoint = self.config.get('npe_checkpoint_to_use')
            if npe_checkpoint is None:
                # Try to find the latest checkpoint in the NPE output directory
                npe_dir = self.data_manager.get_module_dir('npe')
                checkpoint_dir = os.path.join(npe_dir, 'checkpoint')
                if os.path.exists(checkpoint_dir):
                    npe_checkpoint = checkpoint_dir
                else:
                    raise ValueError("NPE checkpoint not specified and no default found")
        
        # Load NPE checkpoint
        self.logger.info(f"Loading NPE checkpoint from {npe_checkpoint}")
        checkpoint = self.checkpoint_manager.load_checkpoint(npe_checkpoint)
        posterior = checkpoint['posterior']
        param_names = checkpoint['param_names']
        summary_stat_keys = checkpoint['summary_stat_keys']
        
        # Load empirical data if not provided
        if empirical_data is None:
            empirical_data_path = self.data_manager.get_output_path('data_prep', 'empirical_data.csv')
            if os.path.exists(empirical_data_path):
                empirical_data = pd.read_csv(empirical_data_path)
            else:
                raise FileNotFoundError(f"Empirical data not found: {empirical_data_path}")
                
        # Validate inputs
        self.validate_inputs(posterior=posterior, empirical_data=empirical_data)
        
        # Get fitting parameters
        num_posterior_samples = self.config.get('num_posterior_samples', 10000)
        save_full_posterior = self.config.get('save_full_posterior', False)
        
        # Fit to empirical data
        fitting_results = self._fit_to_empirical_data(
            posterior=posterior,
            empirical_data=empirical_data,
            param_names=param_names,
            summary_stat_keys=summary_stat_keys,
            num_posterior_samples=num_posterior_samples,
            save_full_posterior=save_full_posterior
        )
        
        # Save outputs
        output_paths = self.save_outputs(fitting_results)
        
        # Return results
        results = {
            'num_subjects': fitting_results['num_subjects'],
            'num_posterior_samples': num_posterior_samples,
            'output_paths': output_paths
        }
        
        self.logger.info("Empirical fitting completed successfully")
        return results
        
    def validate_inputs(self, posterior, empirical_data, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            posterior: Trained posterior.
            empirical_data: Empirical data DataFrame.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        # Check if posterior is provided
        if posterior is None:
            raise ValueError("Posterior not provided")
            
        # Check if empirical data is provided
        if empirical_data is None or len(empirical_data) == 0:
            raise ValueError("Empirical data not provided or empty")
            
        # Check if empirical data has required columns
        required_columns = ['subject', 'choice', 'rt', 'frame', 'cond']
        for col in required_columns:
            if col not in empirical_data.columns:
                raise ValueError(f"Empirical data missing required column: {col}")
                
        return True
        
    def _fit_to_empirical_data(self, posterior, empirical_data, param_names, summary_stat_keys,
                              num_posterior_samples, save_full_posterior) -> Dict[str, Any]:
        """
        Fit the NPE to empirical data.
        
        Args:
            posterior: Trained posterior.
            empirical_data: Empirical data DataFrame.
            param_names: List of parameter names.
            summary_stat_keys: List of summary statistic keys.
            num_posterior_samples: Number of posterior samples per subject.
            save_full_posterior: Whether to save the full posterior samples.
            
        Returns:
            Dict containing the fitting results.
        """
        # Get unique subjects
        subjects = empirical_data['subject'].unique()
        num_subjects = len(subjects)
        
        # Initialize results
        posterior_means = np.zeros((num_subjects, len(param_names)))
        posterior_medians = np.zeros((num_subjects, len(param_names)))
        posterior_stds = np.zeros((num_subjects, len(param_names)))
        
        # Initialize full posterior samples if needed
        if save_full_posterior:
            all_posterior_samples = {}
        
        # Process each subject
        for i, subject in enumerate(subjects):
            self.logger.info(f"Fitting subject {i+1}/{num_subjects} (ID: {subject})")
            
            # Get subject data
            subject_data = empirical_data[empirical_data['subject'] == subject]
            
            # Calculate summary statistics
            stats_results = self.summary_stats_module.run(subject_data)
            summary_stats = stats_results['summary_stats']
            
            # Convert summary stats dictionary to array
            x_obs = np.array([summary_stats.get(key, np.nan) for key in summary_stat_keys])
            
            # Sample from posterior conditioned on observed summary statistics
            x_obs_tensor = torch.tensor(x_obs, dtype=torch.float32).reshape(1, -1)
            posterior_samples = posterior.sample((num_posterior_samples,), x=x_obs_tensor).numpy()
            
            # Calculate posterior statistics
            posterior_means[i] = np.mean(posterior_samples, axis=0)
            posterior_medians[i] = np.median(posterior_samples, axis=0)
            posterior_stds[i] = np.std(posterior_samples, axis=0)
            
            # Save full posterior samples if needed
            if save_full_posterior:
                all_posterior_samples[subject] = posterior_samples
        
        # Create results dictionary
        results = {
            'subjects': subjects,
            'param_names': param_names,
            'posterior_means': posterior_means,
            'posterior_medians': posterior_medians,
            'posterior_stds': posterior_stds,
            'num_subjects': num_subjects,
            'num_posterior_samples': num_posterior_samples
        }
        
        # Add full posterior samples if saved
        if save_full_posterior:
            results['posterior_samples'] = all_posterior_samples
            
        return results
        
    def save_outputs(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save module outputs to disk using the data manager.
        
        Args:
            results: Results to save.
            
        Returns:
            Dict mapping output names to their paths.
        """
        output_paths = {}
        
        # Create DataFrame for posterior summaries
        posterior_summary_data = []
        
        for i, subject in enumerate(results['subjects']):
            for j, param_name in enumerate(results['param_names']):
                posterior_summary_data.append({
                    'subject': subject,
                    'parameter': param_name,
                    'mean': results['posterior_means'][i, j],
                    'median': results['posterior_medians'][i, j],
                    'std': results['posterior_stds'][i, j]
                })
                
        posterior_summary_df = pd.DataFrame(posterior_summary_data)
        
        # Save posterior summaries
        posterior_summary_path = self.data_manager.get_output_path('empirical_fit', 'posterior_summaries.csv')
        posterior_summary_df.to_csv(posterior_summary_path, index=False)
        output_paths['posterior_summaries'] = posterior_summary_path
        
        # Save full posterior samples if available
        if 'posterior_samples' in results:
            # Create directory for posterior samples
            posterior_samples_dir = self.data_manager.get_output_path('empirical_fit', 'posterior_samples')
            os.makedirs(posterior_samples_dir, exist_ok=True)
            
            # Save posterior samples for each subject
            for subject, samples in results['posterior_samples'].items():
                subject_samples_path = os.path.join(posterior_samples_dir, f"subject_{subject}_samples.npy")
                np.save(subject_samples_path, samples)
                
            output_paths['posterior_samples_dir'] = posterior_samples_dir
            
        # Save metadata
        metadata = {
            'num_subjects': results['num_subjects'],
            'num_posterior_samples': results['num_posterior_samples'],
            'param_names': results['param_names']
        }
        
        metadata_path = self.data_manager.get_output_path('empirical_fit', 'metadata.json')
        self.data_manager.save_json(metadata, 'empirical_fit', 'metadata.json')
        output_paths['metadata'] = metadata_path
        
        return output_paths
