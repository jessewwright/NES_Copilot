"""
SBC Module for NES Copilot

This module handles Simulation-Based Calibration for trained NPE models.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, List, Optional, Union

from nes_copilot.module_base import ModuleBase


class SBCModule(ModuleBase):
    """
    SBC module for the NES Copilot system.
    
    Handles Simulation-Based Calibration for trained NPE models.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the SBC module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        super().__init__(config_manager, data_manager, logging_manager)
        
        # Get module configuration
        self.config = self.config_manager.get_module_config('sbc')
        
        # Get device from master config
        self.device = self.config_manager.get_param('device', 'cpu')
        
        # Import required modules
        from nes_copilot.simulation import SimulationModule
        from nes_copilot.summary_stats import SummaryStatsModule
        from nes_copilot.npe import CheckpointManager
        
        # Initialize required modules
        self.simulation_module = SimulationModule(config_manager, data_manager, logging_manager)
        self.summary_stats_module = SummaryStatsModule(config_manager, data_manager, logging_manager)
        self.checkpoint_manager = CheckpointManager(self.logger)
        
    def run(self, npe_checkpoint: Optional[str] = None, trial_template: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform Simulation-Based Calibration on a trained NPE.
        
        Args:
            npe_checkpoint: Optional override for the NPE checkpoint path.
            trial_template: Optional override for the trial template.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the results of the SBC.
        """
        self.logger.info("Starting SBC")
        
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
        prior = checkpoint['prior']
        param_names = checkpoint['param_names']
        summary_stat_keys = checkpoint['summary_stat_keys']
        
        # Load trial template if not provided
        if trial_template is None:
            trial_template_path = self.data_manager.get_output_path('data_prep', 'trial_template.csv')
            if os.path.exists(trial_template_path):
                trial_template = pd.read_csv(trial_template_path)
            else:
                raise FileNotFoundError(f"Trial template not found: {trial_template_path}")
                
        # Validate inputs
        self.validate_inputs(posterior=posterior, prior=prior, trial_template=trial_template)
        
        # Get SBC parameters
        num_sbc_datasets = self.config.get('num_sbc_datasets', 100)
        num_posterior_samples = self.config.get('num_posterior_samples', 1000)
        
        # Get fixed parameters for simulation
        fixed_params = self.config_manager.get_module_config('simulator').get('mvnes_agent', {}).get('fixed_params', {})
        
        # Run SBC
        sbc_results = self._run_sbc(
            posterior=posterior,
            prior=prior,
            param_names=param_names,
            summary_stat_keys=summary_stat_keys,
            trial_template=trial_template,
            fixed_params=fixed_params,
            num_sbc_datasets=num_sbc_datasets,
            num_posterior_samples=num_posterior_samples
        )
        
        # Generate SBC plots
        plot_paths = self._generate_sbc_plots(sbc_results, param_names)
        
        # Save outputs
        output_paths = self.save_outputs({
            'sbc_results': sbc_results,
            'plot_paths': plot_paths
        })
        
        # Return results
        results = {
            'num_sbc_datasets': num_sbc_datasets,
            'num_posterior_samples': num_posterior_samples,
            'ks_test_results': sbc_results['ks_test_results'],
            'output_paths': output_paths
        }
        
        self.logger.info("SBC completed successfully")
        return results
        
    def validate_inputs(self, posterior, prior, trial_template, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            posterior: Trained posterior.
            prior: Prior distribution.
            trial_template: Trial template DataFrame.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        # Check if posterior is provided
        if posterior is None:
            raise ValueError("Posterior not provided")
            
        # Check if prior is provided
        if prior is None:
            raise ValueError("Prior not provided")
            
        # Check if trial template is provided
        if trial_template is None or len(trial_template) == 0:
            raise ValueError("Trial template not provided or empty")
            
        # Check if trial template has required columns
        required_columns = ['frame', 'cond', 'is_gain_frame', 'time_constrained', 'valence_score', 'norm_category_for_trial']
        for col in required_columns:
            if col not in trial_template.columns:
                raise ValueError(f"Trial template missing required column: {col}")
                
        return True
        
    def _run_sbc(self, posterior, prior, param_names, summary_stat_keys, trial_template, 
                fixed_params, num_sbc_datasets, num_posterior_samples) -> Dict[str, Any]:
        """
        Run Simulation-Based Calibration.
        
        Args:
            posterior: Trained posterior.
            prior: Prior distribution.
            param_names: List of parameter names.
            summary_stat_keys: List of summary statistic keys.
            trial_template: Trial template DataFrame.
            fixed_params: Fixed parameters for simulation.
            num_sbc_datasets: Number of SBC datasets to generate.
            num_posterior_samples: Number of posterior samples per SBC dataset.
            
        Returns:
            Dict containing the SBC results.
        """
        # Initialize arrays for ranks
        num_params = len(param_names)
        ranks = np.zeros((num_sbc_datasets, num_params), dtype=int)
        
        # Run SBC for each dataset
        for i in range(num_sbc_datasets):
            self.logger.info(f"Running SBC dataset {i+1}/{num_sbc_datasets}")
            
            # Sample ground truth parameters from prior
            theta_gt = prior.sample((1,)).numpy()[0]
            
            # Convert to dictionary
            theta_gt_dict = {name: float(value) for name, value in zip(param_names, theta_gt)}
            
            # Run simulation with ground truth parameters
            sim_results = self.simulation_module.run(theta_gt_dict, trial_template, fixed_params)
            simulated_data = sim_results['simulated_data']
            
            # Calculate summary statistics
            stats_results = self.summary_stats_module.run(simulated_data)
            summary_stats = stats_results['summary_stats']
            
            # Convert summary stats dictionary to array
            x_obs = np.array([summary_stats.get(key, np.nan) for key in summary_stat_keys])
            
            # Sample from posterior conditioned on observed summary statistics
            x_obs_tensor = torch.tensor(x_obs, dtype=torch.float32).reshape(1, -1)
            posterior_samples = posterior.sample((num_posterior_samples,), x=x_obs_tensor).numpy()
            
            # Calculate ranks
            for j in range(num_params):
                rank = np.sum(posterior_samples[:, j] < theta_gt[j])
                ranks[i, j] = rank
                
        # Calculate KS test results
        ks_test_results = {}
        for j, param_name in enumerate(param_names):
            # Normalize ranks to [0, 1]
            normalized_ranks = ranks[:, j] / num_posterior_samples
            
            # Perform KS test against uniform distribution
            ks_statistic, p_value = stats.kstest(normalized_ranks, 'uniform')
            
            ks_test_results[param_name] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'is_uniform': p_value > 0.05
            }
            
        # Return results
        return {
            'ranks': ranks,
            'normalized_ranks': ranks / num_posterior_samples,
            'ks_test_results': ks_test_results,
            'num_sbc_datasets': num_sbc_datasets,
            'num_posterior_samples': num_posterior_samples,
            'param_names': param_names
        }
        
    def _generate_sbc_plots(self, sbc_results, param_names) -> Dict[str, str]:
        """
        Generate SBC diagnostic plots.
        
        Args:
            sbc_results: SBC results.
            param_names: List of parameter names.
            
        Returns:
            Dict mapping plot names to their paths.
        """
        plot_paths = {}
        
        # Create plots directory
        plots_dir = self.data_manager.get_output_path('sbc', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate rank histograms
        ranks = sbc_results['ranks']
        num_posterior_samples = sbc_results['num_posterior_samples']
        
        for j, param_name in enumerate(param_names):
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot histogram
            plt.hist(ranks[:, j], bins=20, alpha=0.7, color='blue', edgecolor='black')
            
            # Add uniform reference line
            expected_count = len(ranks) / 20  # Expected count per bin for uniform distribution
            plt.axhline(y=expected_count, color='red', linestyle='--', label='Expected (Uniform)')
            
            # Add KS test results
            ks_test = sbc_results['ks_test_results'][param_name]
            plt.title(f"SBC Rank Histogram for {param_name}\nKS p-value: {ks_test['p_value']:.4f}")
            
            # Add labels
            plt.xlabel('Rank')
            plt.ylabel('Count')
            plt.legend()
            
            # Save figure
            plot_path = os.path.join(plots_dir, f"rank_histogram_{param_name}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths[f"rank_histogram_{param_name}"] = plot_path
            
        # Generate ECDF plots
        normalized_ranks = sbc_results['normalized_ranks']
        
        for j, param_name in enumerate(param_names):
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Sort ranks for ECDF
            sorted_ranks = np.sort(normalized_ranks[:, j])
            
            # Calculate ECDF
            ecdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
            
            # Plot ECDF
            plt.plot(sorted_ranks, ecdf, '-', color='blue', label='Empirical CDF')
            
            # Add uniform reference line
            plt.plot([0, 1], [0, 1], '--', color='red', label='Uniform CDF')
            
            # Add KS test results
            ks_test = sbc_results['ks_test_results'][param_name]
            plt.title(f"SBC ECDF for {param_name}\nKS p-value: {ks_test['p_value']:.4f}")
            
            # Add labels
            plt.xlabel('Normalized Rank')
            plt.ylabel('Cumulative Probability')
            plt.legend()
            
            # Save figure
            plot_path = os.path.join(plots_dir, f"ecdf_{param_name}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths[f"ecdf_{param_name}"] = plot_path
            
        # Generate summary plot
        plt.figure(figsize=(12, 8))
        
        # Create bar chart of p-values
        p_values = [sbc_results['ks_test_results'][param]['p_value'] for param in param_names]
        plt.bar(param_names, p_values, alpha=0.7, color='blue', edgecolor='black')
        
        # Add significance threshold
        plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (p=0.05)')
        
        # Add labels
        plt.title("SBC KS Test p-values")
        plt.xlabel('Parameter')
        plt.ylabel('p-value')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Save figure
        plot_path = os.path.join(plots_dir, "ks_test_summary.png")
        plt.savefig(plot_path)
        plt.close()
        
        plot_paths["ks_test_summary"] = plot_path
        
        return plot_paths
        
    def save_outputs(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save module outputs to disk using the data manager.
        
        Args:
            results: Results to save.
            
        Returns:
            Dict mapping output names to their paths.
        """
        output_paths = {}
        
        # Save SBC results
        sbc_results = results['sbc_results']
        
        # Save ranks
        ranks_path = self.data_manager.get_output_path('sbc', 'ranks.csv')
        ranks_df = pd.DataFrame(
            sbc_results['ranks'],
            columns=[f"rank_{param}" for param in sbc_results['param_names']]
        )
        ranks_df.to_csv(ranks_path, index=False)
        output_paths['ranks'] = ranks_path
        
        # Save KS test results
        ks_test_path = self.data_manager.get_output_path('sbc', 'ks_test_results.json')
        self.data_manager.save_json(sbc_results['ks_test_results'], 'sbc', 'ks_test_results.json')
        output_paths['ks_test_results'] = ks_test_path
        
        # Add plot paths
        for name, path in results['plot_paths'].items():
            output_paths[name] = path
            
        return output_paths
