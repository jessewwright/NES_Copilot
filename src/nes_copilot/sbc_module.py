"""
SBC (Simulation-Based Calibration) Module for NES Copilot with sbi Integration

This module implements Simulation-Based Calibration for trained NPE models
to validate posterior estimates.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm

from scipy import stats
from sbi.inference import SNPE_C as SNPE
from sbi.utils import BoxUniform

from nes_copilot.simulation_module import SimulationModule
from nes_copilot.data_prep_module import DataPrepModule
from nes_copilot.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class SBCModule:
    """
    SBC module for the NES Copilot system with sbi integration.
    
    Implements Simulation-Based Calibration to validate posterior estimates
    from trained NPE models.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the SBC module.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(self.config.get_output_dir()) / 'sbc_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize required modules
        self.data_prep = DataPrepModule(config_manager)
        self.simulator = SimulationModule(config_manager)
        
        logger.info(f"Initialized SBCModule with output directory: {self.output_dir}")
    
    def run_sbc(
        self,
        npe,
        n_simulations: int = 100,
        n_posterior_samples: int = 1000,
        n_workers: int = 4,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run Simulation-Based Calibration for a trained NPE model.
        
        Args:
            npe: Trained NPE model.
            n_simulations: Number of SBC simulations to run.
            n_posterior_samples: Number of posterior samples to draw per simulation.
            n_workers: Number of parallel workers for simulation.
            save_results: Whether to save the results to disk.
            
        Returns:
            Dictionary containing SBC results and diagnostics.
        """
        logger.info(f"Starting SBC with {n_simulations} simulations")
        
        # Prepare trial template
        trial_template = self.data_prep.prepare_trial_template()
        
        # Get prior distribution
        prior = self._create_prior()
        
        # Generate ground truth parameters and data
        thetas = prior.sample((n_simulations,))
        x_obs = self.simulator.run_batch_simulations(
            thetas.numpy(),
            trial_template,
            n_workers=n_workers
        )
        
        # Calculate ranks for each parameter
        ranks = self._calculate_ranks(npe, prior, thetas, x_obs, n_posterior_samples)
        
        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(ranks, thetas.shape[1])
        
        # Save results if requested
        if save_results:
            results = {
                'thetas': thetas.numpy(),
                'x_obs': x_obs.numpy() if torch.is_tensor(x_obs) else x_obs,
                'ranks': ranks.numpy(),
                'diagnostics': diagnostics,
                'config': dict(self.config.config)
            }
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f'sbc_results_{timestamp}.npz'
            
            np.savez_compressed(output_path, **results)
            logger.info(f"Saved SBC results to {output_path}")
        
        return {
            'thetas': thetas,
            'x_obs': x_obs,
            'ranks': ranks,
            'diagnostics': diagnostics
        }
    
    def _calculate_ranks(
        self,
        npe,
        prior,
        thetas: torch.Tensor,
        x_obs: torch.Tensor,
        n_posterior_samples: int
    ) -> torch.Tensor:
        """
        Calculate rank statistics for SBC.
        
        Args:
            npe: Trained NPE model.
            prior: Prior distribution.
            thetas: Ground truth parameters (n_simulations, n_params).
            x_obs: Observed data (n_simulations, n_features).
            n_posterior_samples: Number of posterior samples to draw.
            
        Returns:
            Tensor of ranks with shape (n_simulations, n_params).
        """
        n_simulations, n_params = thetas.shape
        ranks = torch.zeros_like(thetas)
        
        logger.info(f"Calculating ranks for {n_simulations} simulations...")
        
        for i in tqdm(range(n_simulations), desc="Running SBC"):
            # Sample from posterior
            posterior = npe.build_posterior()
            theta_samples = posterior.sample(
                (n_posterior_samples,),
                x=x_obs[i].unsqueeze(0).to(self.device),
                show_progress_bars=False
            )
            
            # Calculate ranks
            for j in range(n_params):
                # Count how many samples are below the true value
                rank = (theta_samples[:, j] < thetas[i, j]).sum().item()
                ranks[i, j] = rank
        
        return ranks
    
    def _calculate_diagnostics(
        self,
        ranks: torch.Tensor,
        n_params: int
    ) -> Dict[str, Any]:
        """
        Calculate SBC diagnostics.
        
        Args:
            ranks: Rank statistics from SBC.
            n_params: Number of parameters.
            
        Returns:
            Dictionary containing diagnostic metrics.
        """
        n_simulations = len(ranks)
        n_bins = 10
        
        # Calculate ECDF for each parameter
        ecdfs = []
        for j in range(n_params):
            ecdf = np.sort(ranks[:, j].numpy() / n_simulations)
            ecdfs.append(ecdf)
        
        # Calculate KS test statistics
        ks_stats = []
        for j in range(n_params):
            stat = stats.kstest(ecdfs[j], 'uniform').statistic
            ks_stats.append(stat)
        
        # Calculate histogram-based diagnostics
        histograms = []
        bin_edges = np.linspace(0, n_simulations, n_bins + 1)
        
        for j in range(n_params):
            hist, _ = np.histogram(ranks[:, j].numpy(), bins=bin_edges, density=True)
            histograms.append(hist)
        
        # Calculate expected bin counts under uniform distribution
        expected = np.ones(n_bins) / n_bins
        
        # Calculate chi-squared statistics
        chi2_stats = []
        for hist in histograms:
            chi2 = np.sum((hist - expected) ** 2 / expected) * n_simulations / n_bins
            chi2_stats.append(chi2)
        
        return {
            'ks_stats': ks_stats,
            'chi2_stats': chi2_stats,
            'histograms': histograms,
            'bin_edges': bin_edges,
            'expected_hist': expected
        }
    
    def plot_sbc_results(
        self,
        ranks: torch.Tensor,
        param_names: List[str],
        save_path: Optional[Path] = None
    ) -> Dict[str, plt.Figure]:
        """
        Plot SBC results.
        
        Args:
            ranks: Rank statistics from SBC.
            param_names: List of parameter names.
            save_path: Optional path to save the plots.
            
        Returns:
            Dictionary of matplotlib figures.
        """
        n_params = len(param_names)
        n_bins = 20
        
        # Create figures
        fig_hist, axs_hist = plt.subplots(
            n_params, 1,
            figsize=(10, 3 * n_params),
            sharex=True,
            tight_layout=True
        )
        
        if n_params == 1:
            axs_hist = [axs_hist]
        
        # Plot histograms
        for i, (ax, name) in enumerate(zip(axs_hist, param_names)):
            ax.hist(
                ranks[:, i].numpy(),
                bins=n_bins,
                alpha=0.7,
                density=True,
                label=f'p = {stats.kstest(ranks[:, i].numpy() / len(ranks), "uniform").pvalue:.3f}'
            )
            ax.axhline(1.0 / n_bins, color='r', linestyle='--', label='Expected')
            ax.set_title(f"{name} (KS p-value: {stats.kstest(ranks[:, i].numpy() / len(ranches), 'uniform').pvalue:.3f})")
            ax.legend()
        
        fig_hist.suptitle("SBC Rank Histograms")
        
        # Save figures if path is provided
        if save_path is not None:
            fig_hist.savefig(save_path / 'sbc_histograms.png', bbox_inches='tight')
            plt.close(fig_hist)
        
        return {
            'histograms': fig_hist
        }
    
    def _create_prior(self) -> BoxUniform:
        """Create a BoxUniform prior from the configuration."""
        low, high = self.config.get_prior_bounds()
        return BoxUniform(low=low, high=high, device=self.device)
        
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
