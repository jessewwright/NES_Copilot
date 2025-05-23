"""
PPC Module for NES Copilot

This module handles Posterior Predictive Checks for fitted models.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Union
import multiprocessing
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from nes_copilot.module_base import ModuleBase


class PPCModule(ModuleBase):
    """
    PPC module for the NES Copilot system.
    
    Handles Posterior Predictive Checks for fitted models.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the PPC module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        super().__init__(config_manager, data_manager, logging_manager)
        
        # Get module configuration
        self.config = self.config_manager.get_module_config('ppc')
        
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
        
    def run(self, npe_checkpoint: Optional[str] = None, fitting_results: Optional[str] = None, 
            empirical_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform Posterior Predictive Checks.
        
        Args:
            npe_checkpoint: Optional override for the NPE checkpoint path.
            fitting_results: Optional override for the fitting results path.
            empirical_data: Optional override for the empirical data path.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the results of the PPC.
        """
        self.logger.info("Starting PPC")
        
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
        param_names = checkpoint['param_names']
        summary_stat_keys = checkpoint['summary_stat_keys']
        
        # Get fitting results path
        if fitting_results is None:
            fitting_results = self.config.get('empirical_fit_results_file')
            if fitting_results is None:
                # Try to find the latest fitting results
                fitting_results_path = self.data_manager.get_output_path('empirical_fit', 'posterior_summaries.csv')
                if os.path.exists(fitting_results_path):
                    fitting_results = fitting_results_path
                else:
                    raise ValueError("Fitting results not specified and no default found")
                    
        # Load fitting results
        self.logger.info(f"Loading fitting results from {fitting_results}")
        fitting_df = pd.read_csv(fitting_results)
        
        # Load empirical data if not provided
        if empirical_data is None:
            empirical_data_path = self.data_manager.get_output_path('data_prep', 'empirical_data.csv')
            if os.path.exists(empirical_data_path):
                empirical_data = pd.read_csv(empirical_data_path)
            else:
                raise FileNotFoundError(f"Empirical data not found: {empirical_data_path}")
                
        # Validate inputs
        self.validate_inputs(fitting_df=fitting_df, empirical_data=empirical_data)
        
        # Get PPC parameters
        num_ppc_simulations = self.config.get('num_ppc_simulations', 100)
        num_posterior_draws = self.config.get('num_posterior_draws', 100)
        timeout_seconds = self.config.get('timeout_seconds', 300)
        
        # Get fixed parameters for simulation
        fixed_params = self.config_manager.get_module_config('simulator').get('mvnes_agent', {}).get('fixed_params', {})
        
        # Run PPC
        ppc_results = self._run_ppc(
            fitting_df=fitting_df,
            empirical_data=empirical_data,
            param_names=param_names,
            summary_stat_keys=summary_stat_keys,
            fixed_params=fixed_params,
            num_ppc_simulations=num_ppc_simulations,
            num_posterior_draws=num_posterior_draws,
            timeout_seconds=timeout_seconds
        )
        
        # Generate PPC plots
        plot_paths = self._generate_ppc_plots(ppc_results, summary_stat_keys)
        
        # Save outputs
        output_paths = self.save_outputs({
            'ppc_results': ppc_results,
            'plot_paths': plot_paths
        })
        
        # Return results
        results = {
            'num_subjects': ppc_results['num_subjects'],
            'num_ppc_simulations': num_ppc_simulations,
            'num_posterior_draws': num_posterior_draws,
            'coverage_summary': ppc_results['coverage_summary'],
            'output_paths': output_paths
        }
        
        self.logger.info("PPC completed successfully")
        return results
        
    def validate_inputs(self, fitting_df: pd.DataFrame, empirical_data: pd.DataFrame, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            fitting_df: Fitting results DataFrame.
            empirical_data: Empirical data DataFrame.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        # Check if fitting results are provided
        if fitting_df is None or len(fitting_df) == 0:
            raise ValueError("Fitting results not provided or empty")
            
        # Check if fitting results have required columns
        required_columns = ['subject', 'parameter', 'mean', 'median', 'std']
        for col in required_columns:
            if col not in fitting_df.columns:
                raise ValueError(f"Fitting results missing required column: {col}")
                
        # Check if empirical data is provided
        if empirical_data is None or len(empirical_data) == 0:
            raise ValueError("Empirical data not provided or empty")
            
        # Check if empirical data has required columns
        required_columns = ['subject', 'choice', 'rt', 'frame', 'cond']
        for col in required_columns:
            if col not in empirical_data.columns:
                raise ValueError(f"Empirical data missing required column: {col}")
                
        return True
        
    def _run_ppc(self, fitting_df, empirical_data, param_names, summary_stat_keys, fixed_params,
                num_ppc_simulations, num_posterior_draws, timeout_seconds) -> Dict[str, Any]:
        """
        Run Posterior Predictive Checks.
        
        Args:
            fitting_df: Fitting results DataFrame.
            empirical_data: Empirical data DataFrame.
            param_names: List of parameter names.
            summary_stat_keys: List of summary statistic keys.
            fixed_params: Fixed parameters for simulation.
            num_ppc_simulations: Number of PPC simulations per subject.
            num_posterior_draws: Number of posterior draws to use for PPC.
            timeout_seconds: Timeout for each subject's PPC in seconds.
            
        Returns:
            Dict containing the PPC results.
        """
        # Get unique subjects
        subjects = empirical_data['subject'].unique()
        num_subjects = len(subjects)
        
        # Initialize results
        empirical_stats = {}
        simulated_stats = {}
        coverage = {}
        
        # Process each subject
        for i, subject in enumerate(subjects):
            self.logger.info(f"Running PPC for subject {i+1}/{num_subjects} (ID: {subject})")
            
            # Get subject data
            subject_data = empirical_data[empirical_data['subject'] == subject]
            
            # Calculate empirical summary statistics
            stats_results = self.summary_stats_module.run(subject_data)
            subject_empirical_stats = stats_results['summary_stats']
            empirical_stats[subject] = subject_empirical_stats
            
            # Get subject's fitted parameters
            subject_params = {}
            for param in param_names:
                param_row = fitting_df[(fitting_df['subject'] == subject) & (fitting_df['parameter'] == param)]
                if len(param_row) > 0:
                    subject_params[param] = {
                        'mean': param_row['mean'].values[0],
                        'std': param_row['std'].values[0]
                    }
                    
            # Skip subject if parameters not found
            if len(subject_params) != len(param_names):
                self.logger.warning(f"Skipping subject {subject}: not all parameters found in fitting results")
                continue
                
            # Run PPC for this subject with timeout
            try:
                with ProcessPool(max_workers=1) as pool:
                    future = pool.schedule(
                        self._run_subject_ppc,
                        args=(subject_params, subject_data, param_names, summary_stat_keys, fixed_params, num_ppc_simulations, num_posterior_draws),
                        timeout=timeout_seconds
                    )
                    subject_simulated_stats = future.result()
                    simulated_stats[subject] = subject_simulated_stats
                    
                    # Calculate coverage
                    subject_coverage = self._calculate_coverage(subject_empirical_stats, subject_simulated_stats)
                    coverage[subject] = subject_coverage
                    
            except TimeoutError:
                self.logger.warning(f"Timeout for subject {subject} after {timeout_seconds} seconds")
                
            except Exception as e:
                self.logger.error(f"Error in PPC for subject {subject}: {str(e)}")
                
        # Calculate coverage summary
        coverage_summary = self._calculate_coverage_summary(coverage, summary_stat_keys)
        
        # Return results
        return {
            'empirical_stats': empirical_stats,
            'simulated_stats': simulated_stats,
            'coverage': coverage,
            'coverage_summary': coverage_summary,
            'num_subjects': num_subjects,
            'num_ppc_simulations': num_ppc_simulations,
            'summary_stat_keys': summary_stat_keys
        }
        
    def _run_subject_ppc(self, subject_params, subject_data, param_names, summary_stat_keys, 
                         fixed_params, num_ppc_simulations, num_posterior_draws):
        """
        Run PPC for a single subject.
        
        Args:
            subject_params: Subject's fitted parameters.
            subject_data: Subject's empirical data.
            param_names: List of parameter names.
            summary_stat_keys: List of summary statistic keys.
            fixed_params: Fixed parameters for simulation.
            num_ppc_simulations: Number of PPC simulations.
            num_posterior_draws: Number of posterior draws to use.
            
        Returns:
            Dict mapping simulation index to summary statistics.
        """
        # Initialize results
        simulated_stats = {}
        
        # Draw parameters from posterior
        drawn_params = []
        for _ in range(num_posterior_draws):
            param_set = {}
            for param in param_names:
                mean = subject_params[param]['mean']
                std = subject_params[param]['std']
                # Draw from normal distribution centered on posterior mean
                param_value = np.random.normal(mean, std)
                param_set[param] = param_value
            drawn_params.append(param_set)
            
        # Run simulations for each parameter set
        for i in range(num_ppc_simulations):
            # Select a random parameter set
            param_idx = np.random.randint(0, len(drawn_params))
            param_set = drawn_params[param_idx]
            
            # Run simulation
            sim_results = self.simulation_module.run(param_set, subject_data, fixed_params)
            simulated_data = sim_results['simulated_data']
            
            # Calculate summary statistics
            stats_results = self.summary_stats_module.run(simulated_data)
            sim_stats = stats_results['summary_stats']
            
            # Store results
            simulated_stats[i] = sim_stats
            
        return simulated_stats
        
    def _calculate_coverage(self, empirical_stats, simulated_stats):
        """
        Calculate coverage for a subject.
        
        Args:
            empirical_stats: Empirical summary statistics.
            simulated_stats: Simulated summary statistics.
            
        Returns:
            Dict mapping statistic names to coverage values.
        """
        coverage = {}
        
        for stat_name in empirical_stats:
            # Skip if empirical stat is NaN
            if np.isnan(empirical_stats[stat_name]):
                continue
                
            # Get empirical value
            empirical_value = empirical_stats[stat_name]
            
            # Get simulated values
            simulated_values = []
            for sim_idx in simulated_stats:
                if stat_name in simulated_stats[sim_idx] and not np.isnan(simulated_stats[sim_idx][stat_name]):
                    simulated_values.append(simulated_stats[sim_idx][stat_name])
                    
            # Skip if no valid simulated values
            if len(simulated_values) == 0:
                continue
                
            # Calculate coverage
            lower_bound = np.percentile(simulated_values, 2.5)
            upper_bound = np.percentile(simulated_values, 97.5)
            covered = lower_bound <= empirical_value <= upper_bound
            
            coverage[stat_name] = {
                'empirical_value': empirical_value,
                'simulated_mean': np.mean(simulated_values),
                'simulated_std': np.std(simulated_values),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'covered': covered
            }
            
        return coverage
        
    def _calculate_coverage_summary(self, coverage, summary_stat_keys):
        """
        Calculate coverage summary across all subjects.
        
        Args:
            coverage: Dict mapping subjects to coverage results.
            summary_stat_keys: List of summary statistic keys.
            
        Returns:
            Dict mappin
(Content truncated due to size limit. Use line ranges to read in chunks)