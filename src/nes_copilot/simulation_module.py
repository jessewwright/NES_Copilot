"""
Simulation Module for NES Copilot with sbi Integration

This module handles simulation of the MVNESAgent and integrates with sbi for NPE training.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging

from nes_copilot.agent_mvnes import MVNESAgent
from nes_copilot.summary_stats_module import calculate_summary_stats_roberts
from nes_copilot.stats_schema import validate_summary_stats

logger = logging.getLogger(__name__)

class SimulationModule:
    """
    Simulation module for the NES Copilot system with sbi integration.
    
    Handles simulation of the MVNESAgent and provides an interface for sbi.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the simulation module.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_names = config_manager.get_parameter_names()
        
        # Log initialization
        logger.info(f"Initialized SimulationModule with parameters: {self.param_names}")
    
    def get_simulator(self, trial_template: pd.DataFrame):
        """
        Create a simulator function compatible with sbi.
        
        Args:
            trial_template: DataFrame containing the trial structure template.
            
        Returns:
            A function that takes parameters and returns summary statistics.
        """
        def simulator(params: torch.Tensor) -> torch.Tensor:
            """
            Simulator function compatible with sbi.
            
            Args:
                params: Tensor of shape (n_simulations, n_parameters)
                
            Returns:
                Tensor of shape (n_simulations, n_summary_stats)
            """
            # Convert tensor to numpy if needed
            if isinstance(params, torch.Tensor):
                params = params.numpy()
            
            # Handle both single and batch simulations
            if params.ndim == 1:
                params = params[np.newaxis, :]
            
            n_simulations = params.shape[0]
            all_summary_stats = []
            
            for i in range(n_simulations):
                # Convert parameters to dict
                param_dict = dict(zip(self.param_names, params[i]))
                
                # Run simulation
                sim_data = self.run_single_simulation(param_dict, trial_template)
                
                # Calculate summary statistics
                summary_stats = calculate_summary_stats_roberts(sim_data)
                
                # Convert to numpy array in the correct order
                stat_values = np.array([
                    summary_stats[stat] for stat in self.config.get_summary_stat_keys()
                ])
                all_summary_stats.append(stat_values)
            
            return torch.as_tensor(np.array(all_summary_stats), dtype=torch.float32)
        
        return simulator
    
    def run_single_simulation(
        self, 
        parameters: Dict[str, float], 
        trial_template: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run a single simulation with the given parameters and trial template.
        
        Args:
            parameters: Dictionary of parameter values.
            trial_template: DataFrame containing the trial structure.
            
        Returns:
            DataFrame with simulation results.
        """
        # Create agent with current parameters
        agent = MVNESAgent(**parameters)
        
        # Run simulation for each trial in the template
        results = []
        for _, trial in trial_template.iterrows():
            # Run trial
            choice, rt = agent.run_trial(
                frame=trial['frame'],
                cond=trial['cond'],
                prob=trial['prob'],
                is_gain_frame=trial['is_gain_frame'],
                time_constrained=trial['time_constrained'],
                valence_score=trial['valence_score'],
                norm_category=trial['norm_category_for_trial']
            )
            
            # Store results
            results.append({
                'subject': 1,  # Single subject for simulation
                'trial': trial['trial'],
                'block': trial['block'],
                'frame': trial['frame'],
                'cond': trial['cond'],
                'gamble_chosen': int(choice),
                'reaction_time': rt,
                'gamble_prob': trial['prob'],
                'sure_amount': trial['sure_amount'],
                'endowment': trial['endowment'],
                'valence_score': trial['valence_score'],
                'norm_category': trial['norm_category_for_trial']
            })
        
        return pd.DataFrame(results)
    
    def run_batch_simulations(
        self,
        thetas: np.ndarray,
        trial_template: pd.DataFrame,
        n_workers: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run multiple simulations in parallel.
        
        Args:
            thetas: Array of parameter sets (n_simulations, n_parameters)
            trial_template: DataFrame containing the trial structure
            n_workers: Number of parallel workers
            
        Returns:
            Tuple of (thetas, summary_stats) as numpy arrays
        """
        from joblib import Parallel, delayed
        
        def process_one_theta(theta):
            param_dict = dict(zip(self.param_names, theta))
            sim_data = self.run_single_simulation(param_dict, trial_template)
            summary_stats = calculate_summary_stats_roberts(sim_data)
            return np.array([summary_stats[stat] for stat in self.config.get_summary_stat_keys()])
        
        # Run simulations in parallel
        summary_stats = Parallel(n_jobs=n_workers)(
            delayed(process_one_theta)(theta) for theta in thetas
        )
        
        return thetas, np.array(summary_stats)
        self.logger.info("Starting simulation")
        
        # Validate inputs
        self.validate_inputs(parameters=parameters, trial_template=trial_template)
        
        # Run simulation
        simulated_data = self.simulate(parameters, trial_template, fixed_params)
        self.logger.info(f"Simulated {len(simulated_data)} trials")
        
        # Save outputs
        output_paths = self.save_outputs({'simulated_data': simulated_data})
        
        # Return results
        results = {
            'num_trials': len(simulated_data),
            'parameters': parameters,
            'fixed_params': fixed_params,
            'output_paths': output_paths,
            'simulated_data': simulated_data  # Include the actual data for direct use
        }
        
        self.logger.info("Simulation completed successfully")
        return results
        
    def validate_inputs(self, parameters: Dict[str, float], trial_template: pd.DataFrame, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            parameters: Parameter set (theta) for simulation.
            trial_template: Trial structure template.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        # Check if parameters are provided
        if not parameters:
            raise ValueError("Parameters not provided")
            
        # Check if trial template is provided
        if trial_template is None or len(trial_template) == 0:
            raise ValueError("Trial template not provided or empty")
            
        # Check if trial template has required columns
        required_columns = ['frame', 'cond', 'is_gain_frame', 'time_constrained', 'valence_score', 'norm_category_for_trial']
        for col in required_columns:
            if col not in trial_template.columns:
                raise ValueError(f"Trial template missing required column: {col}")
                
        return True
        
    def simulate(self, parameters: Dict[str, float], trial_template: pd.DataFrame, fixed_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Run simulations with the given parameters and trial template.
        
        Args:
            parameters: Parameter set (theta) for simulation.
            trial_template: Trial structure template.
            fixed_params: Optional fixed parameters for the model.
            
        Returns:
            DataFrame containing simulated choices and RTs.
        """
        # Get fixed parameters from config if not provided
        if fixed_params is None:
            fixed_params = self.config.get('mvnes_agent', {}).get('fixed_params', {})
            
        # Combine parameters and fixed parameters
        all_params = {**fixed_params, **parameters}
        
        # Initialize MVNESAgent
        agent = self.MVNESAgent(**all_params)
        
        # Initialize results DataFrame
        results = []
        
        # Run simulations for each trial
        for _, trial in trial_template.iterrows():
            # Extract trial information
            frame = trial['frame']
            cond = trial['cond']
            is_gain_frame = trial['is_gain_frame']
            time_constrained = trial['time_constrained']
            valence_score = trial['valence_score']
            norm_category = trial['norm_category_for_trial']
            
            # Run trial
            choice, rt = agent.run_mvnes_trial(
                is_gain_frame=is_gain_frame,
                time_constrained=time_constrained,
                valence_score_trial=valence_score,
                norm_category_for_trial=norm_category
            )
            
            # Store results
            results.append({
                'frame': frame,
                'cond': cond,
                'is_gain_frame': is_gain_frame,
                'time_constrained': time_constrained,
                'valence_score': valence_score,
                'norm_category': norm_category,
                'choice': choice,
                'rt': rt
            })
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
        
    def save_outputs(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save module outputs to disk using the data manager.
        
        Args:
            results: Results to save.
            
        Returns:
            Dict mapping output names to their paths.
        """
        output_paths = {}
        
        # Save simulated data
        simulated_data_path = self.data_manager.get_output_path('simulation', 'simulated_data.csv')
        results['simulated_data'].to_csv(simulated_data_path, index=False)
        output_paths['simulated_data'] = simulated_data_path
        
        return output_paths
