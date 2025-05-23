"""
Simulation Module for NES Copilot

This module handles simulation of the MVNESAgent.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

from nes_copilot.module_base import ModuleBase


class SimulationModule(ModuleBase):
    """
    Simulation module for the NES Copilot system.
    
    Handles simulation of the MVNESAgent.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the simulation module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        super().__init__(config_manager, data_manager, logging_manager)
        
        # Get module configuration
        self.config = self.config_manager.get_module_config('simulator')
        
        # Import MVNESAgent
        from nes_copilot.simulation.agent_mvnes import MVNESAgent
        self.MVNESAgent = MVNESAgent
        
    def run(self, parameters: Dict[str, float], trial_template: pd.DataFrame, fixed_params: Optional[Dict[str, float]] = None, **kwargs) -> Dict[str, Any]:
        """
        Run simulations with the given parameters and trial template.
        
        Args:
            parameters: Parameter set (theta) for simulation.
            trial_template: Trial structure template.
            fixed_params: Optional fixed parameters for the model.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the simulation results.
        """
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
