"""
MVNESAgent for NES Copilot

This module implements the Normative Executive System (NES) agent for simulations.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple


class MVNESAgent:
    """
    Multivariate Normative Executive System (MVNES) Agent.
    
    Implements the NES model for decision-making simulations.
    """
    
    def __init__(self, v_norm=1.0, alpha_gain=0.5, beta_val=0.0, logit_z0=0.0, log_tau_norm=0.0, **kwargs):
        """
        Initialize the MVNESAgent.
        
        Args:
            v_norm: Drift rate for the normative process.
            alpha_gain: Gain modulation parameter.
            beta_val: Valence-driven start-point bias.
            logit_z0: Logit of the baseline starting point.
            log_tau_norm: Log of the norm-type temporal decay.
            **kwargs: Additional parameters.
        """
        self.v_norm = v_norm
        self.alpha_gain = alpha_gain
        self.beta_val = beta_val
        self.logit_z0 = logit_z0
        self.log_tau_norm = log_tau_norm
        
        # Store additional parameters
        self.additional_params = kwargs
        
        # Initialize random number generator
        self.rng = np.random.RandomState()
        
    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed.
        """
        self.rng = np.random.RandomState(seed)
        
    def run_mvnes_trial(self, is_gain_frame: bool, time_constrained: bool, 
                        valence_score_trial: float, norm_category_for_trial: str) -> Tuple[int, float]:
        """
        Run a single trial simulation.
        
        Args:
            is_gain_frame: Whether the trial is in the gain frame.
            time_constrained: Whether the trial is time-constrained.
            valence_score_trial: Valence score for the trial.
            norm_category_for_trial: Norm category for the trial.
            
        Returns:
            Tuple of (choice, reaction time).
            Choice: 0 for sure option, 1 for gamble.
            Reaction time: in seconds.
        """
        # Calculate drift rate based on frame and alpha_gain
        drift_rate = self.v_norm
        if is_gain_frame:
            drift_rate *= (1 + self.alpha_gain)
        else:
            drift_rate *= (1 - self.alpha_gain)
            
        # Calculate starting point based on valence and beta_val
        z0 = 1 / (1 + np.exp(-self.logit_z0))  # Convert logit to probability
        z = z0 + self.beta_val * valence_score_trial
        z = np.clip(z, 0.1, 0.9)  # Ensure z is within reasonable bounds
        
        # Calculate temporal decay based on norm category
        tau_norm = np.exp(self.log_tau_norm)
        if norm_category_for_trial == 'gain':
            decay = 1.0  # No decay for gain norms
        else:
            decay = np.exp(-tau_norm)  # Exponential decay for loss norms
            
        # Apply decay to drift rate
        drift_rate *= decay
        
        # Simulate decision process (simplified DDM)
        # In a real implementation, this would be a more complex DDM simulation
        # For this example, we'll use a simplified approach
        
        # Time step for simulation
        dt = 0.001  # 1 ms
        
        # Decision boundaries
        upper_bound = 1.0
        lower_bound = 0.0
        
        # Initial position
        position = z
        
        # Noise scale
        noise_scale = 0.1
        
        # Maximum time (to prevent infinite loops)
        max_time = 10.0  # 10 seconds
        
        # Time counter
        time = 0.0
        
        # Simulate until decision is reached or max time is exceeded
        while (position > lower_bound and position < upper_bound and time < max_time):
            # Update position
            position += drift_rate * dt + noise_scale * np.sqrt(dt) * self.rng.normal()
            
            # Update time
            time += dt
            
        # Determine choice and RT
        if position >= upper_bound:
            choice = 1  # Gamble
        elif position <= lower_bound:
            choice = 0  # Sure option
        else:
            # If max time is reached without decision, randomly choose
            choice = self.rng.randint(0, 2)
            
        # Apply time constraint effect (simplified)
        if time_constrained:
            # Time pressure reduces RT
            time *= 0.8
            
        return choice, time
