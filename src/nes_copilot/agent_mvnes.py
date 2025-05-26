"""
MVNESAgent for NES Copilot

This module implements the Normative Executive System (NES) agent for simulations.
Adapted from the Hegemonikon project's MVNES implementation.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Literal

# Set up logging
logger = logging.getLogger(__name__)

print("DEBUG: Loading MVNESAgent from src/nes_copilot/agent_mvnes.py")

class MVNESAgent:
    """
    Multivariate Normative Executive System (MVNES) Agent.
    
    Implements the NES model for decision-making simulations with the following parameters:
    - v_norm: Drift rate for the normative process
    - alpha_gain: Gain modulation parameter
    - beta_val: Valence modulation parameter
    - logit_z0: Logit of the baseline starting point (fixed)
    - log_tau_norm: Log of the norm-type temporal decay (fixed)
    - a_0: Boundary separation
    - w_s_eff: Starting point bias
    - t_0: Non-decision time
    """
    
    def __init__(self, v_norm: float = 1.0, a_0: float = 1.0, w_s_eff: float = 0.5,
                 t_0: float = 0.2, alpha_gain: float = 0.0, beta_val: float = 0.0,
                 logit_z0: float = 0.0, log_tau_norm: float = -0.693147, **kwargs):
        """
        Initialize the MVNESAgent with the 6-parameter model.
        
        Args:
            v_norm: Drift rate for the normative process (typical range: 0.1-5.0)
            a_0: Boundary separation (typical range: 0.5-3.0)
            w_s_eff: Starting point bias (typical range: 0.3-0.7)
            t_0: Non-decision time in seconds (typical range: 0.1-0.5)
            alpha_gain: Gain modulation parameter (typical range: -1.0 to 1.0)
            beta_val: Valence modulation parameter (typical range: -1.0 to 1.0)
            logit_z0: Logit of the baseline starting point (fixed, default: 0.0)
            log_tau_norm: Log of the norm-type temporal decay (fixed, default: -0.693147)
            **kwargs: Additional parameters (ignored)
        """
        # Core model parameters
        self.v_norm = float(v_norm)
        self.a_0 = float(a_0)
        self.w_s_eff = float(w_s_eff)
        self.t_0 = float(t_0)
        self.alpha_gain = float(alpha_gain)
        self.beta_val = float(beta_val)
        
        # Fixed parameters
        self.logit_z0 = float(logit_z0)
        self.log_tau_norm = float(log_tau_norm)
        
        # Initialize random number generator
        self.rng = np.random.RandomState()
        
        # Parameter validation
        self._validate_parameters()
        
        # Initialize trial count
        self.trial_count = 0
        
        # Set up default configuration
        self.config = {
            'w_s': 1.0,  # Salience weight
            'w_n': 1.0,  # Norm weight
            'threshold_a': self.a_0,
            't': self.t_0,
            'noise_std_dev': 1.0,
            'dt': 0.01,
            'max_time': 2.0,
            'alpha_gain': self.alpha_gain,
            'logit_z0': self.logit_z0,
            'beta_val': self.beta_val,
            'log_tau_norm': self.log_tau_norm,
            'meta_cognitive_on': False  # Disable meta-cognitive monitoring by default
        }
    
    def _validate_parameters(self):
        """Validate parameter ranges and raise ValueError if out of bounds."""
        if not (0.1 <= self.v_norm <= 5.0):
            raise ValueError(f"v_norm ({self.v_norm}) should be between 0.1 and 5.0")
        if not (0.5 <= self.a_0 <= 3.0):
            raise ValueError(f"a_0 ({self.a_0}) should be between 0.5 and 3.0")
        if not (0.3 <= self.w_s_eff <= 0.7):
            raise ValueError(f"w_s_eff ({self.w_s_eff}) should be between 0.3 and 0.7")
        if not (0.1 <= self.t_0 <= 0.5):
            raise ValueError(f"t_0 ({self.t_0}) should be between 0.1 and 0.5")
        if not (-1.0 <= self.alpha_gain <= 1.0):
            raise ValueError(f"alpha_gain ({self.alpha_gain}) should be between -1.0 and 1.0")
        if not (-1.0 <= self.beta_val <= 1.0):
            raise ValueError(f"beta_val ({self.beta_val}) should be between -1.0 and 1.0")
    
    def set_seed(self, seed: int):
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
    
    def run_mvnes_trial(self, is_gain_frame: bool, time_constrained: bool,
                       valence_score_trial: float = 0.0, 
                       norm_category_for_trial: str = 'default') -> Tuple[int, float]:
        print(">>>> MEGA DEBUG: AGENT MVNES RUN_MVNES_TRIAL BEING EXECUTED - VERSION FROM src/nes_copilot/agent_mvnes.py <<<<")
        """
        Run a single trial simulation with the MVNES model.
        
        Args:
            is_gain_frame: Whether the trial is in the gain frame (True) or loss frame (False)
            time_constrained: Whether the trial is time-constrained
            valence_score_trial: Valence score for the trial (default: 0.0)
            norm_category_for_trial: Norm category for trial-specific decay (default: 'default')
            
        Returns:
            Tuple of (choice, reaction_time)
            - choice: 0 for sure option, 1 for gamble
            - reaction_time: in seconds
        """
        # Increment trial counter
        self.trial_count += 1
        
        # Set up trial parameters based on frame
        salience_input = 1.0  # Default salience for Go/No-Go task
        norm_input = 1.0 if is_gain_frame else -1.0  # Norm input based on frame
        
        # Prepare parameters for the trial
        params = {
            'w_s': self.config['w_s'],
            'w_n': self.v_norm,  # Using v_norm as the weight for norm input
            'threshold_a': self.a_0,
            't': self.t_0,
            'noise_std_dev': self.config['noise_std_dev'],
            'dt': self.config['dt'],
            'max_time': self.config['max_time'],
            'alpha_gain': self.alpha_gain,
            'logit_z0': self.logit_z0,
            'beta_val': self.beta_val,
            'log_tau_norm': self.log_tau_norm,
            'meta_cognitive_on': self.config['meta_cognitive_on']
        }
        
        # Run the trial
        result = self._run_trial(
            salience_input=salience_input,
            norm_input=norm_input,
            params=params,
            valence_score_trial=valence_score_trial,
            norm_category_for_trial=norm_category_for_trial
        )
        
        # Apply time constraint if needed
        if time_constrained and 'rt' in result:
            result['rt'] *= 0.8  # Example: 20% reduction in RT for time-constrained trials
        
        return result.get('choice', 0), result.get('rt', 0.0)
    
    def _run_trial(self, salience_input: float, norm_input: float, params: Dict[str, float],
                  valence_score_trial: float = 0.0, norm_category_for_trial: str = 'default') -> Dict[str, Any]:
        """
        Internal method to run a single trial with the MVNES model.
        
        Args:
            salience_input: Strength of the stimulus push
            norm_input: Strength of the norm signal
            params: Dictionary of model parameters
            valence_score_trial: Valence score for the trial
            norm_category_for_trial: Norm category for trial-specific decay
            
        Returns:
            Dictionary containing trial results
        """
        # Extract parameters with defaults
        w_s = params.get('w_s', self.config['w_s'])
        w_n = params.get('w_n', self.config.get('w_n', 1.0))
        base_threshold_a = params.get('threshold_a', self.config['threshold_a'])
        alpha_gain_val = params.get('alpha_gain', self.config['alpha_gain'])
        t = params.get('t', self.config['t'])
        sigma = params.get('noise_std_dev', self.config['noise_std_dev'])
        dt = params.get('dt', self.config['dt'])
        max_time = params.get('max_time', self.config['max_time'])
        
        # Extract valence and decay parameters
        current_logit_z0 = params.get('logit_z0', self.config['logit_z0'])
        current_beta_val = params.get('beta_val', self.config['beta_val'])
        current_log_tau_norm = params.get('log_tau_norm', self.config['log_tau_norm'])
        
        # Apply alpha_gain modulation only to Gain frame (norm_input > 0)
        if norm_input > 0:
            effective_threshold_a = base_threshold_a * (1.0 + alpha_gain_val)
        else:
            effective_threshold_a = base_threshold_a
        
        # Calculate starting point using valence bias
        logit_z_trial = current_logit_z0 + current_beta_val * valence_score_trial
        z_rel_trial = 1.0 / (1.0 + np.exp(-logit_z_trial))  # Sigmoid transformation
        evidence_start_point = z_rel_trial * effective_threshold_a
        evidence = evidence_start_point
        
        # Initialize time tracking and evidence trace
        accumulated_time = 0.0
        evidence_trace = [evidence]
        max_decision_time = max(dt, max_time - t)
        
        if max_decision_time <= 0:
            return {
                'choice': 0,
                'rt': max_time,
                'trace': evidence_trace,
                'timeout': True
            }
        
        # Prepare noise scaling for Wiener process
        noise_scaler = sigma * np.sqrt(dt)
        
        # Main DDM loop with temporal decay
        step_count = 0
        max_steps = int(max_decision_time / dt) + 10
        
        while accumulated_time < max_decision_time and step_count < max_steps:
            # Calculate norm decay factor based on trial category
            norm_category_log_tau = params.get(f'log_tau_{norm_category_for_trial}', current_log_tau_norm)
            tau_norm = np.exp(norm_category_log_tau)
            decay_factor = np.exp(-accumulated_time / tau_norm) if tau_norm > 0 else 1.0
            
            # Calculate current drift rate with temporal decay
            current_v_norm_effect = w_n * decay_factor * norm_input
            current_drift_rate = w_s * salience_input - current_v_norm_effect
            
            # Update evidence with drift and diffusion
            noise = self.rng.normal(0, noise_scaler)
            evidence += current_drift_rate * dt + noise
            evidence_trace.append(evidence)
            
            # Check for boundary crossing
            if evidence >= effective_threshold_a:
                rt = accumulated_time + t
                return {
                    'choice': 1,  # Upper bound = Go response
                    'rt': rt,
                    'trace': evidence_trace,
                    'timeout': False
                }
            elif evidence <= 0:  # Lower boundary at 0
                rt = accumulated_time + t
                return {
                    'choice': 0,  # Lower bound = NoGo response
                    'rt': rt,
                    'trace': evidence_trace,
                    'timeout': False
                }
            
            # Update time and step counter
            accumulated_time += dt
            step_count += 1
        
        # If we get here, we hit the time limit
        return {
            'choice': 0 if evidence < (effective_threshold_a / 2) else 1,  # Choose based on evidence position
            'rt': max_time,
            'trace': evidence_trace,
            'timeout': True
        }
