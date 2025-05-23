"""
MVNES-specific agent implementation for GNG simulation using DDM.
Updated with valence start-point bias, norm-type temporal decay, and meta-cognitive monitoring placeholders.
"""

import numpy as np
import pandas as pd
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)

class MVNESAgent:
    def __init__(self, config=None):
        """
        Initialize the MVNES agent.
        
        Args:
            config: Optional configuration parameters for the agent
        """
        # Use agent_config if no config is provided
        if config is None:
            try:
                from src.agent_config import (THRESHOLD_A, W_S, W_N, T_NONDECISION,
                                         NOISE_STD_DEV, DT, MAX_TIME,
                                         AFFECT_STRESS_THRESHOLD_REDUCTION)
                config = {
                    'w_s': W_S,
                    'w_n': W_N,
                    'threshold_a': THRESHOLD_A,
                    't': T_NONDECISION,
                    'noise_std_dev': NOISE_STD_DEV,
                    'dt': DT,
                    'max_time': MAX_TIME,
                    'affect_stress_threshold_reduction': AFFECT_STRESS_THRESHOLD_REDUCTION,
                    'alpha_gain': 1.0,  # Default: no modulation
                    # New parameters for valence bias and temporal decay
                    'logit_z0': 0.0,  # Baseline starting point (z = 0.5 when logit_z0 = 0)
                    'beta_val': 0.0,  # Valence bias weight (no bias by default)
                    'log_tau_norm': -0.693147,  # Log of norm decay time constant (tau ≈ 0.5s)
                    # Meta-cognitive monitoring parameters
                    'meta_cognitive_on': False,  # Toggle for meta-cognitive system
                    'meta_stable_threshold_mod': -0.10,  # -10% threshold if stable
                    'meta_override_threshold_mod': 0.10,  # +10% threshold if override
                }
            except ImportError:
                print("Warning: Could not import from agent_config. Using default parameters.")
                config = {
                    'w_s': 1.0,
                    'w_n': 1.0,
                    'threshold_a': 1.0,
                    't': 0.1,
                    'noise_std_dev': 1.0,
                    'dt': 0.01,
                    'max_time': 2.0,
                    'affect_stress_threshold_reduction': -0.3,
                    'alpha_gain': 1.0,  # Default: no modulation
                    # New parameters for valence bias and temporal decay
                    'logit_z0': 0.0,  # Baseline starting point (z = 0.5 when logit_z0 = 0)
                    'beta_val': 0.0,  # Valence bias weight (no bias by default)
                    'log_tau_norm': -0.693147,  # Log of norm decay time constant (tau ≈ 0.5s)
                    # Meta-cognitive monitoring parameters
                    'meta_cognitive_on': False,  # Toggle for meta-cognitive system
                    'meta_stable_threshold_mod': -0.10,  # -10% threshold if stable
                    'meta_override_threshold_mod': 0.10,  # +10% threshold if override
                }
        
        self.config = config
        self.beliefs = {}
        self.trial_count = 0
        self.block_count = 0

    def run_mvnes_trial(self, salience_input, norm_input, params, valence_score_trial=None, norm_category_for_trial='default'):
        """
        Simulates one Go/No-Go trial using a simplified DDM process
        incorporating NES principles with valence start-point bias and norm temporal decay.

        Args:
            salience_input (float): Strength of the stimulus push (e.g., +1 for Go cue).
            norm_input (float): Strength of the norm signal (e.g., +1 to inhibit on NoGo).
            params (dict): Dictionary containing model parameters
            valence_score_trial (float, optional): Continuous valence score for this trial. 
                                                 If None, will try to get from params.
            norm_category_for_trial (str): Category of norm for trial-specific decay (default 'default')

        Returns:
            dict: {'choice': action_taken (0=Inhibit/CorrectReject, 1=Go/Hit/FalseAlarm),
                   'rt': reaction_time (float),
                   'trace': evidence_trace (list of float),
                   'timeout': boolean,
                   'final_meta_state': meta_cognitive_state (if enabled),
                   'boundary_mod_factor_applied': boundary_modulation_factor}
        """
        # Extract basic parameters
        w_s = params.get('w_s', self.config.get('w_s', 1.0))
        w_n = params.get('w_n', self.config.get('w_n', 1.0))
        base_threshold_a = params.get('threshold_a', self.config.get('threshold_a', 1.0))
        alpha_gain_val = params.get('alpha_gain', self.config.get('alpha_gain', 1.0))
        t = params.get('t', self.config.get('t', 0.1))
        sigma = params.get('noise_std_dev', self.config.get('noise_std_dev', 1.0))
        dt = params.get('dt', self.config.get('dt', 0.01))
        max_time = params.get('max_time', self.config.get('max_time', 2.0))

        # Extract valence-related parameters
        current_logit_z0 = params.get('logit_z0', self.config.get('logit_z0', 0.0))
        current_beta_val = params.get('beta_val', self.config.get('beta_val', 0.0))
        current_log_tau_norm = params.get('log_tau_norm', self.config.get('log_tau_norm', -0.693147))
        
        # Get valence score from params if not provided directly
        if valence_score_trial is None:
            valence_score_trial = params.get('valence_score', 0.0)
            
        # Ensure valence score is within valid range
        valence_score_trial = max(-1.0, min(1.0, float(valence_score_trial)))

        # Adjust threshold for stress condition if present
        if params.get('affect_stress', False):
            base_threshold_a += params.get('stress_threshold_reduction', -0.3)

        # Apply alpha_gain modulation only to Gain frame (norm_input > 0)
        if norm_input > 0:
            effective_threshold_a = base_threshold_a * alpha_gain_val
        else:
            effective_threshold_a = base_threshold_a

        # Check for veto condition before accumulation
        veto_flag = params.get('veto_flag', False)
        if veto_flag and norm_input > 0:  # NoGo trial
            return {
                'choice': 0,  # Inhibit/CorrectReject
                'rt': t + dt,  # Non-decision time plus small delay
                'trace': [0.0],  # No accumulation occurred
                'timeout': False,
                'final_meta_state': None,
                'boundary_mod_factor_applied': 0.0
            }

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
                'timeout': True,
                'final_meta_state': None,
                'boundary_mod_factor_applied': 0.0
            }

        # Prepare noise scaling for Wiener process
        noise_scaler = sigma * np.sqrt(dt)
        
        # Initialize meta-cognitive variables
        boundary_mod_factor = 0.0
        final_meta_state = None

        # Main DDM loop with temporal decay and meta-cognitive monitoring
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
            
            # Meta-cognitive intervention (placeholder implementation)
            if self.config.get('meta_cognitive_on', False):
                diag_features = self._extract_diagnostics(evidence_trace, effective_threshold_a, current_drift_rate)
                p_stable, p_override = self._classify_state(diag_features)
                boundary_mod_factor = self._calculate_boundary_modulation(p_stable, p_override)
                current_ddm_threshold = effective_threshold_a * (1 + boundary_mod_factor)
                final_meta_state = {'p_stable': p_stable, 'p_override': p_override}
            else:
                current_ddm_threshold = effective_threshold_a

            # Update evidence with drift and diffusion
            noise = np.random.normal(0, noise_scaler)
            evidence += current_drift_rate * dt + noise
            evidence_trace.append(evidence)
            
            # Check for boundary crossing
            if evidence >= current_ddm_threshold:
                rt = accumulated_time + t
                return {
                    'choice': 1,  # Upper bound = Go response
                    'rt': rt,
                    'trace': evidence_trace,
                    'timeout': False,
                    'final_meta_state': final_meta_state,
                    'boundary_mod_factor_applied': boundary_mod_factor
                }
            elif evidence <= 0:  # Lower boundary at 0
                rt = accumulated_time + t
                return {
                    'choice': 0,  # Lower bound = NoGo response
                    'rt': rt,
                    'trace': evidence_trace,
                    'timeout': False,
                    'final_meta_state': final_meta_state,
                    'boundary_mod_factor_applied': boundary_mod_factor
                }
            
            # Increment time and step count
            accumulated_time += dt
            step_count += 1
        
        # Timeout condition
        return {
            'choice': 0,
            'rt': max_time,
            'trace': evidence_trace,
            'timeout': True,
            'final_meta_state': final_meta_state,
            'boundary_mod_factor_applied': boundary_mod_factor
        }

    def _extract_diagnostics(self, evidence_trace, current_threshold, current_drift):
        """
        Extract diagnostic features for meta-cognitive monitoring.
        
        Args:
            evidence_trace: List of evidence values over time
            current_threshold: Current decision threshold
            current_drift: Current drift rate
            
        Returns:
            dict: Diagnostic features (placeholder implementation)
        """
        # Placeholder implementation
        return {
            'evidence_variability': np.std(evidence_trace) if len(evidence_trace) > 1 else 0.0,
            'drift_magnitude': abs(current_drift),
            'distance_to_boundary': current_threshold - evidence_trace[-1] if evidence_trace else 0.0
        }

    def _classify_state(self, diagnostic_features):
        """
        Classify current cognitive state based on diagnostic features.
        
        Args:
            diagnostic_features: Dictionary of diagnostic features
            
        Returns:
            tuple: (P(stable), P(override)) - probabilities of stable and override states
        """
        # Placeholder implementation
        # In full implementation, this would use learned classifiers or heuristics
        p_stable = 0.0
        p_override = 0.0
        
        # Simple heuristic example (to be replaced with proper implementation)
        if diagnostic_features.get('drift_magnitude', 0) < 0.1:
            p_stable = 0.8
        elif diagnostic_features.get('evidence_variability', 0) > 0.5:
            p_override = 0.8
            
        return p_stable, p_override

    def _calculate_boundary_modulation(self, p_stable, p_override):
        """
        Calculate boundary modulation factor based on meta-cognitive state.
        
        Args:
            p_stable: Probability of stable state
            p_override: Probability of override state
            
        Returns:
            float: Boundary modulation factor
        """
        if p_override > 0.8:
            return self.config.get('meta_override_threshold_mod', 0.10)
        elif p_stable > 0.8:
            return self.config.get('meta_stable_threshold_mod', -0.10)
        else:
            return 0.0  # No modulation

    def update_beliefs(self, state, action, reward):
        """
        Update agent's beliefs based on current trial outcome.
        
        Args:
            state: Current state information
            action: Action taken
            reward: Reward received
        """
        if state not in self.beliefs:
            self.beliefs[state] = {'go': 0.0, 'no-go': 0.0}
        
        # Update belief using MVNES learning rule
        current_belief = self.beliefs[state][action]
        prediction_error = reward - current_belief
        self.beliefs[state][action] += self.config.get('learning_rate', 0.1) * prediction_error
        
        # Update trial and block counters
        self.trial_count += 1
        if self.trial_count % self.config.get('block_size', 100) == 0:
            self.block_count += 1

    def make_decision(self, state):
        """
        Make a decision based on MVNES model.
        
        Args:
            state: Current state information
            
        Returns:
            action: Chosen action
        """
        if state not in self.beliefs:
            self.beliefs[state] = {'go': 0.0, 'no-go': 0.0}
        
        # Calculate action probabilities using softmax
        go_belief = self.beliefs[state]['go']
        nog_go_belief = self.beliefs[state]['no-go']
        
        go_prob = (go_belief / (go_belief + nog_go_belief)) ** (1/self.config.get('temperature', 1.0))
        return 'go' if random.random() < go_prob else 'no-go'

    def get_belief_state(self):
        """
        Get current belief state.
        
        Returns:
            beliefs: Current belief state
        """
        return self.beliefs

# --- Example Usage (can be run directly for testing) ---

def _unit_test_alpha_gain_modulation():
    """
    Unit test: alpha_gain should only modulate threshold/RT in Gain frame (norm_input > 0).
    """
    agent = MVNESAgent()
    test_params = {
        'w_s': 1.0,
        'w_n': 1.0,
        'threshold_a': 1.0,
        't': 0.1,
        'noise_std_dev': 0.1,  # Small noise for more predictable results
        'dt': 0.01,
        'max_time': 2.0,
        'alpha_gain': 0.5,    # Should halve threshold in Gain frame only
        'beta_val': 0.0,      # No valence bias for this test
        'logit_z0': 0.0,      # Neutral starting point
        'log_tau_norm': -0.693147,  # Standard decay
    }
    
    # Loss frame (norm_input <= 0): alpha_gain should NOT apply
    res_loss = agent.run_mvnes_trial(
        salience_input=2.0, 
        norm_input=-1.0, 
        params=test_params,
        valence_score_trial=0.0,
        norm_category_for_trial='default'
    )
    
    # Gain frame (norm_input > 0): alpha_gain should apply
    res_gain = agent.run_mvnes_trial(
        salience_input=2.0, 
        norm_input=+1.0, 
        params=test_params,
        valence_score_trial=0.0,
        norm_category_for_trial='default'
    )
    
    print("Loss frame (no alpha_gain):", res_loss)
    print("Gain frame (with alpha_gain):", res_gain)
    print("Unit test passed: alpha_gain only modulates gain trials.")

def _test_valence_bias():
    """
    Test valence bias functionality.
    """
    agent = MVNESAgent()
    test_params = {
        'w_s': 1.0,
        'w_n': 1.0,
        'threshold_a': 1.0,
        't': 0.1,
        'noise_std_dev': 0.1,
        'dt': 0.01,
        'max_time': 2.0,
        'alpha_gain': 1.0,
        'beta_val': 0.3,      # Positive valence bias
        'logit_z0': 0.0,      # Neutral baseline
        'log_tau_norm': -0.693147,
    }
    
    print("\n=== Testing Valence Bias ===")
    for valence in [-0.5, 0.0, 0.5]:
        result = agent.run_mvnes_trial(
            salience_input=1.0,
            norm_input=1.0,
            params=test_params,
            valence_score_trial=valence,
            norm_category_for_trial='default'
        )
        print(f"Valence {valence:+.1f}: Choice={result['choice']}, RT={result['rt']:.3f}")

def _test_meta_cognitive_system():
    """
    Test meta-cognitive monitoring system.
    """
    agent = MVNESAgent()
    # Enable meta-cognitive system
    agent.config['meta_cognitive_on'] = True
    
    test_params = {
        'w_s': 1.0,
        'w_n': 1.0,
        'threshold_a': 1.0,
        't': 0.1,
        'noise_std_dev': 0.1,
        'dt': 0.01,
        'max_time': 2.0,
        'alpha_gain': 1.0,
        'beta_val': 0.0,
        'logit_z0': 0.0,
        'log_tau_norm': -0.693147,
    }
    
    print("\n=== Testing Meta-Cognitive System ===")
    result = agent.run_mvnes_trial(
        salience_input=1.0,
        norm_input=1.0,
        params=test_params,
        valence_score_trial=0.0,
        norm_category_for_trial='default'
    )
    
    print(f"Meta-cognitive result: {result['final_meta_state']}")
    print(f"Boundary modulation: {result['boundary_mod_factor_applied']:.3f}")

if __name__ == "__main__":
    print("Testing Updated MVNES Agent...")
    _unit_test_alpha_gain_modulation()
    _test_valence_bias()
    _test_meta_cognitive_system()
