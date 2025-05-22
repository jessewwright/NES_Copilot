"""
Trial Generator for NES Copilot

This module handles generating trial templates for simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union


class TrialGenerator:
    """
    Trial generator for the NES Copilot system.
    
    Handles generating trial templates for simulations.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the trial generator.
        
        Args:
            config: Configuration dictionary.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        
        # Get trial template configuration
        template_config = self.config.get('trial_template', {})
        self.num_template_trials = template_config.get('num_template_trials', 1000)
        
    def generate_trial_template(self, empirical_data: pd.DataFrame, valence_scores: Dict[str, float], seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a trial template for simulations.
        
        Args:
            empirical_data: Preprocessed empirical data.
            valence_scores: Dict mapping stimuli to valence scores.
            seed: Random seed for reproducibility.
            
        Returns:
            DataFrame containing the trial template.
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Sample trials from empirical data
        sampled_indices = np.random.choice(
            empirical_data.index,
            size=self.num_template_trials,
            replace=True
        )
        template = empirical_data.loc[sampled_indices].copy()
        
        # Add valence scores if not already present
        if 'valence_score' not in template.columns:
            template['valence_score'] = template.apply(
                lambda row: valence_scores.get(f"{row['frame']}_{row['cond']}", 0.0),
                axis=1
            )
        
        # Add norm category for trial (placeholder logic)
        template['norm_category_for_trial'] = template.apply(
            lambda row: 'gain' if row['is_gain_frame'] else 'loss',
            axis=1
        )
        
        # Reset index
        template.reset_index(drop=True, inplace=True)
        
        self.logger.info(f"Generated trial template with {len(template)} trials")
        return template
