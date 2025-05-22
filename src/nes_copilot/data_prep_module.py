"""
Data Preparation Module for NES Copilot

This module handles loading and preprocessing empirical data, calculating valence scores,
and generating trial templates.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from nes_copilot.module_base import ModuleBase


class DataPrepModule(ModuleBase):
    """
    Data preparation module for the NES Copilot system.
    
    Handles loading and preprocessing empirical data, calculating valence scores,
    and generating trial templates.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the data preparation module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        super().__init__(config_manager, data_manager, logging_manager)
        
        # Get module configuration
        self.config = self.config_manager.get_module_config('data_prep')
        
    def run(self, raw_data_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run the data preparation module.
        
        Args:
            raw_data_path: Optional override for the raw data path in config.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the results of the data preparation.
        """
        self.logger.info("Starting data preparation")
        
        # Validate inputs
        self.validate_inputs(raw_data_path=raw_data_path)
        
        # Load and preprocess empirical data
        empirical_data = self.load_and_preprocess_data(raw_data_path)
        self.logger.info(f"Loaded and preprocessed empirical data: {len(empirical_data)} rows")
        
        # Calculate valence scores
        valence_scores = self.calculate_valence_scores(empirical_data)
        self.logger.info(f"Calculated valence scores for {len(valence_scores)} stimuli")
        
        # Generate trial template
        trial_template = self.generate_trial_template(empirical_data, valence_scores)
        self.logger.info(f"Generated trial template with {len(trial_template)} trials")
        
        # Save outputs
        output_paths = self.save_outputs({
            'empirical_data': empirical_data,
            'valence_scores': valence_scores,
            'trial_template': trial_template
        })
        
        # Return results
        results = {
            'num_subjects': empirical_data['subject'].nunique(),
            'num_trials': len(empirical_data),
            'num_template_trials': len(trial_template),
            'output_paths': output_paths
        }
        
        self.logger.info("Data preparation completed successfully")
        return results
        
    def validate_inputs(self, raw_data_path: Optional[str] = None, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            raw_data_path: Optional override for the raw data path in config.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        # Get the raw data path from config if not provided
        if not raw_data_path:
            raw_data_path = self.data_manager.get_input_path('roberts_data')
            
        # Check if the raw data file exists
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
            
        return True
        
    def load_and_preprocess_data(self, raw_data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess the empirical data.
        
        Args:
            raw_data_path: Optional override for the raw data path in config.
            
        Returns:
            Preprocessed DataFrame.
        """
        # Get the raw data path from config if not provided
        if not raw_data_path:
            raw_data_path = self.data_manager.get_input_path('roberts_data')
            
        # Load the raw data
        self.logger.info(f"Loading raw data from {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        
        # Apply filtering criteria
        filtering_config = self.config.get('filtering', {})
        trial_type = filtering_config.get('trial_type', 'target')
        exclude_subjects = filtering_config.get('exclude_subjects', [])
        
        # Filter by trial type
        if trial_type:
            df = df[df['trialType'] == trial_type]
            
        # Exclude subjects
        if exclude_subjects:
            df = df[~df['subject'].isin(exclude_subjects)]
            
        # Add derived columns
        df['is_gain_frame'] = df['frame'] == 'gain'
        df['time_constrained'] = df['cond'] == 'tc'
        
        return df
        
    def calculate_valence_scores(self, empirical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate valence scores for stimuli using sentiment analysis.
        
        Args:
            empirical_data: Preprocessed empirical data.
            
        Returns:
            Dict mapping stimuli to valence scores.
        """
        # Get valence processor configuration
        valence_config = self.config.get('valence_processor', {})
        model_name = valence_config.get('model_name', 'cardiffnlp/twitter-roberta-base-sentiment')
        rescaling = valence_config.get('rescaling', {'min_value': -1.0, 'max_value': 1.0})
        
        # Extract unique stimuli
        # Note: In a real implementation, we would extract the actual stimulus text
        # For now, we'll use a placeholder approach based on frame and condition
        stimuli = empirical_data[['frame', 'cond']].drop_duplicates()
        
        # Initialize sentiment model
        self.logger.info(f"Loading sentiment model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Calculate valence scores
        valence_scores = {}
        for _, row in stimuli.iterrows():
            # Generate a placeholder stimulus text
            stimulus = f"This is a {row['frame']} frame with {row['cond']} condition"
            
            # Calculate sentiment
            inputs = tokenizer(stimulus, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Get sentiment scores
            scores = outputs.logits.softmax(dim=1).numpy()[0]
            
            # Convert to a single valence score (negative to positive)
            # RoBERTa sentiment model typically outputs [negative, neutral, positive]
            raw_valence = scores[2] - scores[0]  # positive - negative
            
            # Rescale to the desired range
            min_val = rescaling.get('min_value', -1.0)
            max_val = rescaling.get('max_value', 1.0)
            rescaled_valence = min_val + (raw_valence + 1) / 2 * (max_val - min_val)
            
            # Store the valence score
            key = f"{row['frame']}_{row['cond']}"
            valence_scores[key] = rescaled_valence
            
        return valence_scores
        
    def generate_trial_template(self, empirical_data: pd.DataFrame, valence_scores: Dict[str, float]) -> pd.DataFrame:
        """
        Generate a trial template for simulations.
        
        Args:
            empirical_data: Preprocessed empirical data.
            valence_scores: Dict mapping stimuli to valence scores.
            
        Returns:
            DataFrame containing the trial template.
        """
        # Get trial template configuration
        template_config = self.config.get('trial_template', {})
        num_template_trials = template_config.get('num_template_trials', 1000)
        
        # Set random seed for reproducibility
        seed = self.config_manager.get_param('seed', 12345)
        np.random.seed(seed)
        
        # Sample trials from empirical data
        sampled_indices = np.random.choice(
            empirical_data.index,
            size=num_template_trials,
            replace=True
        )
        template = empirical_data.loc[sampled_indices].copy()
        
        # Add valence scores
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
        
        return template
        
    def save_outputs(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save module outputs to disk using the data manager.
        
        Args:
            results: Results to save.
            
        Returns:
            Dict mapping output names to their paths.
        """
        output_paths = {}
        
        # Save preprocessed empirical data
        empirical_data_path = self.data_manager.get_output_path('data_prep', 'empirical_data.csv')
        results['empirical_data'].to_csv(empirical_data_path, index=False)
        output_paths['empirical_data'] = empirical_data_path
        
        # Save valence scores
        valence_scores_path = self.data_manager.get_output_path('data_prep', 'valence_scores.json')
        self.data_manager.save_json(results['valence_scores'], 'data_prep', 'valence_scores.json')
        output_paths['valence_scores'] = valence_scores_path
        
        # Save trial template
        template_config = self.config.get('trial_template', {})
        save_path = template_config.get('save_path', 'trial_template.csv')
        trial_template_path = self.data_manager.get_output_path('data_prep', save_path)
        results['trial_template'].to_csv(trial_template_path, index=False)
        output_paths['trial_template'] = trial_template_path
        
        return output_paths
