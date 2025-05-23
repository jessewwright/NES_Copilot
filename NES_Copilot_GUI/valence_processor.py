"""
Valence Processor for NES Copilot

This module handles calculating valence scores for stimuli using sentiment analysis.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any, List, Optional, Union


class ValenceProcessor:
    """
    Valence processor for the NES Copilot system.
    
    Handles calculating valence scores for stimuli using sentiment analysis.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the valence processor.
        
        Args:
            config: Configuration dictionary.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        
        # Get valence processor configuration
        valence_config = self.config.get('valence_processor', {})
        self.model_name = valence_config.get('model_name', 'cardiffnlp/twitter-roberta-base-sentiment')
        self.rescaling = valence_config.get('rescaling', {'min_value': -1.0, 'max_value': 1.0})
        
        # Initialize sentiment model
        self.logger.info(f"Loading sentiment model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
    def calculate_valence_scores(self, stimuli: List[str]) -> Dict[str, float]:
        """
        Calculate valence scores for stimuli using sentiment analysis.
        
        Args:
            stimuli: List of stimulus texts.
            
        Returns:
            Dict mapping stimuli to valence scores.
        """
        valence_scores = {}
        
        for stimulus in stimuli:
            # Calculate sentiment
            inputs = self.tokenizer(stimulus, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get sentiment scores
            scores = outputs.logits.softmax(dim=1).numpy()[0]
            
            # Convert to a single valence score (negative to positive)
            # RoBERTa sentiment model typically outputs [negative, neutral, positive]
            raw_valence = scores[2] - scores[0]  # positive - negative
            
            # Rescale to the desired range
            min_val = self.rescaling.get('min_value', -1.0)
            max_val = self.rescaling.get('max_value', 1.0)
            rescaled_valence = min_val + (raw_valence + 1) / 2 * (max_val - min_val)
            
            # Store the valence score
            valence_scores[stimulus] = rescaled_valence
            
        self.logger.info(f"Calculated valence scores for {len(valence_scores)} stimuli")
        return valence_scores
        
    def get_trial_valence_scores(self, df):
        """
        Calculate valence scores for trials in a DataFrame.
        
        Args:
            df: DataFrame containing trials.
            
        Returns:
            DataFrame with added valence_score column.
        """
        # Extract unique combinations of frame and condition
        frame_cond_pairs = df[['frame', 'cond']].drop_duplicates()
        
        # Generate placeholder stimulus texts
        stimuli = []
        for _, row in frame_cond_pairs.iterrows():
            stimulus = f"This is a {row['frame']} frame with {row['cond']} condition"
            stimuli.append(stimulus)
            
        # Calculate valence scores
        valence_scores = self.calculate_valence_scores(stimuli)
        
        # Create a mapping from frame-condition pairs to valence scores
        frame_cond_to_valence = {}
        for i, (_, row) in enumerate(frame_cond_pairs.iterrows()):
            key = f"{row['frame']}_{row['cond']}"
            frame_cond_to_valence[key] = valence_scores[stimuli[i]]
            
        # Add valence scores to the DataFrame
        df_with_valence = df.copy()
        df_with_valence['valence_score'] = df_with_valence.apply(
            lambda row: frame_cond_to_valence.get(f"{row['frame']}_{row['cond']}", 0.0),
            axis=1
        )
        
        return df_with_valence
