"""
Valence Processor for NES Copilot

This module handles calculating valence scores for stimuli using RoBERTa sentiment analysis
with the following transformation pipeline:
1. Get raw logits from RoBERTa
2. Apply tanh activation
3. Mean-center the scores
4. Rescale by maximum absolute value
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class ValenceProcessor:
    """
    Valence processor for the NES Copilot system.
    
    Handles calculating valence scores for stimuli using RoBERTa sentiment analysis
    with the specified transformation pipeline.
    """
    
    def __init__(self, model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment',
                 device: Optional[str] = None):
        """
        Initialize the valence processor.
        
        Args:
            model_name: Name of the pre-trained RoBERTa model to use.
            device: Device to run the model on ('cuda', 'mps', or 'cpu').
                   If None, will use CUDA if available, then MPS, then CPU.
        """
        # Determine device
        if device is None:
            self.device = (
                'cuda' if torch.cuda.is_available() else
                'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = device
            
        # Load model and tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=self.config
        ).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize statistics for normalization
        self.scores_history = []
        self.mean_valence = 0.0
        self.max_abs_valence = 1.0  # Initial value, will be updated
        
    def _process_single_text(self, text: str) -> float:
        """
        Process a single text through the RoBERTa model and return raw logits.
        
        Args:
            text: Input text to process
            
        Returns:
            Raw logits from the model
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False, output_attentions=False)
            
        # Return logits (before softmax)
        return outputs.logits.cpu().numpy()[0]
    
    def _transform_scores(self, logits: np.ndarray) -> float:
        """
        Apply the transformation pipeline to model logits.
        
        Transformation pipeline:
        1. Apply tanh to logits
        2. Mean-center the scores
        3. Rescale by maximum absolute value
        
        Args:
            logits: Raw logits from the model
            
        Returns:
            Transformed valence score
        """
        # 1. Apply tanh to logits
        tanh_scores = np.tanh(logits)
        
        # For sentiment analysis, we typically want the difference between positive and negative
        # Assuming logits are [negative, neutral, positive]
        if len(tanh_scores) >= 3:
            # Use positive - negative as the valence signal
            raw_valence = tanh_scores[2] - tanh_scores[0]
        else:
            # Fallback if logits structure is different
            raw_valence = tanh_scores[0] if len(tanh_scores) > 0 else 0.0
        
        # 2. Mean-center the score (using running mean if available)
        mean_centered = raw_valence - self.mean_valence
        
        # 3. Rescale by maximum absolute value (avoid division by zero)
        if self.max_abs_valence > 1e-6:
            rescaled = mean_centered / self.max_abs_valence
        else:
            rescaled = mean_centered
            
        # Ensure score is in [-1, 1] range
        return np.clip(rescaled, -1.0, 1.0).item()
    
    def update_normalization_stats(self, scores: List[float]):
        """
        Update the running statistics for score normalization.
        
        Args:
            scores: List of new scores to incorporate into statistics
        """
        if not scores:
            return
            
        # Update history
        self.scores_history.extend(scores)
        
        # Update mean and max absolute value
        if self.scores_history:
            scores_array = np.array(self.scores_history)
            self.mean_valence = float(np.mean(scores_array))
            self.max_abs_valence = max(1e-6, float(np.max(np.abs(scores_array - self.mean_valence))))
    
    def calculate_valence_scores(self, texts: Union[str, List[str]], 
                               update_stats: bool = True) -> Union[float, Dict[str, float]]:
        """
        Calculate valence scores for one or more texts.
        
        Args:
            texts: Single text or list of texts to score
            update_stats: Whether to update running statistics with new scores
            
        Returns:
            For a single text: float valence score
            For multiple texts: dict mapping texts to their valence scores
        """
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]
            
        # Process all texts
        logits_list = [self._process_single_text(text) for text in texts]
        raw_scores = [self._transform_scores(logits) for logits in logits_list]
        
        # Update normalization statistics if requested
        if update_stats and raw_scores:
            self.update_normalization_stats(raw_scores)
            
        # Apply final transformation with updated stats
        final_scores = [self._transform_scores(logits) for logits in logits_list]
        
        # Return in appropriate format
        if single_text:
            return final_scores[0]
        else:
            return {text: score for text, score in zip(texts, final_scores)}
    
    def get_trial_valence_scores(self, df, text_column: str = 'text', 
                               id_columns: Optional[List[str]] = None) -> dict:
        """
        Calculate valence scores for trials in a DataFrame.
        
        Args:
            df: DataFrame containing trial data
            text_column: Name of the column containing the text to score
            id_columns: Columns that uniquely identify each trial
            
        Returns:
            Dictionary mapping trial IDs to valence scores
        """
        if id_columns is None:
            id_columns = ['trial_id']
            
        # Get unique texts and their mappings to trial IDs
        unique_texts = df[text_column].unique().tolist()
        
        # Calculate valence scores for all unique texts
        text_to_score = self.calculate_valence_scores(unique_texts)
        
        # Create a mapping from trial ID to valence score
        trial_scores = {}
        for _, row in df.iterrows():
            trial_id = tuple(row[col] for col in id_columns)
            if len(trial_id) == 1:
                trial_id = trial_id[0]  # Simplify if only one ID column
            trial_scores[trial_id] = text_to_score[row[text_column]]
            
        return trial_scores
