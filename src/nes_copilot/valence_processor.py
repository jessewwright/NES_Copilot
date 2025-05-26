"""
Valence Processor for NES Copilot

This module handles calculating valence scores for stimuli using RoBERTa sentiment analysis
with the following transformation pipeline:
1. Generate rich text descriptions for trial configurations
2. Get raw logits from RoBERTa
3. Apply tanh activation
4. Mean-center the scores
5. Rescale by maximum absolute value
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# Set up logging
logger = logging.getLogger(__name__)

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
        
    def _generate_rich_trial_text(self, row: pd.Series) -> str:
        """
        Generate a rich text description for a trial based on its parameters.
        
        Args:
            row: A pandas Series containing trial parameters
            
        Returns:
            A string describing the trial in natural language
        """
        # Extract parameters with defaults for safety
        frame = row.get('frame', 'gain').lower()
        cond = row.get('cond', 'ntc').lower()
        prob = float(row.get('prob', 0.5))
        sure_outcome = float(row.get('sureOutcome', 0.0))
        endow = float(row.get('endow', 0.0))
        
        # Format numbers for display
        prob_pct = f"{prob*100:.0f}%"
        sure_amt = f"${abs(sure_outcome):.2f}"
        gamble_amt = f"${abs(endow):.2f}"
        
        # Generate text based on frame and condition
        if frame == 'gain':
            if cond == 'tc':
                # Time-constrained gain frame
                text = (
                    f"Option A: Receive {sure_amt} for sure. "
                    f"Option B: A gamble with a {prob_pct} chance to win {gamble_amt} "
                    f"and a {100-prob*100:.0f}% chance to win nothing. "
                    "You must decide quickly which option to choose."
                )
            else:  # NTC
                # Non-time-constrained gain frame
                text = (
                    f"You can choose to either receive {sure_amt} for sure, "
                    f"or take a gamble with a {prob_pct} chance to win {gamble_amt} "
                    f"and a {100-prob*100:.0f}% chance to win nothing. "
                    "Please consider your options carefully before deciding."
                )
        else:  # loss frame
            if cond == 'tc':
                # Time-constrained loss frame
                text = (
                    f"Option A: Lose {sure_amt} for sure. "
                    f"Option B: A gamble with a {prob_pct} chance to lose nothing "
                    f"and a {100-prob*100:.0f}% chance to lose {gamble_amt}. "
                    "You must decide quickly which option to choose."
                )
            else:  # NTC
                # Non-time-constrained loss frame
                text = (
                    f"You must choose between losing {sure_amt} for sure, "
                    f"or taking a gamble with a {prob_pct} chance to lose nothing "
                    f"and a {100-prob*100:.0f}% chance to lose {gamble_amt}. "
                    "Please consider your options carefully before deciding."
                )
                
        return text
    
    def get_trial_valence_scores_for_df(self, df_trials: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate valence scores for trials in a DataFrame and add them as a new column.
        
        This function:
        1. Generates rich text descriptions for each unique trial configuration
        2. Processes these texts through RoBERTa sentiment analysis
        3. Applies the transformation pipeline to the scores
        4. Maps the scores back to the original DataFrame
        
        Args:
            df_trials: DataFrame containing trial data with columns:
                     - frame: 'gain' or 'loss'
                     - cond: 'tc' or 'ntc'
                     - prob: probability (0-1)
                     - sureOutcome: sure outcome amount
                     - endow: endowment amount for gamble
                     
        Returns:
            DataFrame with an additional 'valence_score' column
            
        Raises:
            ValueError: If required columns are missing from the input DataFrame
        """
        # Make a copy to avoid modifying the input
        df = df_trials.copy()
        
        # Check for required columns
        required_columns = ['frame', 'cond', 'prob', 'sureOutcome', 'endow']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
            
        # Add a temporary unique ID for each row to handle duplicates
        df['_temp_id'] = range(len(df))
        
        try:
            # Step 1: Generate rich text descriptions for each unique trial configuration
            logger.info("Generating rich text descriptions for trial configurations...")
            
            # Create a unique key for each configuration to avoid redundant processing
            df['_config_key'] = df.apply(
                lambda x: f"{x['frame']}_{x['cond']}_{x['prob']:.4f}_{x['sureOutcome']:.2f}_{x['endow']:.2f}", 
                axis=1
            )
            
            # Get unique configurations and their rich texts
            unique_configs = df[['_config_key'] + required_columns].drop_duplicates()
            unique_configs['rich_text'] = unique_configs.apply(self._generate_rich_trial_text, axis=1)
            
            # Map rich texts back to original DataFrame
            text_mapping = dict(zip(unique_configs['_config_key'], unique_configs['rich_text']))
            df['rich_text'] = df['_config_key'].map(text_mapping)
            
            # Step 2: Process texts through RoBERTa and get valence scores
            logger.info(f"Processing {len(unique_configs)} unique trial configurations through RoBERTa...")
            
            # Get scores for unique texts
            unique_texts = unique_configs['rich_text'].tolist()
            try:
                # First pass to get initial scores
                text_scores = self.calculate_valence_scores(unique_texts, update_stats=True)
                
                # Second pass with updated statistics
                text_scores = self.calculate_valence_scores(unique_texts, update_stats=False)
                
                # Create mapping from text to score
                text_to_score = dict(zip(unique_texts, text_scores))
                
                # Map scores back to configurations
                unique_configs['valence_score'] = unique_configs['rich_text'].map(text_to_score)
                
                # Create mapping from config key to score
                score_mapping = dict(zip(unique_configs['_config_key'], unique_configs['valence_score']))
                
                # Add scores to original DataFrame
                df['valence_score'] = df['_config_key'].map(score_mapping)
                
                # Log statistics about the generated scores
                scores = df['valence_score'].dropna()
                if not scores.empty:
                    logger.info(f"Generated {len(scores)} valence scores with:"
                               f" min={scores.min():.4f}, max={scores.max():.4f}, "
                               f"mean={scores.mean():.4f}, std={scores.std():.4f}")
                else:
                    logger.warning("No valid valence scores were generated")
                
            except Exception as e:
                logger.error(f"Error calculating valence scores: {str(e)}")
                # Fallback: Use neutral scores if calculation fails
                df['valence_score'] = 0.0
                
        except Exception as e:
            logger.error(f"Error in get_trial_valence_scores_for_df: {str(e)}")
            raise
            
        finally:
            # Clean up temporary columns
            df.drop(columns=['_temp_id', '_config_key'], errors='ignore')
            
        return df
