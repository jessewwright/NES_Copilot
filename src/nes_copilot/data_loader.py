"""
Data Loader for NES Copilot

This module handles loading and preprocessing empirical data.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional, Union


class DataLoader:
    """
    Data loader for the NES Copilot system.
    
    Handles loading and preprocessing empirical data.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess the empirical data.
        
        Args:
            data_path: Path to the raw data file.
            
        Returns:
            Preprocessed DataFrame.
        """
        # Check if the data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Load the raw data
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
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
        
        self.logger.info(f"Preprocessed data: {len(df)} rows, {df['subject'].nunique()} subjects")
        
        return df
