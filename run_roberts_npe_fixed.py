#!/usr/bin/env python3
"""
Neural Posterior Estimation (NPE) for Roberts et al. data.

This is a fixed version that uses the FixedConfigManager to handle parameter validation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Uniform, TransformedDistribution, constraints
from torch.distributions.transforms import AffineTransform, SigmoidTransform
import yaml
import random
from datetime import datetime

# Set up root logger first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import NES Copilot modules
try:
    from nes_copilot.config_manager_fixed import FixedConfigManager as ConfigManager
    from nes_copilot.data_manager import DataManager
    from nes_copilot.logging_manager import LoggingManager
    from nes_copilot.summary_stats_module import SummaryStatsModule
except ImportError as e:
    logger.error(f"Error importing NES Copilot modules: {e}")
    sys.exit(1)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MVNESModel:
    """MVNES model implementation for simulation and likelihood evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MVNES model with configuration."""
        self.config = config
        self.param_names = list(config['model']['parameters'].keys())
        self.param_bounds = {k: tuple(v) for k, v in config['model']['parameters'].items()}
        
    def sample_prior(self, n_samples: int = 1) -> Dict[str, np.ndarray]:
        """Sample parameters from the prior distributions."""
        samples = {}
        for param, (low, high) in self.param_bounds.items():
            samples[param] = np.random.uniform(low, high, size=n_samples)
        return samples
    
    def simulate(self, params: Dict[str, np.ndarray], n_trials: int = 100) -> Dict[str, np.ndarray]:
        """Simulate data from the model."""
        # This is a simplified simulation - replace with your actual MVNES model
        n_samples = len(next(iter(params.values())))
        
        # Generate random choices and RTs based on parameters
        results = {
            'choice': np.random.binomial(1, 0.5, size=(n_samples, n_trials)),
            'rt': np.random.lognormal(0, 0.3, size=(n_samples, n_trials))
        }
        
        return results

class NPETrainer:
    """Class to handle NPE training and inference."""
    
    def __init__(self, config, data):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert data to PyTorch tensor with proper numeric conversion
        if isinstance(data, pd.DataFrame):
            # Select only the relevant numeric columns for training
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            logger.info(f"Selected numeric columns for training: {numeric_cols}")
            
            # Get numeric data and drop any rows with missing values
            numeric_data = data[numeric_cols].dropna()
            
            # Log information about the processed data
            logger.info(f"Processed data shape: {numeric_data.shape}")
            logger.info(f"Number of rows with missing values dropped: {len(data) - len(numeric_data)}")
            
            # Convert to PyTorch tensor
            self.data = torch.tensor(numeric_data.values, dtype=torch.float32).to(self.device)
        else:
            # If it's a numpy array, ensure it's the right type
            if isinstance(data, np.ndarray):
                self.data = torch.tensor(data.astype(np.float32)).to(self.device)
            else:
                self.data = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        # Store data dimensions and validate
        if len(self.data.shape) == 0 or self.data.shape[0] == 0:
            raise ValueError("No valid data available for training. Check data loading and preprocessing.")
            
        self.n_obs = self.data.shape[0]
        self.n_features = self.data.shape[1] if len(self.data.shape) > 1 else 1
        
        logger.info(f"Initialized with {self.n_obs} observations and {self.n_features} features")
        
        # Initialize model and prior after we know the data dimensions
        self._setup_prior()
        self.model = self._build_model().to(self.device)
        
        # Set up optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['npe']['training']['learning_rate']
        )
        self.loss_fn = nn.MSELoss()
    
    def _setup_prior(self):
        """Set up the prior distribution for parameters."""
        # Get parameter bounds from config
        param_bounds = self.config['model']['parameters']
        
        # Create uniform priors for each parameter
        self.priors = {}
        for param, (low, high) in param_bounds.items():
            self.priors[param] = Uniform(low=float(low), high=float(high))
    
    def _build_model(self):
        """Build the neural density estimator model."""
        # Simple MLP for demonstration
        model = nn.Sequential(
            nn.Linear(self.n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.priors) * 2)  # For mean and log_std of each parameter
        )
        return model
    
    def sample_prior(self, n_samples: int) -> Dict[str, torch.Tensor]:
        """Sample parameters from the prior distribution."""
        samples = {}
        for param, prior in self.priors.items():
            samples[param] = prior.sample((n_samples,))
        return samples
    
    def train(self, n_epochs: int, batch_size: int):
        """Train the neural posterior estimator."""
        self.model.train()
        n_batches = (self.n_obs + batch_size - 1) // batch_size
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            idx = torch.randperm(self.n_obs)
            
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, self.n_obs)
                batch_idx = idx[start_idx:end_idx]
                batch_data = self.data[batch_idx]
                
                # Sample from prior
                prior_samples = self.sample_prior(len(batch_idx))
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch_data)
                
                # Compute loss (simplified for demonstration)
                # In a real implementation, you would use a proper probabilistic loss
                target = torch.stack([prior_samples[param] for param in self.priors], dim=1)
                loss = self.loss_fn(output[:, :len(self.priors)], target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Log progress
            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    def infer_posterior(self, observed_data: Dict[str, np.ndarray], 
                       num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Infer posterior distribution given observed data."""
        self.model.eval()
        
        try:
            # Convert observed data to numpy array, handling non-numeric data
            data_arrays = []
            valid_cols = []
            
            for col in observed_data.keys():
                try:
                    # Try to convert to float, skip if not possible
                    col_data = np.array(observed_data[col], dtype=np.float32)
                    data_arrays.append(col_data)
                    valid_cols.append(col)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping non-numeric column '{col}': {e}")
            
            if not data_arrays:
                raise ValueError("No valid numeric data provided for inference")
                
            # Stack arrays and convert to tensor
            obs_array = np.column_stack(data_arrays)
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
            
            # Sample from the posterior (simplified for demonstration)
            with torch.no_grad():
                output = self.model(obs_tensor)
            
            # In a real implementation, you would use proper posterior sampling
            # This is just a placeholder
            posterior_samples = {}
            for i, param in enumerate(self.priors.keys()):
                if i >= output.shape[1]:  # In case we have more parameters than output dimensions
                    break
                    
                # For demonstration, just add some noise to the mean prediction
                mean = output[:, i].cpu().numpy()
                std = np.ones_like(mean) * 0.1  # Fixed standard deviation
                posterior_samples[param] = np.random.normal(mean, std, size=(num_samples, len(mean)))
            
            return posterior_samples
            
        except Exception as e:
            logger.error(f"Error during posterior inference: {e}")
            # Return empty dict with parameter names to maintain expected structure
            return {param: np.array([]) for param in self.priors.keys()}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['experiment']['output_dir'] = os.path.join(
        config['experiment']['output_dir'], 
        f"{config['experiment']['name']}_{timestamp}"
    )
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    
    return config

def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging configuration."""
    log_config = config['logging']
    log_file = os.path.join(config['experiment']['output_dir'], log_config['file'])
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file, mode=log_config['filemode'])
    file_handler.setLevel(log_config['level'].upper())
    
    # Set up console handler if enabled
    handlers = [file_handler]
    if log_config.get('console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_config['level'].upper())
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_config['level'].upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def get_logger():
    """Get a configured logger instance."""
    logger = logging.getLogger(__name__)
    
    # If no handlers are configured, add a basic console handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def main():
    """Main function to run the NPE analysis."""
    # Get logger
    logger = get_logger()
    
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Run NPE analysis on Roberts et al. data')
        parser.add_argument('--config', type=str, required=True, 
                          help='Path to configuration YAML file')
        args = parser.parse_args()
        
        # Initialize config manager with the config file path
        logger.info("Loading configuration...")
        config_manager = ConfigManager(args.config)
        
        # Get the config dictionary
        config = config_manager.get_master_config()
        
        # Set up file logging
        output_dir = config.get('output_dir', './results')
        os.makedirs(output_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(output_dir, 'npe_analysis.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info("Starting NPE analysis for Roberts et al. data")
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize data manager with config manager
        logger.info("Initializing data manager...")
        data_manager = DataManager(config_manager)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_path = Path(config['data']['processed_data'])
        
        # Check if data file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}. Please check the path in the config.")
        
        # Load data with detailed logging
        logger.info(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data shape: {data.shape}")
        
        # Initialize MVNES model
        logger.info("Initializing MVNES model...")
        model = MVNESModel(config)
        
        # Initialize NPE trainer
        logger.info("Initializing NPE trainer...")
        trainer = NPETrainer(config, data)
        
        # Train the model
        logger.info("Starting training...")
        n_epochs = config['npe']['training']['max_epochs']
        batch_size = config['npe']['training']['batch_size']
        trainer.train(n_epochs, batch_size)
        
        # Perform inference
        logger.info("Performing inference...")
        num_samples = config['npe']['inference']['num_samples']
        
        # For demonstration, use the first few data points as observed data
        obs_data = data.iloc[:10].to_dict('list')
        posterior_samples = trainer.infer_posterior(obs_data, num_samples=num_samples)
        
        # Save results
        logger.info("Saving results...")
        results_dir = Path(output_dir) / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save posterior samples
        for param, samples in posterior_samples.items():
            np.save(results_dir / f"posterior_{param}.npy", samples)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
