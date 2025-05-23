#!/usr/bin/env python3
"""
Neural Posterior Estimation (NPE) for Roberts et al. data.

This script performs Neural Posterior Estimation (NPE) on the Roberts et al. dataset
using the MVNES model. It includes data loading, simulation, training, and inference.
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
    from nes_copilot.config_manager import ConfigManager
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
        # Create a dictionary of parameter bounds
        self.param_bounds = {}
        for param_name, (low, high) in self.config['model']['parameters'].items():
            self.param_bounds[param_name] = (low, high)
        
        self.param_names = list(self.param_bounds.keys())
        self.n_params = len(self.param_names)
        
    def _build_model(self) -> nn.Module:
        """Build the neural density estimator model."""
        n_hidden = 128
        n_params = self.n_params
        n_features = self.n_features  # Number of features in the data
        
        # The model will take parameters as input and output the parameters of a distribution over the data
        model = nn.Sequential(
            nn.Linear(n_params, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            # Output both mean and log_std for each feature in the data
            nn.Linear(n_hidden, n_features * 2)  # For mean and log_std for each feature
        )
        return model
        
    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """Sample parameters from the prior distribution."""
        samples = []
        for param_name in self.param_names:
            low, high = self.param_bounds[param_name]
            # Sample from uniform prior
            samples.append(torch.rand(n_samples) * (high - low) + low)
        return torch.stack(samples, dim=1).to(self.device)
    
    def train(self, n_epochs: int, batch_size: int) -> List[float]:
        """Train the neural posterior estimator."""
        self.model.train()
        losses = []
        
        if self.n_obs == 0:
            raise ValueError("Cannot train with zero observations. Check data loading and preprocessing.")
            
        # Ensure batch_size is not larger than number of observations
        batch_size = min(batch_size, self.n_obs)
        
        logger.info(f"Starting training for {n_epochs} epochs with batch size {batch_size}")
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            # Create batches
            indices = torch.randperm(self.n_obs)
            n_batches = (self.n_obs + batch_size - 1) // batch_size  # Ceiling division
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.n_obs)
                batch_indices = indices[start_idx:end_idx]
                batch_data = self.data[batch_indices]
                
                # Sample from prior
                params = self.sample_prior(len(batch_indices))
                
                # Forward pass
                outputs = self.model(params)
                
                # Split outputs into mean and log_std for each feature
                # The first half is mean, second half is log_std
                mean, log_std = torch.chunk(outputs, 2, dim=1)
                std = torch.exp(log_std) + 1e-6  # Add small constant for numerical stability
                
                # Create normal distribution and calculate log probability
                # We need to ensure the shapes match: [batch_size, n_features]
                dist = torch.distributions.Normal(mean, std)
                
                # Ensure batch_data is the right shape: [batch_size, n_features]
                if len(batch_data.shape) == 1:
                    batch_data = batch_data.unsqueeze(1)
                
                # Calculate log probability for each feature and sum across features
                log_prob = dist.log_prob(batch_data).sum(dim=1, keepdim=True)
                
                # Negative log likelihood loss
                loss = -log_prob.mean()
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / self.n_obs
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
                
        return losses
    
    def infer_posterior(self, observed_data: Dict[str, np.ndarray], 
                        num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Infer posterior distribution given observed data."""
        self.model.eval()
        
        # Calculate summary statistics of observed data
        # TODO: Implement summary statistics calculation
        
        # Sample from posterior
        with torch.no_grad():
            posterior_samples = {}
            # TODO: Implement posterior sampling
            
        return posterior_samples

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
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create output directory if it doesn't exist
    output_dir = config.get('output_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up file handler
    log_file = os.path.join(output_dir, 'npe_analysis.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the handlers to the root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def get_logger():
    """Get a configured logger instance."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Only add handlers if they don't exist
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
        # Set up paths
        config_path = "config/roberts_npe_config.yaml"
        
        # Initialize config manager with the config file path
        logger.info("Loading configuration...")
        config_manager = ConfigManager(config_path)
        
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
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        raise
    
    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_path = Path(config['data']['processed_data'])
        
        # Check if data file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}. Please check the path in the config.")
        
        # Load data with detailed logging
        logger.info(f"Loading data from {data_path}...")
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data with shape: {data.shape}")
            
            # Log column information
            logger.info(f"Data columns: {data.columns.tolist()}")
            logger.info(f"Data types:\n{data.dtypes}")
            
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                logger.warning(f"Missing values found in data:\n{missing_values[missing_values > 0]}")
            else:
                logger.info("No missing values found in the data")
                
            # Ensure we have data to work with
            if len(data) == 0:
                raise ValueError("Loaded data is empty. Please check the data file.")
                
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise
        
        # Initialize model and trainer
        logger.info("Initializing MVNES model and NPE trainer...")
        model = MVNESModel(config)
        trainer = NPETrainer(config, data)
        
        # Train the model
        logger.info("Starting NPE training...")
        losses = trainer.train(
            n_epochs=config['npe']['training']['max_epochs'],
            batch_size=config['npe']['training']['batch_size']
        )
        
        # Save training results
        logger.info("Saving training results...")
        # Use the output_dir from the config root or default to './results'
        results_dir = Path(config.get('output_dir', './results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training losses
        loss_file = results_dir / 'training_losses.npy'
        np.save(loss_file, np.array(losses))
        logger.info(f"Saved training losses to {loss_file}")
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        
        # Save the plot
        plot_file = results_dir / 'training_curve.png'
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Saved training curve to {plot_file}")
        
        # Run inference on observed data
        logger.info("Running inference on observed data...")
        posterior_samples = trainer.infer_posterior(
            data,
            num_samples=config['npe']['inference']['num_samples']
        )
        
        # Save posterior samples
        logger.info("Saving posterior samples...")
        posterior_df = pd.DataFrame(posterior_samples)
        posterior_df.to_csv(results_dir / 'posterior_samples.csv', index=False)
        
        logger.info("NPE analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during NPE analysis: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
