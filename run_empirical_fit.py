#!/usr/bin/env python
"""
Script for running empirical fitting with a pre-trained NPE model.

This script loads a pre-trained NPE model, processes empirical data, calculates
summary statistics, and performs posterior inference to estimate model parameters.
"""

import argparse
import json
import logging
import os
import sys
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Any
import json
import numpy as np

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nflows')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.int64)) else k: convert_numpy_types(v) 
                for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj]
    return obj
import pandas as pd
import torch
from tqdm import tqdm
import sbi.utils as sbi_utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from nes_copilot.checkpoint_manager import CheckpointManager
from nes_copilot.valence_processor import ValenceProcessor
from nes_copilot.summary_stats_module import SummaryStatsModule, ROBERTS_SUMMARY_STAT_KEYS

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Only show WARNING and above
    format='%(message)s',  # Simpler format
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('empirical_fitting.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure pandas and tqdm
pd.options.mode.chained_assignment = None

# Configure tqdm for clean progress bars
try:
    tqdm._instances.clear()
except:
    pass  # If clearing fails, continue anyway

def load_npe_posterior(npe_file_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Loads a pre-trained NPE density estimator and builds a posterior object.
    
    Args:
        npe_file_path: Path to the checkpoint file
        device: Device to load the model onto ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing the posterior and other components
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading NPE posterior from {npe_file_path}")
    
    try:
        # Import necessary modules
        from sbi.inference import SNPE_C as SNPE
        from sbi.utils import BoxUniform
        import torch.nn as nn
        
        # Load the checkpoint
        checkpoint = torch.load(npe_file_path, map_location=device)
        
        if not isinstance(checkpoint, dict) or 'density_estimator_state_dict' not in checkpoint:
            raise ValueError("NPE checkpoint error: Expected dict with 'density_estimator_state_dict'.")
        
        # Extract metadata from checkpoint
        param_names = checkpoint.get('param_names', ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain', 'beta_val'])
        num_summary_stats = checkpoint.get('num_summary_stats')
        summary_stat_keys = checkpoint.get('summary_stat_keys', list(ROBERTS_SUMMARY_STAT_KEYS))
        prior_params = checkpoint.get('prior_params')
        
        # Get density estimator kwargs from checkpoint or use defaults
        density_estimator_build_kwargs = checkpoint.get('density_estimator_build_kwargs', {})
        nsf_kwargs = density_estimator_build_kwargs.get('nsf_specific_kwargs', 
            {'num_transforms': 8, 'hidden_features': 256, 'num_bins': 8, 'num_blocks': 2})
        
        logger.info(f"Parameter names: {param_names}")
        logger.info(f"Number of summary stats: {num_summary_stats}")
        
        # Validate summary stats
        if num_summary_stats is None:
            raise ValueError("'num_summary_stats' missing from checkpoint.")
        
        current_script_num_stats = len(summary_stat_keys) if summary_stat_keys else 0
        if current_script_num_stats != num_summary_stats:
            logger.warning(f"Summary stats mismatch: Checkpoint expects {num_summary_stats} stats, "
                         f"but script has {current_script_num_stats}.")
        
        # Reconstruct Prior
        if prior_params and \
           len(prior_params.get('low', [])) == len(param_names) and \
           len(prior_params.get('high', [])) == len(param_names):
            low_bounds = torch.tensor(prior_params['low'], dtype=torch.float32, device=device)
            high_bounds = torch.tensor(prior_params['high'], dtype=torch.float32, device=device)
            logger.info("Using prior bounds from checkpoint metadata.")
        else:
            logger.warning("Using default prior bounds from script.")
            low_bounds = torch.tensor([0.1] * len(param_names), device=device)
            high_bounds = torch.tensor([10.0] * len(param_names), device=device)
        
        prior = BoxUniform(low=low_bounds, high=high_bounds, device=device)
        
        # Define the embedding net (MLP) - matching checkpoint architecture
        def build_embedding_net():
            return nn.Sequential(
                nn.Linear(num_summary_stats, 60),  # Changed to match checkpoint
                nn.ReLU(),
                nn.Linear(60, 60),  # Changed to match checkpoint
                nn.ReLU(),
                nn.Linear(60, 30),  # Output dimension for embedding
                nn.ReLU()
            )
        
        embedding_net = build_embedding_net().to(device)
        
        # For sbi v0.22.0, create the neural network first
        # Using dimensions that match the checkpoint
        neural_net = sbi_utils.posterior_nn(
            model='nsf',
            hidden_features=60,  # Changed to match checkpoint
            num_transforms=3,    # Adjusted based on error message showing transforms 1,3,5
            num_bins=8,          # Using default from nsf_kwargs
            embedding_net=embedding_net,
            num_components=1,    # Simplified to match checkpoint
            z_score_theta=None,  # Disable automatic z-scoring
            z_score_x=None       # Disable automatic z-scoring
        )
        
        # Initialize SNPE with the neural network
        inference = SNPE(
            prior=prior,
            density_estimator=neural_net,
            device=device,
            show_progress_bars=False  # Disable progress bars
        )
        
        # Create more realistic dummy data
        logger.info("Building NSF network structure with dummy data...")
        num_dummy_samples = 100  # More samples for stable initialization
        dummy_theta = prior.sample((num_dummy_samples,)).to(device)
        dummy_x = torch.randn(num_dummy_samples, num_summary_stats, device=device) * 0.5 + 1.0  # Mean=1.0, std=0.5
        
        # Build the network with a small training step
        _ = inference.append_simulations(dummy_theta, dummy_x).train(
            max_num_epochs=2,
            training_batch_size=32,
            validation_fraction=0.1,
            stop_after_epochs=2,
            show_train_summary=False
        )
        
        # Now load the state dict
        if 'density_estimator_state_dict' in checkpoint:
            # Get the state dict from the checkpoint
            state_dict = checkpoint['density_estimator_state_dict']
            
            # Load the state dict into the built network
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Suppress the verbose state dict loading output
                import contextlib
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                    inference._neural_net.load_state_dict(state_dict, strict=False)
        
        # Build the posterior
        posterior = inference.build_posterior()
        
        # Store components
        npe_components = {
            'posterior': posterior,
            'prior': prior,
            'param_names': param_names,
            'summary_stat_keys': summary_stat_keys,
            'num_summary_stats': num_summary_stats,
            'metadata': {k: v for k, v in checkpoint.items() if k != 'density_estimator_state_dict'}
        }
        
        logger.info(f"NPE loaded successfully. Originally trained with {checkpoint.get('npe_train_sims', 'N/A')} simulations.")
        return npe_components
        
    except Exception as e:
        logger.error(f"Failed to load NPE posterior from {npe_file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the empirical data.
    
    Args:
        data_path: Path to the empirical data CSV file
        
    Returns:
        Preprocessed DataFrame with valence scores
    """
    try:
        logger.info(f"Loading empirical data from {data_path}")
        
        # Load the data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from {data_path}")
        
        # Filter for target trials
        if 'trialType' in df.columns:
            df = df[df['trialType'] == 'target'].copy()
            logger.info(f"Filtered to {len(df)} target trials")
        
        # Initialize valence processor
        valence_processor = ValenceProcessor()
        
        # Process valence scores if needed
        if 'valence_score_trial' not in df.columns:
            logger.info("Calculating valence scores for trials...")
            # This is a placeholder - implement actual valence score calculation
            # based on your trial descriptions
            df['valence_score_trial'] = 0.0  # Placeholder
        
        # Add norm category (default)
        if 'norm_category_for_trial' not in df.columns:
            df['norm_category_for_trial'] = 'default'
        
        # Ensure required columns exist
        required_cols = [
            'subject', 'trialType', 'frame', 'cond', 'chose_gamble',
            'gamble_payoff', 'gamble_loss', 'sure_payoff', 'sure_loss',
            'valence_score_trial', 'norm_category_for_trial'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col in ['valence_score_trial'] else ''
        
        logger.info(f"Preprocessed data with {len(df)} trials")
        return df
        
    except Exception as e:
        logger.error(f"Error loading and preprocessing data: {e}")
        raise

def calculate_summary_stats(
    df: pd.DataFrame,
    summary_stat_keys: List[str],
    subject_id: Optional[str] = None
) -> torch.Tensor:
    """
    Calculate summary statistics for a subject's data.
    
    Args:
        df: DataFrame containing the subject's trial data
        summary_stat_keys: List of summary statistic keys to calculate
        subject_id: Optional subject ID for logging
        
    Returns:
        Tensor of calculated summary statistics
    """
    try:
        logger.info(f"Calculating summary statistics for subject {subject_id or 'all'}")
        
        # Create a simple logging manager if none is provided
        class SimpleLogger:
            def info(self, msg):
                print(f"[INFO] {msg}")
            
            def error(self, msg):
                print(f"[ERROR] {msg}")
                
            def get_logger(self, name):
                # Return self to satisfy the ModuleBase's requirement for a logger
                return self
        
        # Initialize summary stats module with dummy config and simple logger
        summary_module = SummaryStatsModule(None, None, SimpleLogger())
        
        # Calculate summary statistics
        results = summary_module.calculate_summary_stats_roberts(
            df,
            stat_keys=summary_stat_keys
        )
        
        # Convert to tensor
        stats_tensor = torch.tensor(
            [results.get(stat, 0.0) for stat in summary_stat_keys],
            dtype=torch.float32
        )
        
        logger.info(f"Calculated {len(summary_stat_keys)} summary statistics")
        return stats_tensor.unsqueeze(0)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Error calculating summary statistics: {e}")
        raise

def run_inference(
    npe_components: Dict[str, Any],
    x_observed: torch.Tensor,
    num_samples: int = 1000,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run inference using the pre-trained NPE.
    
    Args:
        npe_components: Dictionary containing the NPE components
        x_observed: Observed summary statistics [1, num_stats]
        num_samples: Number of posterior samples to generate
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing posterior samples and diagnostics
    """
    try:
        logger.info(f"Running inference with {num_samples} posterior samples")
        
        # Get posterior and other components
        posterior = npe_components['posterior']
        param_names = npe_components['param_names']
        
        # Ensure x_observed has the right shape [1, num_stats]
        if x_observed.dim() == 1:
            x_observed = x_observed.unsqueeze(0)
        
        # Move to device
        x_observed = x_observed.to(device)
        
        # Sample from the posterior
        with torch.no_grad():
            posterior_samples = posterior.sample(
                (num_samples,),
                x=x_observed,
                show_progress_bars=True
            )
        
        # Move to CPU for further processing
        posterior_samples = posterior_samples.cpu()
        
        # Calculate posterior summaries
        posterior_mean = posterior_samples.mean(dim=0)
        posterior_std = posterior_samples.std(dim=0)
        posterior_median = torch.median(posterior_samples, dim=0).values
        
        logger.info(f"Generated {len(posterior_samples)} posterior samples")
        
        return {
            'samples': posterior_samples.numpy(),
            'mean': posterior_mean.numpy(),
            'std': posterior_std.numpy(),
            'median': posterior_median.numpy(),
            'param_names': param_names
        }
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run empirical fitting with a pre-trained NPE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    # Accept either --npe_file or --npe_checkpoint for backward compatibility
    npe_group = parser.add_mutually_exclusive_group(required=True)
    npe_group.add_argument(
        "--npe_file",
        type=str,
        help="Path to the pre-trained NPE model file (alternative to --npe_checkpoint)"
    )
    npe_group.add_argument(
        "--npe_checkpoint",
        type=str,
        help="Path to the pre-trained NPE checkpoint file (alternative to --npe_file)"
    )
    
    # Data arguments
    parser.add_argument(
        '--data_file',
        type=str,
        default='roberts_framing_data/ftp_osf_data.csv',
        help='Path to empirical data file (CSV). Alias for --roberts_data_file.'
    )
    parser.add_argument(
        '--roberts_data_file',
        type=str,
        default=None,
        help='Path to Roberts framing data file (CSV). Overrides --data_file if both are provided.'
    )
    parser.add_argument(
        '--subject_ids',
        type=str,
        default='',
        help='Comma-separated list of subject IDs to process (default: all)'
    )
    parser.add_argument(
        '--fitted_params_file',
        type=str,
        default=None,
        help='Path to save/load fitted parameters (CSV). If file exists, will load parameters instead of fitting.'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='empirical_fit_output',
        help='Output directory for results'
    )
    
    # Inference parameters
    parser.add_argument(
        '--num_posterior_samples',
        type=int,
        default=1000,
        help='Number of posterior samples to generate per subject'
    )
    
    # Technical settings
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (cpu or cuda)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--summary_stats',
        type=str,
        nargs='+',
        default=ROBERTS_SUMMARY_STAT_KEYS,
        help='List of summary statistics to calculate (default: ROBERTS_SUMMARY_STAT_KEYS)'
    )
    
    return parser.parse_args()


def save_visualizations(all_results: Dict, output_dir: Path, subject_dfs: Dict[int, pd.DataFrame]):
    """Generate and save visualizations for the empirical fitting results."""
    try:
        # Create plots directory
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results).T.reset_index().rename(columns={'index': 'subject'})
        
        # Get parameter names from the first subject's results
        if not results_df.empty and 'posterior_mean' in results_df.columns:
            # Get the first subject's results to determine parameter names
            first_subject = next(iter(all_results.values()))
            param_names = [f'param_{i}' for i in range(len(first_subject['posterior_mean']))]
            
            # Create a new DataFrame with parameter means as columns
            plot_data = []
            for subj_id, results in all_results.items():
                row = {'subject': subj_id}
                for i, param in enumerate(param_names):
                    row[f'param_{i}_mean'] = results['posterior_mean'][i]
                plot_data.append(row)
            
            plot_df = pd.DataFrame(plot_data)
            
            # 1. Parameter Distributions
            plt.figure(figsize=(15, 10))
            num_params = len(param_names)
            n_cols = 3
            n_rows = (num_params + n_cols - 1) // n_cols  # Ceiling division
            
            for i, param in enumerate(param_names):
                plt.subplot(n_rows, n_cols, i+1)
                param_col = f'param_{i}_mean'
                if param_col in plot_df.columns:
                    sns.histplot(plot_df[param_col], kde=True)
                    plt.title(f'Distribution of {param} mean')
                    plt.xlabel(param)
            plt.tight_layout()
            plt.savefig(plots_dir / 'parameter_distributions.png')
            plt.close()
            
            # 2. Pairplot of parameter means
            param_cols = [f'param_{i}_mean' for i in range(num_params)]
            param_cols = [col for col in param_cols if col in plot_df.columns]
            
            if len(param_cols) > 1:
                try:
                    g = sns.pairplot(plot_df[['subject'] + param_cols].dropna())
                    plt.suptitle('Pairplot of Parameter Means', y=1.02)
                    g.savefig(plots_dir / 'parameter_pairplot.png')
                    plt.close()
                except Exception as e:
                    logger.error(f"Error creating pairplot: {e}", exc_info=True)
            
            logger.info(f"Saved visualizations to {plots_dir}")
        else:
            logger.warning("No parameter means found for visualization")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}", exc_info=True)


def save_additional_outputs(inference_results: Dict, output_dir: Path, param_names: List[str]):
    """Save additional outputs like histograms and density plots."""
    try:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Create histograms for each parameter
        samples = inference_results['samples']
        for i, param in enumerate(param_names):
            plt.figure(figsize=(8, 6))
            sns.histplot(samples[:, i], kde=True)
            plt.title(f'Posterior distribution: {param}')
            plt.xlabel(param)
            plt.tight_layout()
            plt.savefig(plots_dir / f'posterior_{param}.png')
            plt.close()
            
    except Exception as e:
        print(f"DEBUG: Error saving additional outputs: {e}")
        logger.error(f"Error saving additional outputs: {e}", exc_info=True)


def main():
    """Main function for running empirical fitting."""
    # Initialize debug information
    debug_info = {
        'start_time': datetime.now(),
        'current_stage': 'initialization',
        'success': False,
        'error': None
    }
    
    def log_debug_info(stage, success=True, error=None):
        """Helper function to log debug information."""
        debug_info['current_stage'] = stage
        debug_info['success'] = success
        if error:
            debug_info['error'] = str(error)
        print(f"\n{'='*40} DEBUG INFO {'='*40}")
        print(f"Stage: {debug_info['current_stage']}")
        print(f"Status: {'SUCCESS' if debug_info['success'] else 'FAILED'}")
        if error:
            print(f"Error: {error}")
        print(f"Elapsed time: {datetime.now() - debug_info['start_time']}")
        print("="*90 + "\n")
    
    try:
        # Initialize logger
        logger = None
        log_file = None
        
        # Stage 1: Parse command line arguments
        try:
            print("\n" + "="*80)
            print("STAGE 1: PARSING COMMAND LINE ARGUMENTS")
            print("="*80 + "\n")
            
            args = parse_args()
            print("DEBUG: Command line arguments parsed successfully")
            print(f"DEBUG: Output directory: {args.output_dir}")
            print(f"DEBUG: NPE checkpoint: {args.npe_checkpoint}")
            print(f"DEBUG: Data file: {args.data_file}")
            
            log_debug_info('parse_arguments', True)
            
        except Exception as e:
            error_msg = f"Failed to parse command line arguments: {str(e)}"
            log_debug_info('parse_arguments', False, error_msg)
            raise
        
        # Stage 2: Set up output directory and logging
        try:
            print("\n" + "="*80)
            print("STAGE 2: SETTING UP OUTPUT AND LOGGING")
            print("="*80 + "\n")
            
            output_dir = Path(args.output_dir)
            print(f"DEBUG: Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = output_dir / 'empirical_fitting.log'
            print(f"DEBUG: Setting up logging to {log_file}")
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            logger = logging.getLogger(__name__)
            
            logger.info("=" * 80)
            logger.info("Starting empirical fitting")
            logger.info(f"Log file: {log_file.absolute()}")
            logger.info(f"Command: {' '.join(sys.argv)}")
            logger.info("-" * 80)
            
            log_debug_info('setup_logging', True)
            
        except Exception as e:
            error_msg = f"Failed to set up output and logging: {str(e)}"
            log_debug_info('setup_logging', False, error_msg)
            raise
        
        # Stage 3: Load NPE model
        try:
            logger.info("=" * 80)
            logger.info(f"{'STAGE 3: LOADING NPE MODEL':^80}")
            logger.info("=" * 80)
            
            # Use npe_file if npe_checkpoint is not provided
            npe_path = args.npe_checkpoint or args.npe_file
            logger.info(f"Loading NPE model from {npe_path}")
            logger.info(f"Using device: {args.device}")
            
            try:
                npe_components = load_npe_posterior(npe_path, device=args.device)
                logger.info("Successfully loaded NPE model")
            except Exception as e:
                error_msg = f"Failed to load NPE model: {str(e)}"
                logger.error(error_msg, exc_info=True)
                logger.info("=" * 80)
                logger.info(f"{'STAGE 3: LOAD_NPE_MODEL':^80}")
                logger.info(f"{'STATUS: FAILED':^80}")
                logger.info(f"{'ERROR: ' + error_msg:^80}")
                logger.info(f"{'Elapsed time: ' + str(datetime.now() - debug_info['start_time']):^80}")
                logger.info("=" * 80)
                return False
            
            if not npe_components:
                raise ValueError("NPE components are None or empty")
                
            param_names = npe_components.get('param_names')
            if not param_names:
                raise ValueError("No parameter names found in NPE components")
            
            print(f"DEBUG: Successfully loaded NPE model with {len(param_names)} parameters")
            print(f"DEBUG: Parameter names: {param_names}")
            logger.info(f"Loaded NPE model with {len(param_names)} parameters")
            
            log_debug_info('load_npe_model', True)
            
        except Exception as e:
            error_msg = f"Failed to load NPE model: {str(e)}"
            logger.error(error_msg, exc_info=True) if logger else None
            log_debug_info('load_npe_model', False, error_msg)
            raise
        
        # Load and preprocess data
        print("\nDEBUG: Loading and preprocessing data...")
        data_file = args.data_file
        print(f"DEBUG: Data file: {data_file}")
        print(f"DEBUG: File exists: {os.path.exists(data_file)}")
        
        df = load_and_preprocess_data(data_file)
        
        if df is None:
            raise ValueError("DataFrame is None after loading")
            
        if 'subject' not in df.columns:
            raise ValueError("DataFrame missing 'subject' column")
            
        num_subjects = len(df['subject'].unique())
        print(f"DEBUG: Successfully loaded {len(df)} rows for {num_subjects} subjects")
        logger.info(f"Loaded data for {num_subjects} subjects")
        
        if num_subjects == 0:
            logger.warning("No subjects found in the data")
            print("DEBUG: WARNING - No subjects found in the data")
        else:
            print(f"DEBUG: Sample subject IDs: {df['subject'].unique()[:5]}")
        
        # Process each subject
        all_results = {}
        subject_list = df['subject'].unique()
        print(f"\nDEBUG: Found {len(subject_list)} unique subjects to process")
        
        for i, subject_id in enumerate(tqdm(subject_list, desc="Processing subjects")):
            print(f"\nDEBUG: Processing subject {i+1}/{len(subject_list)}: {subject_id}")
            logger.info(f"Processing subject {i+1}/{len(subject_list)}: {subject_id}")
            
            try:
                # Filter data for this subject
                subject_df = df[df['subject'] == subject_id].copy()
                
                if subject_df.empty:
                    print(f"DEBUG: WARNING - No data found for subject {subject_id}")
                    logger.warning(f"No data found for subject {subject_id}")
                    continue
                    
                print(f"DEBUG: Found {len(subject_df)} trials for subject {subject_id}")
                
                # Calculate summary statistics
                print("DEBUG: Calculating summary statistics...")
                x_observed = calculate_summary_stats(
                    df=subject_df,
                    summary_stat_keys=args.summary_stats,
                    subject_id=subject_id
                )
                
                # Run inference
                print("DEBUG: Running inference...")
                inference_results = run_inference(
                    npe_components=npe_components,
                    x_observed=x_observed,
                    num_samples=args.num_posterior_samples,
                    device=args.device
                )
                
                # Save results
                print("DEBUG: Saving results...")
                subject_dir = output_dir / f"subject_{subject_id}"
                subject_dir.mkdir(exist_ok=True)
                
                # Save posterior samples
                samples_df = pd.DataFrame(
                    inference_results['samples'],
                    columns=param_names
                )
                samples_file = subject_dir / 'posterior_samples.csv'
                samples_df.to_csv(samples_file, index=False)
                
                # Save summary statistics
                summary_df = pd.DataFrame({
                    'parameter': param_names,
                    'mean': inference_results['mean'],
                    'std': inference_results['std'],
                    'median': inference_results['median']
                })
                summary_file = subject_dir / 'posterior_summary.csv'
                summary_df.to_csv(summary_file, index=False)
                
                # Save additional outputs
                save_additional_outputs(inference_results, subject_dir, param_names)
                
                # Store results for aggregation
                all_results[subject_id] = {
                    'posterior_mean': inference_results['mean'].tolist(),
                    'posterior_std': inference_results['std'].tolist(),
                    'posterior_median': inference_results['median'].tolist(),
                    'num_trials': len(subject_df)
                }
                
                print(f"DEBUG: Successfully processed subject {subject_id}")
                logger.info(f"Successfully processed subject {subject_id}")
                
            except Exception as e:
                error_msg = f"Error processing subject {subject_id}: {str(e)}"
                print(f"\nDEBUG: {error_msg}")
                logger.error(error_msg, exc_info=True)
                print("DEBUG: Continuing with next subject...")
                continue
        
        # Save aggregated results
        if all_results:
            print("\nDEBUG: Saving aggregated results...")
            output_file = output_dir / 'empirical_fitting_results.csv'
            results_list = []
            
            for subj_id, results in all_results.items():
                row = {'subject': subj_id}
                row.update({
                    f"{param}_mean": mean_val for param, mean_val in zip(param_names, results['posterior_mean'])
                })
                row.update({
                    f"{param}_std": std_val for param, std_val in zip(param_names, results['posterior_std'])
                })
                row['num_trials'] = results['num_trials']
                results_list.append(row)
            
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(output_file, index=False)
            print(f"DEBUG: Saved aggregated results to {output_file}")
            logger.info(f"Saved aggregated results for {len(results_df)} subjects to {output_file}")
        
        # Generate visualizations
        print("\nDEBUG: Generating visualizations...")
        save_visualizations(all_results, output_dir, {k: df[df['subject'] == k] for k in all_results.keys()})
        
        print("\n" + "="*80)
        print("DEBUG: Empirical fitting completed successfully!")
        print("="*80 + "\n")
        logger.info("=" * 80)
        logger.info("Empirical fitting completed successfully!")
        logger.info(f"Results saved to: {output_dir.absolute()}")
        
        return True
        
    except Exception as e:
        error_msg = f"FATAL ERROR: {str(e)}"
        print("\n" + "!"*80)
        print(f"DEBUG: {error_msg}")
        print("!"*80 + "\n")
        if logger:
            logger.error(error_msg, exc_info=True)
        return False
    
    # Determine which data file to use
    data_file = args.roberts_data_file or args.data_file
    
    # Load fitted parameters if provided
    if args.fitted_params_file:
        try:
            with open(args.fitted_params_file, 'r') as f:
                fitted_params = json.load(f)
            logger.info(f"Loaded fitted parameters from {args.fitted_params_file}")
        except Exception as e:
            logger.error(f"Error loading fitted parameters: {e}", exc_info=True)
            return False
    
    all_results = {}
    
    # Convert to list if needed and show progress
    subject_list = list(all_subjects) if not isinstance(all_subjects, list) else all_subjects
    
    print(f"\nDEBUG: Preparing to process {len(subject_list)} subjects")
    logger.info(f"Processing {len(subject_list)} subjects")
    
    if not subject_list:
        print("DEBUG: WARNING - No subjects to process")
        logger.warning("No subjects to process")
        return False
        
    for i, subject_id in enumerate(tqdm(subject_list, desc="Processing subjects")):
        print(f"\nDEBUG: Processing subject {i+1}/{len(subject_list)}: {subject_id}")
        logger.info(f"Processing subject {i+1}/{len(subject_list)}: {subject_id}")
        
        try:
            # Filter data for this subject
            subject_df = df[df['subject'] == subject_id].copy()
            
            if subject_df.empty:
                print(f"DEBUG: WARNING - No data found for subject {subject_id}")
                logger.warning(f"No data found for subject {subject_id}")
                continue
                
            print(f"DEBUG: Found {len(subject_df)} trials for subject {subject_id}")
            # Calculate summary statistics for this subject
            x_observed = calculate_summary_stats(
                df=subject_df,
                summary_stat_keys=npe_components['summary_stat_keys'] or list(ROBERTS_SUMMARY_STAT_KEYS),
                subject_id=subject_id
            )
            
            # Run inference for this subject
            inference_results = run_inference(
                npe_components=npe_components,
                x_observed=x_observed,
                num_samples=args.num_samples,
                device=args.device
            )
            
            # Create subject-specific output directory
            subject_dir = output_dir / f'subject_{subject_id}'
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            # Save inference results
            save_additional_outputs(
                inference_results,
                subject_dir,
                param_names
            )
            
            # Save results for aggregation
            all_results[subject_id] = {
                'posterior_mean': inference_results.get('mean', []).tolist(),
                'posterior_std': inference_results.get('std', []).tolist(),
                'posterior_median': inference_results.get('median', []).tolist(),
                'num_trials': len(subject_df)
            }
            
            logger.info(f"Successfully processed subject {subject_id}")
            print(f"DEBUG: Successfully processed subject {subject_id}")
                
        except Exception as e:
            error_msg = f"ERROR processing subject {subject_id}: {str(e)}"
            print(f"\nDEBUG: {error_msg}")
            logger.error(error_msg, exc_info=True)
            print(f"DEBUG: Continuing with next subject...")
    
    # Save aggregated results
    if all_results:
        # Save combined results
        output_file = output_dir / 'empirical_fitting_results_6param.csv'
        try:
            results_df = pd.DataFrame.from_dict(all_results, orient='index')
            results_df.index.name = 'subject_id'
            results_df.to_csv(output_file)
            logger.info(f"Saved aggregated results to {output_file}")
            print(f"DEBUG: Saved aggregated results to {output_file}")
            
            # Save visualizations
            save_visualizations(all_results, output_dir, subject_dfs={
                int(k.split('_')[-1]): df[df['subject'] == int(k.split('_')[-1])] 
                for k in all_results.keys()
            })
            
            # Save fitted parameters if requested
            if args.fitted_params_file:
                try:
                    with open(args.fitted_params_file, 'w') as f:
                        json.dump(convert_numpy_types(all_results), f, indent=2)
                    logger.info(f"Saved fitted parameters to {args.fitted_params_file}")
                except Exception as e:
                    print(f"DEBUG: Error saving fitted parameters: {e}")
                    logger.error(f"Error saving fitted parameters: {e}", exc_info=True)
            
            logger.info("=" * 80)
            success_msg = f"Empirical fitting completed successfully for {len(all_results)} subjects"
            logger.info(success_msg)
            print(f"\nDEBUG: {success_msg}")
            return True
            
        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            print(f"\nDEBUG: {error_msg}")
            logger.error(error_msg, exc_info=True)
            return False
    else:
        success_msg = "Empirical fitting completed but no results were generated"
        logger.warning(success_msg)
        print(f"\nDEBUG: {success_msg}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
