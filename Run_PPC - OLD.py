#!/usr/bin/env python3

"""Run Posterior Predictive Checks (PPCs) for the 5‑parameter NES model.

This script follows the specification laid out in *PPC_Script_Request.md* and
implements best‑practice structure for computational‑cognitive‑science PPC code.
It assumes the following project layout (relative or absolute paths may be
supplied through CLI arguments):

project_root/
 ├─ src/
 │   ├─ agent_mvnes.py            # MVNESAgent implementation
 │   └─ stats_schema.py           # ROBERTS_SUMMARY_STAT_KEYS + helpers
 └─ data/
     ├─ npe_model.pt              # trained 5‑parameter NPE checkpoint
     ├─ fitted_params.csv         # empirical posterior summaries per subject
     └─ roberts_data.csv          # raw Roberts et al. behavioural data

The script *does not* retrain NPEs or perform SBC.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from datetime import datetime
import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sbi.utils import get_log_root
# In sbi v0.22.0, use DirectPosterior from sbi.inference.posteriors
from sbi.inference.posteriors import DirectPosterior as Posterior
from tqdm import tqdm

# Define the 6 parameters used in the NES model
PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain', 'beta_val']

from nes_copilot.roberts_stats import calculate_summary_stats_roberts as calculate_summary_stats
from nes_copilot.config_manager_fixed import FixedConfigManager as ConfigManager
from nes_copilot.agent_mvnes import MVNESAgent


def load_roberts_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the Roberts et al. framing task data.
    
    Args:
        file_path: Path to the CSV file containing the raw data.
        
    Returns:
        Preprocessed DataFrame with standardized column names.
    """
    # Load the raw data
    df = pd.read_csv(file_path)
    
    # Standardize column names (lowercase with underscores)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Map column names to expected names if needed
    column_mapping = {
        'gaincolor': 'frame',  # 'gain' or 'loss' frame
        'trialtype': 'trial_type',  # 'TC' or 'NTC'
        'rt': 'rt',  # reaction time in seconds
        'choice': 'choice',  # 1 for gamble, 0 for sure
        'subject': 'subject',  # subject ID
        'trial': 'trial_number',  # trial number
        'gamble': 'gamble_amount',  # amount of the gamble
        'sure': 'sure_amount',  # amount of the sure option
        'chosegamble': 'chose_gamble',  # 1 if chose gamble, 0 if chose sure
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_columns = ['frame', 'trial_type', 'rt', 'choice', 'subject']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the input data")
    
    # Convert frame to lowercase
    if 'frame' in df.columns:
        df['frame'] = df['frame'].str.lower()
    
    # Convert trial_type to uppercase
    if 'trial_type' in df.columns:
        df['trial_type'] = df['trial_type'].str.upper()
    
    # Ensure choice is binary (0 or 1)
    if 'choice' in df.columns:
        df['choice'] = df['choice'].astype(int)
        if not df['choice'].isin([0, 1]).all():
            raise ValueError("Choice column must contain only 0s and 1s")
    
    # Ensure RT is positive
    if 'rt' in df.columns:
        if (df['rt'] <= 0).any():
            raise ValueError("All reaction times must be positive")
    
    return df
import matplotlib.gridspec as gridspec

# Set up plotting style
sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
import pandas as pd
import torch
from tqdm.auto import tqdm

# Add 'src/' directory to PYTHONPATH at runtime
sys.path.append(str(Path(__file__).parent / "src"))

try:
    # For sbi 0.22.0
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform           # type: ignore
except ImportError as err:  # pragma: no cover
    try:
        # For sbi 0.23.0 and later
        from sbi.inference.trainers.npe import SNPE
        from sbi.utils import BoxUniform
    except ImportError:
        raise ImportError("sbi is required for this script. Install via `pip install sbi`.\n" + str(err)) from err

# --------------------------------------------------------------------------------------
# Local project imports (must be import‑able via PYTHONPATH or installed as a package)
# --------------------------------------------------------------------------------------
try:
    # First try importing from nes_copilot module
    from nes_copilot.agent_mvnes import MVNESAgent  # type: ignore
    from nes_copilot.roberts_stats import (
        ROBERTS_SUMMARY_STAT_KEYS,
        validate_summary_stats,
        calculate_summary_stats_roberts as calculate_summary_stats
    )
    
except ImportError as import_err:  # pragma: no cover
    try:
        # Fallback to direct imports if nes_copilot module is not available
        from agent_mvnes import MVNESAgent  # type: ignore
        from roberts_stats import (
            ROBERTS_SUMMARY_STAT_KEYS,
            validate_summary_stats,
            calculate_summary_stats_roberts as calculate_summary_stats
        )
    except ImportError as fallback_err:
        raise ImportError(
            "Could not import local project modules. Ensure `src/` is on PYTHONPATH.\n"
            f"Primary error: {import_err}\n"
            f"Fallback error: {fallback_err}"
        ) from fallback_err

# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_ppc",
        description="Posterior Predictive Checks for the 5‑parameter NES model",
    )
    # Make npe_file optional since we're using empirical fits
    parser.add_argument("--npe_file", type=Path, required=False, help="[Optional] Path to trained NPE .pt file")
    parser.add_argument("--fitted_params_file", type=Path, required=True, help="CSV with empirical posterior summaries")
    parser.add_argument("--roberts_data_file", type=Path, required=False, default=None, help="Raw Roberts data CSV (trial‑level)")
    parser.add_argument("--output_dir", type=Path, default=Path("ppc_outputs"),
                        help="Output directory for PPC results")
    parser.add_argument("--subject_ids", type=str, default=None, help="Comma‑separated subject IDs to analyse (default: all)")
    parser.add_argument("--num_posterior_draws", type=int, default=500, help="# parameter draws per subject posterior")
    parser.add_argument("--num_simulations_per_draw", type=int, default=1, help="# simulated datasets per draw")
    parser.add_argument("--n_sim_reps", type=int, default=100,
                        help="Number of simulation replicates per subject for PPC")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (torch & numpy)")
    parser.add_argument("--threshold_scale", type=float, default=1.0,
                        help="Scaling factor for the threshold parameter (a_0)")
    parser.add_argument("--empirical_fits_only", action="store_true",
                        help="Use empirical median fits instead of posterior sampling")
    parser.add_argument("--ppc_version", type=str, default="v9_6param_empirical_beta",
                        help="Version identifier for output directory")
    parser.add_argument("--force_cpu", action="store_true", help="Run NPE sampling on CPU even if CUDA available")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def setup_logging(debug: bool = True) -> None:
    """Configure root logger for console output.
    
    Args:
        debug: If True, set log level to DEBUG. Otherwise, use INFO.
    """
    # Always set root logger to DEBUG to capture all messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(asctime)s %(name)s ‑ %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr
    )
    
    # Set the level for the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Also set the level for the console handler if it exists
    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)
    
    # Enable debug logging for our specific modules
    for module in ['__main__', 'stats_schema', 'posterior_predictive_check']:
        logger = logging.getLogger(module)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
    logging.debug("Debug logging enabled")


def load_npe_posterior(checkpoint_path: Path, device: torch.device) -> Posterior:
    """Reconstruct an sbi Posterior object from saved checkpoint."""
    logging.info("Loading NPE checkpoint from %s", checkpoint_path)
    chk = torch.load(checkpoint_path, map_location=device)

    # Validate mandatory keys ----------------------------------------------------------
    required_keys = {
        "density_estimator_state_dict",
        "num_summary_stats",
        "summary_stat_keys",
        "param_names",
        "prior_params",
    }
    missing = required_keys.difference(chk.keys())
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")

    # Ensure summary statistics schema matches runtime ---------------------------------
    validate_summary_stats()
    runtime_keys = list(ROBERTS_SUMMARY_STAT_KEYS)
    if list(chk["summary_stat_keys"]) != runtime_keys:
        raise ValueError("Summary‑stat keys in checkpoint do not match stats_schema.py definition.")

    # Rebuild prior and dummy SNPE -----------------------------------------------------
    low = torch.as_tensor(chk["prior_params"]["low"], dtype=torch.float32)
    high = torch.as_tensor(chk["prior_params"]["high"], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high, device=device)

    dummy_sim_x = torch.zeros((2, chk["num_summary_stats"]), dtype=torch.float32, device=device)
    dummy_sim_theta = torch.zeros((2, len(chk["param_names"])), dtype=torch.float32, device=device)

    inference = SNPE(prior=prior, device=device)
    density_estimator = inference.append_simulations(theta=dummy_sim_theta, x=dummy_sim_x).train(max_num_epochs=0)
    density_estimator.load_state_dict(chk["density_estimator_state_dict"], strict=True)
    posterior = inference.build_posterior(density_estimator)
    logging.info("Posterior reconstructed successfully (sbi version %s)", chk.get("sbi_version", "unknown"))
    return posterior


def safe_float(x):
    """Safely convert value to float, handling None and NaN."""
    if pd.isna(x):
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

def safe_bool(x):
    """Safely convert value to boolean."""
    if pd.isna(x):
        return False
    try:
        return bool(x)
    except (TypeError, ValueError):
        return False

def safe_str(x, max_len=20):
    """Safely convert value to string with max length."""
    if pd.isna(x):
        return ''
    try:
        return str(x)[:max_len]
    except (TypeError, ValueError):
        return ''

def get_trial_structure(df: pd.DataFrame) -> List[Dict]:
    """Converts a preprocessed subject DataFrame to a list of trial dictionaries.
    
    Args:
        df: Preprocessed DataFrame containing trial data for one subject
            
    Returns:
        List of dictionaries, where each dictionary represents a trial
        
    Raises:
        ValueError: If required columns are missing from the input DataFrame
    """
    required_cols = [
        'prob', 
        'is_gain_frame', 
        'frame', 
        'cond', 
        'time_constrained', 
        'valence_score', 
        'norm_category_for_trial'
    ]
    
    # Check for missing columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame passed to get_trial_structure is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Create a working copy with only the required columns
    df2 = df[required_cols].copy()
    
    # Convert columns to appropriate types
    numeric_cols = ['prob', 'valence_score']
    bool_cols = ['is_gain_frame', 'time_constrained']
    str_cols = ['frame', 'cond', 'norm_category_for_trial']
    
    # Convert numeric columns to float first
    for col in numeric_cols:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
    
    # Fill numeric columns with their means in a single operation
    numeric_cols_exist = [col for col in numeric_cols if col in df2.columns]
    if numeric_cols_exist:
        means = df2[numeric_cols_exist].mean()
        df2[numeric_cols_exist] = df2[numeric_cols_exist].fillna(means)
        logging.warning(f"Filled missing values in numeric columns with their means: {means.to_dict()}")
    
    # Handle boolean columns
    for col in bool_cols:
        if col in df2.columns:
            df2[col] = df2[col].astype(bool)
    
    # Handle string columns - forward fill then fill remaining with empty string
    for col in str_cols:
        if col in df2.columns:
            df2[col] = df2[col].fillna(method='ffill').fillna('')
    
    # Ensure all required columns are present and have the correct type
    for col in required_cols:
        if col not in df2.columns:
            if col in numeric_cols:
                df2[col] = 0.0
            elif col in bool_cols:
                df2[col] = False
            else:
                df2[col] = ''
    
    # Log first few rows for debugging
    logging.debug("First 3 rows of trial structure:")
    for i in range(min(3, len(df2))):
        logging.debug("  Row %d: %s", i, df2.iloc[i].to_dict())
    
    # Return as list of dicts
    return df2[required_cols].to_dict('records')


def simulate_one_full_dataset(
    params: Dict[str, float],
    trial_struct: Sequence[Dict],  # List of trial dictionaries
    agent: MVNESAgent,
) -> Dict[str, float]:
    """Run MVNESAgent across a subject's trial structure and compute summary stats.
    
    Args:
        params: Dictionary of model parameters including:
               - v_norm: float
               - a_0: float
               - w_s_eff: float
               - t_0: float
               - alpha_gain: float
               - beta_val: float
               - logit_z0: float (optional, defaults to 0.0)
               - log_tau_norm: float (optional, defaults to -0.693147)
               - meta_cognitive_on: bool (optional, defaults to False)
        trial_struct: List of dictionaries containing trial information
        agent: Instance of MVNESAgent
        
    Returns:
        Dictionary of summary statistics for the simulated data
    """
    # Log parameter values for debugging
    logging.debug("Simulate_one_full_dataset received parameters:")
    for param_name_key, value in params.items():
        logging.debug(f"  {param_name_key}: {value}")
    
    # Initialize list to store simulated trial data
    simulated_rows = []
    
    # Validate trial structure
    if len(trial_struct) == 0:
        logging.warning("Empty trial structure provided")
        return {key: np.nan for key in ROBERTS_SUMMARY_STAT_KEYS}

    # These are the parameters that will be passed to agent.run_mvnes_trial's `params` dictionary
    # We include all DDM parameters that the agent needs for simulation
    base_params_for_trial_method = {
        # Core DDM parameters, taken from the 'params' dict passed to simulate_one_full_dataset
        'w_s': params['w_s'],
        'w_n': params['w_n'],
        'threshold_a': params['threshold_a'],
        't': params['t'],
        'alpha_gain': params['alpha_gain'],
        'beta_val': params['beta_val'],
        
        # Optional parameters with defaults
        'logit_z0': params.get('logit_z0', 0.0),
        'log_tau_norm': params.get('log_tau_norm', -0.693147),
        'meta_cognitive_on': params.get('meta_cognitive_on', False),
        
        # Simulation control parameters
        'noise_std_dev': params.get('noise_std_dev', 1.0),
        'dt': params.get('dt', 0.01)
        # 'max_time' will be added per trial below
    }

    # Log trial structure information
    logging.debug(f"Processing {len(trial_struct)} trials")
    
    # Iterate over the list of trial dictionaries
    for i, trial_info in enumerate(trial_struct):
        try:
            # Extract trial data from dictionary
            prob = float(trial_info.get('prob', 0.5))
            is_gain_frame = bool(trial_info.get('is_gain_frame', False))
            frame_str_val = 'gain' if is_gain_frame else 'loss'  # Renamed to avoid conflict with 'frame' column in DataFrame
            cond = str(trial_info.get('cond', ''))
            time_constrained = bool(trial_info.get('time_constrained', True))
            valence_score = float(trial_info.get('valence_score', 0.0))
            
            # Set up trial parameters
            salience_input = float(prob)
            norm_input = 1.0 if is_gain_frame else -1.0
            
            # Prepare the params for this specific trial run
            current_trial_run_params = base_params_for_trial_method.copy()
            current_trial_run_params['max_time'] = 1.0 if time_constrained else 3.0
            
            # Get norm category information
            norm_category = str(trial_info.get('norm_category_for_trial', 'default'))
            category_mapping = {
                'default': 0, 'fairness': 1, 'altruism': 2, 
                'reciprocity': 3, 'trust': 4, 'cooperation': 5
            }
            norm_category_code = category_mapping.get(norm_category.lower(), 0)
            
            # Log first few trials for debugging
            if i < 3:
                logging.debug(f"Trial {i} - prob: {prob:.2f}, frame: {frame_str_val}, cond: {cond}")
                logging.debug(f"  salience_input: {salience_input:.2f}, norm_input: {norm_input:.2f}")
                logging.debug(f"  valence_score: {valence_score:.2f}, norm_category: {norm_category} (code: {norm_category_code})")
                logging.debug(f"  trial_run_params: {', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in current_trial_run_params.items())}")
            
            # Run the trial with the agent - pass parameters according to agent's signature
            trial_output = agent.run_mvnes_trial(
                salience_input,          # Positional: Strength of stimulus push
                norm_input,              # Positional: Norm input (+1 for gain, -1 for loss)
                params=current_trial_run_params,  # All parameters passed as a single dictionary
                valence_score_trial=valence_score,
                norm_category_for_trial=norm_category_code
            )
            
            # Process trial output and store results
            if trial_output is not None:
                # Extract relevant information from trial output
                rt = trial_output.get('rt', np.nan)
                choice = trial_output.get('choice', 0)
                
                # Store trial results
                simulated_rows.append({
                    'trial': i + 1,
                    'prob': prob,
                    'frame': frame_str_val,
                    'cond': cond,
                    'time_constrained': time_constrained,
                    'valence_score': valence_score,
                    'norm_category': norm_category,
                    'rt': rt,
                    'choice': choice,
                    'correct': int(choice == (1 if is_gain_frame else 0))
                })
            
        except TypeError as e:
            logging.error(f"TypeError in trial {i}: {e}")
            logging.error(f"Trial info: {trial_info}")
            # Log each field in the trial info dictionary
            for field_key, field_value in trial_info.items():
                logging.error(f"  {field_key}: {field_value} (type: {type(field_value)})")
            raise
        except Exception as e:
            logging.error(f"Error processing trial {i}: {str(e)}")
            continue

    # Convert to DataFrame for summary statistics
    if not simulated_rows:
        logging.warning("No valid trials simulated")
        return {key: np.nan for key in ROBERTS_SUMMARY_STAT_KEYS}
        
    df_sim = pd.DataFrame(simulated_rows)
    
    # Log frame and condition values for debugging
    logging.debug(f"Frame values in simulation: {df_sim['frame'].unique().tolist()}")
    logging.debug(f"Condition values in simulation: {df_sim['cond'].unique().tolist()}")
    
    # Log gamble rates per condition
    if not df_sim.empty:
        for frame in df_sim['frame'].unique():
            for cond in df_sim['cond'].unique():
                mask = (df_sim['frame'] == frame) & (df_sim['cond'] == cond)
                if mask.any():
                    gamble_rate = df_sim.loc[mask, 'choice'].mean()
                    logging.debug(f"Gamble rate for frame={frame}, cond={cond}: {gamble_rate:.2f}")
    
    try:
        # Calculate summary statistics
        stats_dict = calculate_summary_stats(df_sim, ROBERTS_SUMMARY_STAT_KEYS)
        
        # Log calculated statistics
        logging.debug("Calculated statistics:")
        for key, value in stats_dict.items():
            logging.debug(f"  {key}: {value}")
            
    except Exception as e:
        logging.error(f"Error calculating summary statistics: {e}", exc_info=True)
        # Return NaN for all stats if calculation fails
        stats_dict = {key: np.nan for key in ROBERTS_SUMMARY_STAT_KEYS}
    
    # Add frame and condition information to stats
    stats_dict['frame'] = frame if 'frame' in locals() else 'unknown'
    stats_dict['condition'] = cond if 'cond' in locals() else 'unknown'
    
    # Ensure all required stats are in the output
    for key in ROBERTS_SUMMARY_STAT_KEYS:
        if key not in stats_dict:
            stats_dict[key] = np.nan
    
    return stats_dict


def simulate_n_replicates(
    params: Dict[str, float],
    trial_struct: Sequence[Dict],
    agent: MVNESAgent,
    n_reps: int = 100
) -> List[Dict[str, float]]:
    """Run multiple simulations with the same parameters to build a distribution of summary stats.
    
    Args:
        params: Dictionary of model parameters
        trial_struct: List of dictionaries containing trial information
        agent: Instance of MVNESAgent
        n_reps: Number of replications to run
        
    Returns:
        List of summary statistics dictionaries, one per replication
    """
    all_stats = []
    
    for i in tqdm(range(n_reps), desc=f"Simulations for subject", leave=False):
        # Run one full simulation
        stats = simulate_one_full_dataset(params, trial_struct, agent)
        all_stats.append(stats)
    
    return all_stats


def collect_ppc_for_subject(
    subj_id: str,
    df_empirical: pd.DataFrame,
    posterior: Posterior,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict]:
    """Run PPC simulations for a single subject and return observed vs simulated stats.
    
    Args:
        subj_id: Subject ID to process
        df_empirical: DataFrame containing empirical data for all subjects
        posterior: sbi Posterior object for parameter sampling
        args: Command line arguments
        device: Device to use for computations (e.g., 'cuda' or 'cpu')
        
    Returns:
        Dictionary containing observed and simulated summary statistics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing subject {subj_id}")
    
    # Get subject-specific data
    df_subj = df_empirical[df_empirical['subject'] == subj_id].copy()
    if len(df_subj) == 0:
        logger.warning(f"No data found for subject {subj_id}")
        return {}
    
    # Ensure required columns are present
    required_cols = ['prob', 'is_gain_frame', 'frame', 'cond', 'time_constrained', 
                    'valence_score', 'norm_category_for_trial']
    
    # Add missing columns with default values if necessary
    for col in required_cols:
        if col not in df_subj.columns:
            if col == 'is_gain_frame':
                df_subj['is_gain_frame'] = df_subj['frame'].str.lower() == 'gain'
            elif col == 'time_constrained':
                df_subj['time_constrained'] = df_subj['cond'].str.lower() == 'tc'
            elif col == 'valence_score':
                df_subj['valence_score'] = 0.0  # Should be replaced with actual valence processing
            elif col == 'norm_category_for_trial':
                df_subj['norm_category_for_trial'] = 'default'
    
    # Convert trial structure to NumPy record array
    try:
        trial_struct = get_trial_structure(df_subj)
        logger.debug(f"Created trial structure with {len(trial_struct)} trials")
        
        # Log first few trials for debugging
        for i in range(min(3, len(trial_struct))):
            logger.debug(f"Trial {i} structure: {dict(trial_struct[i])}")
            
    except Exception as e:
        logger.error(f"Error creating trial structure for subject {subj_id}: {str(e)}", exc_info=True)
        return {}
    
    # Sample parameters from the posterior
    try:
        # Sample parameters for this subject
        theta_samples = posterior.sample((args.num_posterior_samples,), 
                                      show_progress_bars=True).cpu().numpy()
        logger.info(f"Sampled {len(theta_samples)} parameter vectors from posterior")
    except Exception as e:
        logger.error(f"Error sampling from posterior for subject {subj_id}: {str(e)}", exc_info=True)
        return {}
    
    # Initialize agent (single instance for all simulations)
    agent = MVNESAgent()
    
    # Get parameter names from the posterior
    param_names = getattr(posterior, 'parameter_names', None)
    if param_names is None:
        # Fallback if parameter_names is not available
        param_names = [f'param_{i}' for i in range(theta_samples.shape[1])]
        logger.warning(f"No parameter names in posterior, using default: {param_names}")
    
    # Log parameter mapping
    logger.debug(f"Parameter names from posterior: {param_names}")
    logger.debug(f"Expected parameter order: {PARAM_NAMES}")
    
    # Prepare to collect results
    all_sim_stats = []
    
    # Process each posterior sample
    for sample_idx, theta_single_sample_np in enumerate(tqdm(
        theta_samples, 
        desc=f"Simulating subj {subj_id}", 
        leave=False
    )):
        try:
            # Create parameter dictionary directly from the posterior sample
            params_dict = {name: float(val) for name, val in zip(param_names, theta_single_sample_np)}
            
            # Add fixed parameters that MVNESAgent expects
            params_dict.update({
                'logit_z0': 0.0,  # Default starting point
                'log_tau_norm': -0.693147,  # Log of norm decay time constant (tau ≈ 0.5s)
                'meta_cognitive_on': False  # Disable meta-cognitive system by default
            })
            
            # Log the parameters for debugging (first and last few samples)
            if sample_idx < 2 or sample_idx >= len(theta_samples) - 2:
                logger.debug(f"Sample {sample_idx} parameters: {params_dict}")
            
            # Run simulations with these parameters
            sim_reps = simulate_n_replicates(
                params=params_dict,
                trial_struct=trial_struct,
                agent=agent,
                n_reps=args.num_simulations_per_draw
            )
            
            # Add to our collection of results
            all_sim_stats.extend(sim_reps)
            
        except Exception as e:
            logger.error(f"Error in simulation {sample_idx} for subject {subj_id}: {str(e)}", exc_info=True)
            continue
    
    # Calculate observed statistics
    try:
        observed_stats = calculate_summary_stats(df_subj)
        logger.debug(f"Observed stats: {observed_stats}")
    except Exception as e:
        logger.error(f"Error calculating observed stats for subject {subj_id}: {str(e)}", exc_info=True)
        return {}
    
    # Return results
    return {
        'observed': observed_stats,
        'simulated': all_sim_stats,
        'n_simulations': len(all_sim_stats),
        'n_posterior_samples': len(theta_samples)
    }


def is_close(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Helper function to compare floating point numbers with tolerance."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= (atol + rtol * abs(b))

def plot_ppc_distributions(
    ppc_results: Dict[str, Dict[str, Dict]], 
    output_dir: Path,
    subject_ids: Optional[List[str]] = None,
    stats_to_plot: Optional[List[str]] = None
) -> None:
    """Generate PPC plots for each subject and summary statistic.
    
    Args:
        ppc_results: Dictionary containing PPC results
        output_dir: Base directory to save plots
        subject_ids: List of subject IDs to plot (if None, plot all)
        stats_to_plot: List of stats to plot (if None, plot all)
    """
    if subject_ids is None:
        subject_ids = list(ppc_results.keys())
    
    # Default stats to plot if none specified
    if stats_to_plot is None:
        # Use a representative sample of stats if none specified
        stats_to_plot = [
            'prop_gamble_overall',
            'mean_rt_overall',
            'framing_effect_ntc',
            'framing_effect_tc',
            'rt_std_overall',
            'prop_gamble_Gain_NTC',
            'prop_gamble_Loss_TC',
            'mean_rt_Gain_TC',
            'mean_rt_Loss_NTC'
        ]
    
    # Create output directories
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for subj in tqdm(subject_ids, desc="Generating PPC plots"):
        if subj not in ppc_results:
            continue
            
        obs_stats_dict = ppc_results[subj].get('observed_summary_stats', {})
        sim_stats_list_of_dicts = ppc_results[subj].get('simulated_summary_stats_list', [])
        
        for stat in stats_to_plot:
            if stat not in obs_stats_dict or not sim_stats_list_of_dicts:
                continue
                
            try:
                obs_val = obs_stats_dict[stat]
                sim_vals_for_stat = [s_dict.get(stat, np.nan) for s_dict in sim_stats_list_of_dicts 
                                  if isinstance(s_dict, dict) and stat in s_dict]
                sim_vals_for_stat = np.array([s for s in sim_vals_for_stat if not pd.isna(s)])
                
                if len(sim_vals_for_stat) == 0 or np.all(np.isnan(sim_vals_for_stat)):
                    continue
                
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # Plot histogram of simulated values
                sns.histplot(sim_vals_for_stat, kde=True, stat='density', 
                            color='skyblue', edgecolor='white',
                            bins=min(30, max(5, len(sim_vals_for_stat)//10)))
                
                # Add observed value line
                ymin, ymax = plt.ylim()
                plt.axvline(obs_val, color='red', linestyle='--', linewidth=2,
                           label=f'Observed: {obs_val:.3f}')
                
                # Add percentiles
                lower = np.percentile(sim_vals_for_stat, 5)
                upper = np.percentile(sim_vals_for_stat, 95)
                plt.axvspan(lower, upper, alpha=0.2, color='gray',
                           label='90% CI')
                
                # Formatting
                plt.title(f'Subject {subj} - {stat}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                
                # Save figure
                plt.tight_layout()
                plt.savefig(plot_dir / f'{stat}.png')
                plt.close()
                
            except Exception as e:
                logging.warning(f"Error plotting {stat} for subject {subj}: {str(e)}")
                plt.close('all')

def calculate_detailed_coverage(ppc_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """Calculate detailed coverage statistics for each subject and statistic.
    
    Args:
        ppc_results: Dictionary with subject IDs as keys and nested dictionaries
                   containing 'observed' and 'simulated' summary statistics
                   
    Returns:
        DataFrame with detailed coverage statistics
    """
    rows = []
    
    for subj, data in ppc_results.items():
        obs_stats_dict = data.get('observed_summary_stats', {})
        sim_stats_list_of_dicts = data.get('simulated_summary_stats_list', [])
        
        # Get all stats from observed data
        stats_to_process = list(obs_stats_dict.keys())
        
        for stat in stats_to_process:
            # Extract values for this stat from all simulations
            sim_vals_for_stat = [s_dict.get(stat, np.nan) for s_dict in sim_stats_list_of_dicts 
                              if isinstance(s_dict, dict) and stat in s_dict]
            sim_vals_for_stat = np.array([s for s in sim_vals_for_stat if not pd.isna(s)])
            obs_val = obs_stats_dict.get(stat, np.nan)  # Use .get() for safety
            
            # Skip if no valid simulations or missing observed value
            if len(sim_vals_for_stat) == 0 or np.all(np.isnan(sim_vals_for_stat)) or np.isnan(obs_val):
                continue
            
            # Calculate statistics
            sim_mean = np.nanmean(sim_vals_for_stat)
            sim_std = np.nanstd(sim_vals_for_stat, ddof=1)
            
            # Calculate coverage
            lower_90 = np.percentile(sim_vals_for_stat, 5)
            upper_90 = np.percentile(sim_vals, 95)
            covered_90 = int(lower_90 <= obs_val <= upper_90)
            
            lower_95 = np.percentile(sim_vals, 2.5)
            upper_95 = np.percentile(sim_vals, 97.5)
            covered_95 = int(lower_95 <= obs_val <= upper_95)
            
            rows.append({
                'subject_id': subj,
                'stat_key': stat,
                'observed': obs_val,
                'simulated_mean': sim_mean,
                'simulated_std': sim_std,
                'covered_90': covered_90,
                'covered_95': covered_95,
                'ci_90_lower': lower_90,
                'ci_90_upper': upper_90,
                'n_simulations': len(sim_vals)
            })
    
    if not rows:
        logging.warning("No valid data for coverage calculation")
        return pd.DataFrame()
    
    return pd.DataFrame(rows)

def aggregate_coverage(ppc_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """Compute coverage statistics across subjects and summary stat keys.
    
    This function processes the results of multiple simulations per subject
    to calculate coverage statistics for each summary statistic.
    This is a legacy function that wraps calculate_detailed_coverage for backward compatibility.
    Prefer using calculate_detailed_coverage for new code.
    
    Args:
        ppc_results: Dictionary with subject IDs as keys and nested dictionaries
                   containing 'observed' and 'simulated' summary statistics
                   
    Returns:
        DataFrame with coverage statistics for each summary statistic
    """
    logging.info(f"Aggregating coverage for {len(ppc_results)} subjects")
    rows = []
    
    for subj, data in ppc_results.items():
        obs = data.get("observed", {})
        sims = data.get("simulated", {})
        
        if not obs or not sims:
            logging.warning(f"Subject {subj} is missing observed or simulated data")
            continue
            
        for key in set(obs.keys()) | set(sims.keys()):
            if key not in obs or key not in sims or not sims[key]:
                continue
                
            obs_val = obs[key]
            sim_vals = sims[key] if isinstance(sims[key], list) else [sims[key]]
            
            # For each simulation, check if observed value is within the simulated distribution
            for sim_val in sim_vals:
                # Use tolerance-based comparison for floating point numbers
                is_covered = is_close(obs_val, sim_val)
                rows.append({
                    "subject": subj,
                    "stat_key": key,
                    "covered_90": float(is_covered),
                    "covered_95": float(is_covered),
                    "observed": obs_val,
                    "simulated": sim_val
                })
    
    if not rows:
        logging.warning("No valid data for coverage calculation")
        return pd.DataFrame()
        
    logging.info(f"Processed {len(rows)} data points for coverage calculation")
    df = pd.DataFrame(rows)
    
    # Calculate coverage statistics
    if not df.empty:
        # Group by stat_key and calculate coverage statistics
        summary = df.groupby("stat_key").agg({
            "covered_90": ["sum", "mean", "count"],
            "covered_95": ["sum", "mean"],
            "observed": "first",
            "simulated": "first"
        })
        
        # Flatten column names
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        
        # Rename columns for clarity
        summary = summary.rename(columns={
            "covered_90_sum": "n_covered_90",
            "covered_90_mean": "pct_covered_90",
            "covered_90_count": "n_subjects",
            "covered_95_sum": "n_covered_95",
            "covered_95_mean": "pct_covered_95",
            "observed_first": "observed_value",
            "simulated_first": "simulated_value"
        })
        
        # Add number of subjects with data for each stat
        summary = summary.reset_index()
        logging.info(f"Coverage calculated for {len(summary)} unique statistics")
        return summary
    
    return pd.DataFrame()


# Main entry point
# --------------------------------------------------------------------------------------

def test_unconditioned_sampling(posterior):
    """Test sampling from the posterior without conditioning on data."""
    logging.info("Testing unconditioned sampling...")
    try:
        t0 = time.time()
        with torch.no_grad():
            test_sample = posterior.sample((1,))
        elapsed = time.time() - t0
        logging.info(f"Unconditioned sample took {elapsed:.2f} seconds")
        return True
    except Exception as e:
        logging.error(f"Unconditioned sampling failed: {str(e)}")
        return False


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for running PPCs."""
    args = parse_args(argv)
    setup_logging(debug=args.debug)
    
    # Device selection ----------------------------------------------------------------
    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up output directory with versioning
    if args.empirical_fits_only:
        args.output_dir = Path(str(args.output_dir) + f'_{args.ppc_version}')
        logging.info(f"Using empirical fits only mode. Output will be saved to: {args.output_dir}")
        
        # Create output directory and save diagnostic info
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / 'diagnostic_info.txt', 'w') as f:
            f.write(f'Simulation Method: empirical_fits_only\n')
            f.write(f'Parameter Source: median of fitted posterior\n')
            f.write(f'Threshold Scale: {args.threshold_scale}\n')
            f.write(f'Seed: {args.seed}\n')
            f.write(f'Timestamp: {datetime.now().isoformat()}\n')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load empirical fits ---------------------------------------------------------------------
    logging.info(f"Loading empirical fits from {args.fitted_params_file}")
    try:
        # First, read the file as plain text to check its content
        with open(args.fitted_params_file, 'r') as f:
            first_few_lines = [next(f) for _ in range(5)]
        logging.debug(f"First few lines of raw file:\n{''.join(first_few_lines)}")
        
        # Now read with pandas
        df_fits = pd.read_csv(args.fitted_params_file)
        logging.info(f"Successfully loaded {len(df_fits)} rows from {args.fitted_params_file}")
        
        # Ensure subject column is string type
        if 'subject' in df_fits.columns:
            df_fits['subject'] = df_fits['subject'].astype(str).str.strip()
        
        # Log basic info about the fits file
        logging.info(f"DataFrame shape: {df_fits.shape}")
        logging.info(f"DataFrame columns: {df_fits.columns.tolist()}")
        logging.info(f"DataFrame dtypes:\n{df_fits.dtypes}")
        
        # Log first few rows with more detail
        logging.info("First few rows of data:")
        for i, row in df_fits.head().iterrows():
            logging.info(f"Row {i}:")
            for col in df_fits.columns:
                val = row[col]
                logging.info(f"  {col}: {val} (type: {type(val).__name__})")
        
        # Check for any non-numeric values in parameter columns
        param_columns = [col for col in df_fits.columns if '_mean' in col or '_std' in col]
        for col in param_columns:
            if col in df_fits.columns:
                sample_value = df_fits[col].iloc[0] if not df_fits.empty else None
                logging.info(f"Parameter column '{col}':")
                logging.info(f"  dtype = {df_fits[col].dtype}")
                logging.info(f"  sample value = {sample_value}")
                logging.info(f"  value type = {type(sample_value).__name__}")
                
                # Check if the value is a string that might contain JSON/dict
                if isinstance(sample_value, str) and ('{' in sample_value or '[' in sample_value):
                    logging.warning(f"Column '{col}' contains what looks like a JSON/dict string")
                
                # Try to convert to float to see if it works
                try:
                    if pd.notna(sample_value):
                        float_val = float(sample_value)
                        logging.info(f"  Successfully converted to float: {float_val}")
                except (ValueError, TypeError) as e:
                    logging.error(f"  Could not convert to float: {e}")
        
        # Ensure subject_id column exists and convert to string
        if 'subject_id' not in df_fits.columns:
            # Try to find a subject ID column with a different name
            possible_id_cols = [col for col in df_fits.columns if 'subj' in col.lower() or 'id' in col.lower()]
            if possible_id_cols:
                logging.warning(f"Using column '{possible_id_cols[0]}' as subject ID in fits file")
                df_fits['subject'] = df_fits[possible_id_cols[0]].astype(str).str.strip()
            else:
                raise ValueError("No subject ID column found in fitted parameters file")
        else:
            df_fits["subject"] = df_fits["subject_id"].astype(str).str.strip()
        
        # Log subject ID information
        unique_subjects = df_fits['subject'].unique()
        logging.info(f"Loaded fits for {len(unique_subjects)} unique subjects")
        logging.debug(f"First 10 subject IDs in fits: {unique_subjects[:10].tolist()}")
        logging.debug(f"Fits columns: {df_fits.columns.tolist()}")
        
        # Check for any non-numeric subject IDs
        non_numeric = [s for s in unique_subjects if not str(s).strip().isdigit()]
        if non_numeric:
            logging.warning(f"Found {len(non_numeric)} non-numeric subject IDs in fits file")
    
    except Exception as e:
        logging.error(f"Error loading empirical fits from {args.fitted_params_file}: {str(e)}")
        raise

    # Load empirical data ------------------------------------------------------------------
    if args.roberts_data_file is None:
        raise ValueError("--roberts_data_file must be provided (raw trial‑level data)")
        
    logging.info(f"Loading raw data from {args.roberts_data_file}")
    try:
        df_raw = pd.read_csv(args.roberts_data_file)
        # Ensure subject column is string type
        if 'subject' in df_raw.columns:
            df_raw['subject'] = df_raw['subject'].astype(str).str.strip()
        logging.info(f"Successfully loaded {len(df_raw)} rows from {args.roberts_data_file}")
        
        # Log basic info about the raw data
        logging.debug(f"Raw data columns: {df_raw.columns.tolist()}")
        logging.debug(f"First few rows of raw data:\n{df_raw.head().to_string()}")
        
        # Standardize column names (case-insensitive)
        col_mapping = {col: col.lower() for col in df_raw.columns}
        df_raw = df_raw.rename(columns=col_mapping)
        
        # Ensure required columns exist
        required_cols = ['trialtype', 'frame', 'cond', 'prob', 'choice', 'rt', 'subject']
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in raw data: {missing_cols}")
        
        # Filter for target trials only
        df_raw = df_raw[df_raw['trialtype'].str.lower() == 'target'].copy()
        
        # Convert data types
        df_raw['subject'] = df_raw['subject'].astype(str).str.strip()
        df_raw['choice'] = pd.to_numeric(df_raw['choice'], errors='coerce').fillna(0).astype(int)
        df_raw['rt'] = pd.to_numeric(df_raw['rt'], errors='coerce')
        df_raw['prob'] = pd.to_numeric(df_raw['prob'], errors='coerce')
        
        # Create derived columns
        df_raw['is_gain_frame'] = df_raw['frame'].str.lower() == 'gain'
        df_raw['time_constrained'] = df_raw['cond'].str.lower() == 'tc'
        
        # Calculate salience input (from probability)
        df_raw['salience_input'] = df_raw['prob']
        
        # Calculate norm input (+1 for gain, -1 for loss)
        df_raw['norm_input'] = df_raw['frame'].map({'gain': 1.0, 'loss': -1.0, '1': 1.0, '-1': -1.0, 'gainframe': 1.0, 'lossframe': -1.0})
        
        # Add valence score (placeholder - replace with actual ValenceProcessor if available)
        if 'valence_score' not in df_raw.columns:
            df_raw['valence_score'] = 0.0  # Default value if not available
            
        # Add norm category
        df_raw['norm_category_for_trial'] = 'default'
        
        # Log the preprocessed data
        logging.info(f"Preprocessed data shape: {df_raw.shape}")
        logging.debug(f"Preprocessed data columns: {df_raw.columns.tolist()}")
        logging.debug(f"Frame values: {df_raw['frame'].unique().tolist()}")
        logging.debug(f"Condition values: {df_raw['cond'].unique().tolist()}")
        logging.debug(f"Gain frame counts:\n{df_raw['is_gain_frame'].value_counts()}")
        logging.debug(f"Time constrained counts:\n{df_raw['time_constrained'].value_counts()}")
        
    except Exception as e:
        logging.error(f"Error loading or preprocessing raw data from {args.roberts_data_file}: {str(e)}", exc_info=True)
        raise
    
    # Debug: Print detailed subject info from raw data
    if 'subject' not in df_raw.columns:
        available_cols = df_raw.columns.tolist()
        logging.error(f"'subject' column not found in raw data. Available columns: {available_cols}")
        # Try to find potential subject ID columns
        possible_subj_cols = [col for col in available_cols if 'subj' in col.lower() or 'id' in col.lower()]
        if possible_subj_cols:
            logging.warning(f"Using column '{possible_subj_cols[0]}' as subject identifier")
            df_raw['subject'] = df_raw[possible_subj_cols[0]].astype(str).str.strip()
        else:
            raise ValueError("No subject identifier column found in raw data")
    
    # Map column names to expected names for Roberts framing data
    column_mapping = {}
    
    # For salience input, use 'gainColor' which indicates the frame (gain/loss)
    if 'gainColor' in df_raw.columns:
        column_mapping['salience_input'] = 'gainColor'
        logging.info("Using column 'gainColor' as salience_input")
    
    # For norm input, we'll use 'frame' which indicates the framing condition
    # This is a simplification - you may need to adjust this based on your model's needs
    if 'frame' in df_raw.columns:
        column_mapping['norm_input'] = 'frame'
        logging.info("Using column 'frame' as norm_input")
    
    # Check if we found the required columns
    required_columns = ['salience_input', 'norm_input']
    missing_columns = [col for col in required_columns if col not in column_mapping]
    
    if missing_columns:
        # If we're missing required columns, try to use default values
        logging.warning(f"Could not map all required columns, using defaults for: {missing_columns}")
        logging.warning(f"Available columns: {df_raw.columns.tolist()}")
        
        # Set default values for missing columns
        if 'salience_input' in missing_columns:
            logging.warning("No salience input column found, using default value of 1.0")
            df_raw['salience_input'] = 1.0
            
        if 'norm_input' in missing_columns:
            logging.warning("No norm input column found, using default value of 0.0")
            df_raw['norm_input'] = 0.0
    
    # Create new columns with expected names if they don't match
    for new_name, old_name in column_mapping.items():
        if new_name != old_name and new_name not in df_raw.columns:
            df_raw[new_name] = df_raw[old_name]  # Copy data to new column with expected name
    
    # Convert frame values to numeric if needed
    if 'frame' in df_raw.columns and df_raw['frame'].dtype == object:
        # Map frame values to numeric (e.g., 'gain' -> 1, 'loss' -> -1)
        frame_mapping = {'gain': 1.0, 'loss': -1.0}
        df_raw['norm_input'] = df_raw['frame'].map(lambda x: frame_mapping.get(str(x).lower().strip(), 0.0))
        logging.info(f"Converted frame values to numeric. Unique values: {df_raw['norm_input'].unique()}")
    
    # Convert gainColor to numeric (1 for gain, -1 for loss, 0 for anything else)
    if 'gainColor' in df_raw.columns:
        df_raw['gainColor'] = df_raw['gainColor'].astype(str).str.strip()
        color_mapping = {'gain': 1.0, 'loss': -1.0, '1': 1.0, '-1': -1.0}
        df_raw['salience_input'] = df_raw['gainColor'].str.lower().map(lambda x: color_mapping.get(x, 0.0))
        logging.info(f"Converted gainColor to numeric. Unique values: {df_raw['salience_input'].unique()}")
    
    # Ensure subject column exists and clean it
    df_raw['subject'] = df_raw['subject'].astype(str).str.strip()
    unique_subjects = df_raw['subject'].unique()
    logging.debug(f"First 10 subject IDs in raw data: {unique_subjects[:10].tolist()}")
    logging.debug(f"Total unique subjects in raw data: {len(unique_subjects)}")
    logging.debug(f"Raw data columns: {df_raw.columns.tolist()}")
    
    # Check for any non-numeric subject IDs
    non_numeric = [s for s in unique_subjects if not s.isdigit()]
    if non_numeric:
        logging.warning(f"Found {len(non_numeric)} non-numeric subject IDs in raw data")
        if non_numeric:
            logging.warning(f"Found {len(non_numeric)} non-numeric subject IDs in raw data")

    # Filter to target trials (customise per experiment) ------------------------------
    logging.debug(f"Raw data shape before filtering: {df_raw.shape}")
    logging.debug(f"Columns in raw data: {df_raw.columns.tolist()}")
    
    # The column is 'trialtype' (lowercase) after preprocessing
    if 'trialtype' not in df_raw.columns:
        logging.error("'trialtype' column not found in raw data. Available columns: %s", df_raw.columns.tolist())
    
    logging.debug(f"Unique trial types: {df_raw['trialtype'].unique().tolist() if 'trialtype' in df_raw.columns else 'N/A'}")
    
    df_raw = df_raw[df_raw["trialtype"] == "target"].copy()
    logging.debug(f"Data shape after filtering for target trials: {df_raw.shape}")

    # List of subjects to run ---------------------------------------------------------
    # Normalize all subject IDs to strings for safe comparison
    if 'subject' not in df_raw.columns:
        # Try to find a subject column with a different name
        possible_subj_cols = [col for col in df_raw.columns if 'subj' in col.lower() or 'id' in col.lower()]
        if possible_subj_cols:
            logging.warning(f"Using column '{possible_subj_cols[0]}' as subject identifier")
            df_raw['subject'] = df_raw[possible_subj_cols[0]].astype(str).str.strip()
        else:
            raise ValueError("No subject identifier column found in raw data")
    
    # Ensure subject column exists and clean it
    df_raw['subject'] = df_raw['subject'].astype(str).str.strip()
    all_subjects = sorted(df_raw['subject'].unique())
    logging.info(f"Found {len(all_subjects)} unique subjects in raw data")
    
    # Get subjects with fits
    subjects_with_fits = set(df_fits['subject'].unique())
    logging.info(f"Found {len(subjects_with_fits)} subjects with parameter fits")
    
    # Find intersection of subjects with data and fits
    valid_subjects = sorted(list(set(all_subjects) & subjects_with_fits))
    
    if args.subject_ids:
        # Handle user-specified subject IDs
        selected = [s.strip() for s in args.subject_ids.split(",") if s.strip()]
        subject_ids = [s for s in selected if s in valid_subjects]
        missing_ids = set(selected) - set(subject_ids)
        if missing_ids:
            logging.warning("Some requested subject IDs were not found in both data and fits: %s", ", ".join(missing_ids))
    else:
        # Use all subjects with both data and fits
        subject_ids = valid_subjects
    
    if not subject_ids:
        raise ValueError("No valid subjects found with both trial data and parameter fits")
        
    logging.info("Running PPC for %d subjects with both data and parameter fits", len(subject_ids))
    logging.debug("Subject IDs to process: %s", subject_ids[:10] + (["..."] if len(subject_ids) > 10 else []))

    # Run PPC per subject using empirical fits ----------------------------------------
    ppc_results: Dict[str, Dict[str, Dict]] = {}
    processed_count = 0
    total_subjects = len(subject_ids)
    
    # Set up progress bar
    with tqdm(total=total_subjects, desc="Running PPC", unit="subject") as pbar:
        for subj in subject_ids:
            subj = str(subj)  # Ensure subject ID is a string
            
            try:
                # Get empirical parameters for this subject
                try:
                    # Convert subject ID to string for consistent comparison
                    subj_str = str(subj).strip()
                    
                    # Debug: Log the types and values we're comparing
                    logging.debug(f"Looking for subject ID: '{subj_str}' (type: {type(subj_str)})")
                    logging.debug(f"Available subject IDs in fits (first 5): {df_fits['subject'].astype(str).str.strip().unique()[:5].tolist()}")
                    
                    # Try exact match first
                    row = df_fits[df_fits["subject"].astype(str).str.strip() == subj_str].squeeze()
                    
                    if row.empty:
                        # Try case-insensitive match if exact match fails
                        row = df_fits[df_fits["subject"].astype(str).str.strip().str.lower() == subj_str.lower()].squeeze()
                    # Always use mean values from empirical fits with error handling
                    try:
                        # Debug: Print the entire row to inspect its structure
                        logging.debug(f"Row data for subject {subj}: {row}")
                        logging.debug(f"Row type: {type(row)}")
                        logging.debug(f"Row index: {row.index.tolist() if hasattr(row, 'index') else 'No index'}")
                        
                        # Debug: Print the type of each value in the row
                        logging.debug("Row value types:")
                        for col in row.index:
                            logging.debug(f"  {col}: {type(row[col])} = {row[col]}")
                        
                        # Safely extract and convert each parameter with detailed error handling
                        def safe_get_float(row, keys, default=None):
                            for key in keys:
                                try:
                                    if key in row:
                                        value = row[key]
                                        logging.debug(f"Found {key} with value {value} (type: {type(value)})")
                                        if pd.isna(value):
                                            logging.warning(f"{key} is NA, using default {default}")
                                            return default
                                        return float(value)
                                except Exception as e:
                                    logging.warning(f"Error getting {key}: {e}")
                            return default
                        
                        # Extract parameters with fallbacks
                        base_threshold = safe_get_float(row, ["a_0_mean", "a_0"], 1.0)
                        w_s = safe_get_float(row, ["w_s_eff_mean", "w_s_eff"], 1.0)
                        t_0 = safe_get_float(row, ["t_0_mean", "t_0"], 0.3)
                        v_norm = safe_get_float(row, ["v_norm_mean", "v_norm"], 1.0)
                        alpha_gain = safe_get_float(row, ["alpha_gain_mean", "alpha_gain"], 1.0)
                        beta_val = safe_get_float(row, ["beta_val_mean", "beta_val"], 0.0)
                        
                        logging.info(f"Subject {subj}: Using mean parameter values")
                        logging.info(f"Extracted parameters - a_0: {base_threshold}, w_s: {w_s}, t_0: {t_0}, "
                                   f"v_norm: {v_norm}, alpha_gain: {alpha_gain}, beta_val: {beta_val}")
                        
                    except Exception as e:
                        logging.error(f"Unexpected error processing parameters for subject {subj}: {str(e)}", exc_info=True)
                        logging.error(f"Row data: {row}")
                        pbar.update(1)
                        continue
                        
                except Exception as e:
                    logging.error(f"Error processing subject {subj}: {str(e)}", exc_info=True)
                    pbar.update(1)
                    continue
                
                # Debug: Print the types of all parameters before conversion
                logging.debug("Parameter types before conversion:")
                logging.debug(f"  w_s: {type(w_s)} = {w_s}")
                logging.debug(f"  v_norm: {type(v_norm)} = {v_norm}")
                logging.debug(f"  base_threshold: {type(base_threshold)} = {base_threshold}")
                logging.debug(f"  t_0: {type(t_0)} = {t_0}")
                logging.debug(f"  alpha_gain: {type(alpha_gain)} = {alpha_gain}")
                logging.debug(f"  beta_val: {type(beta_val)} = {beta_val}")
                
                # Apply threshold scaling
                try:
                    base_threshold = float(base_threshold) * float(args.threshold_scale)
                    logging.info(f"Subject {subj}: Threshold scaled to {base_threshold:.3f} (scale={args.threshold_scale})")
                except (TypeError, ValueError) as e:
                    logging.error(f"Error scaling threshold: {e}")
                    logging.error(f"base_threshold type: {type(base_threshold)}, value: {base_threshold}")
                    logging.error(f"args.threshold_scale type: {type(args.threshold_scale)}, value: {args.threshold_scale}")
                    raise
                
                # Create parameters dictionary for the simulation
                try:
                    # Create params dictionary for the simulation function
                    # These keys should match what agent.run_mvnes_trial expects in its 'params' argument
                    # or what simulate_one_full_dataset will use to construct that.
                    params_dict = {
                        # Core DDM parameters (using agent's internal attribute names)
                        'w_s': float(w_s),
                        'w_n': float(v_norm),
                        'threshold_a': float(base_threshold),
                        't': float(t_0),
                        'alpha_gain': float(alpha_gain),
                        'beta_val': float(beta_val),  # Added beta_val

                        # Fixed parameters (can override agent defaults if needed for specific sim runs)
                        'logit_z0': 0.0,
                        'log_tau_norm': -0.693147,  # Default log(0.5)
                        'meta_cognitive_on': False,


                        # Simulation-specific parameters for run_mvnes_trial's 'params' argument
                        'noise_std_dev': 1.0,
                        'dt': 0.01,
                        # 'max_time': 2.0, # max_time is better determined per-trial inside simulate_one_full_dataset

                        # Other informational items
                        'subject': subj,
                        'debug': args.debug
                    }
                    
                    logging.debug(f"Parameters for subject {subj}: {params_dict}")
                    
                except Exception as e:
                    logging.error(f"Error creating parameters for subject {subj}: {str(e)}")
                    logging.error(f"Parameter types - w_s: {type(w_s)}, w_n: {type(v_norm)}, threshold_a: {type(base_threshold)}, t: {type(t_0)}")
                    logging.error(f"Parameter values - w_s: {w_s}, w_n: {v_norm}, threshold_a: {base_threshold}, t: {t_0}")
                    raise
                
                # Initialize the agent with the parameters
                agent = MVNESAgent()
                logging.debug(f"Initialized agent for subject {subj}")
                logging.debug(f"Parameters for simulation: {params_dict}")
                
                # Update agent parameters with the current subject's parameters
                agent.w_s = float(params_dict['w_s'])
                agent.w_n = float(params_dict['w_n'])
                agent.threshold_a = float(params_dict['threshold_a'])
                agent.t = float(params_dict['t'])
                agent.alpha_gain = float(params_dict.get('alpha_gain', 1.0))
                agent.beta_val = float(params_dict.get('beta_val', 0.0))
                
                # Get trial structure for this subject with more robust filtering
                try:
                    # First try exact match
                    df_subj = df_raw[df_raw["subject"] == subj].copy()
                    
                    # If no exact match, try case-insensitive match
                    if df_subj.empty:
                        logging.debug(f"No exact match for subject {subj}, trying case-insensitive match")
                        df_subj = df_raw[df_raw["subject"].str.lower() == subj.lower()].copy()
                    
                    # If still no match, try stripping any whitespace or special characters
                    if df_subj.empty:
                        logging.debug(f"No case-insensitive match for subject {subj}, trying with stripped IDs")
                        df_subj = df_raw[df_raw["subject"].str.strip(" \t\n") == subj.strip(" \t\n")].copy()
                    
                    if df_subj.empty:
                        logging.warning(f"No trial data found for subject {subj} after multiple matching attempts")
                        logging.debug(f"Available subjects in raw data: {sorted(df_raw['subject'].unique().tolist())}")
                        pbar.update(1)
                        continue
                        
                    logging.debug(f"Found {len(df_subj)} trials for subject {subj}")
                    
                    # Get trial structure with error handling
                    try:
                        # Log available columns and data types for debugging
                        logging.debug(f"Available columns for subject {subj}: {df_subj.columns.tolist()}")
                        logging.debug(f"Data types for subject {subj}:\n{df_subj.dtypes}")
                        
                        # Create a copy of all columns to preserve the trial structure
                        trial_data = df_subj.copy()
                        
                        # Log initial data summary
                        logging.debug(f"Initial data for subject {subj}:\n{trial_data.head()}")
                        logging.debug(f"Initial data shape: {trial_data.shape}")
                        
                        # Ensure required columns exist and have correct data types
                        required_cols = ['prob', 'is_gain_frame', 'frame', 'cond', 'time_constrained', 'valence_score', 'norm_category_for_trial']
                        for col in required_cols:
                            if col not in trial_data.columns:
                                raise ValueError(f"Required column '{col}' not found in trial data for subject {subj}")
                            
                            # Convert to appropriate data type if needed
                            if col in ['prob', 'valence_score'] and not pd.api.types.is_numeric_dtype(trial_data[col]):
                                logging.warning(f"Converting {col} to numeric for subject {subj}")
                                trial_data[col] = pd.to_numeric(trial_data[col], errors='coerce')
                            elif col == 'is_gain_frame' and not pd.api.types.is_bool_dtype(trial_data[col]):
                                logging.warning(f"Converting {col} to boolean for subject {subj}")
                                trial_data[col] = trial_data[col].astype(bool)
                        
                        # Log data after type conversion
                        logging.debug(f"Data after type conversion for subject {subj}:\n{trial_data.head()}")
                        
                        # Check for missing values before dropping
                        if trial_data.isnull().any().any():
                            missing_counts = trial_data.isnull().sum()
                            logging.warning(f"Found missing values for subject {subj}:\n{missing_counts}")
                            
                            # Try to fill missing values with column means if possible
                            for col in trial_data.columns:
                                if trial_data[col].isnull().any():
                                    if trial_data[col].notna().any():  # If there are some non-null values
                                        if pd.api.types.is_numeric_dtype(trial_data[col]):
                                            col_mean = trial_data[col].mean()
                                            trial_data[col] = trial_data[col].fillna(col_mean)
                                            logging.warning(f"Filled missing values in numeric column '{col}' with mean {col_mean:.2f}")
                                        else:
                                            # For non-numeric columns, NaNs will be handled by the subsequent dropna()
                                            # or by get_trial_structure if the column is one of its required_cols.
                                            logging.debug(f"Non-numeric column '{col}' has NaNs; will not be filled with mean here.")
                        
                        # Drop any remaining rows with missing values
                        n_before = len(trial_data)
                        trial_data = trial_data.dropna()
                        n_after = len(trial_data)
                        
                        if n_after < n_before:
                            logging.warning(f"Dropped {n_before - n_after} rows with missing values for subject {subj} (kept {n_after} rows)")
                        
                        if len(trial_data) == 0:
                            logging.error(f"No valid trial data for subject {subj} after cleaning")
                            logging.error(f"Original data shape: {df_subj.shape}")
                            logging.error(f"Sample of problematic data:\n{df_subj.head().to_dict()}")
                            pbar.update(1)
                            continue
                            
                        trial_struct = get_trial_structure(trial_data)
                        if trial_struct is None or len(trial_struct) == 0:
                            logging.warning(f"Empty trial structure for subject {subj}, skipping")
                            pbar.update(1)
                            continue
                            
                        # --- CALCULATE OBSERVED SUMMARY STATS ---
                        try:
                            # Use the cleaned trial_data DataFrame that get_trial_structure was based on
                            observed_summary_stats = calculate_summary_stats(trial_data, ROBERTS_SUMMARY_STAT_KEYS)
                            logging.info(f"Calculated observed summary stats for subject {subj}")
                        except Exception as e:
                            logging.error(f"Error calculating OBSERVED summary stats for subject {subj}: {e}", exc_info=True)
                            pbar.update(1)
                            continue  # Skip subject if observed stats fail
                        # --- END OBSERVED STATS ---
                    except Exception as e:
                        logging.error(f"Error creating trial structure for subject {subj}: {str(e)}")
                        logging.debug(f"Subject {subj} data sample: {df_subj.head().to_dict()}")
                        pbar.update(1)
                        continue
                        
                except Exception as e:
                    logging.error(f"Unexpected error processing subject {subj}: {str(e)}", exc_info=True)
                    pbar.update(1)
                    continue
                
                # Run simulations using the comprehensive params_dict defined earlier
                logging.info(f"Running {args.n_sim_reps} simulations for subject {subj}")
                results = simulate_n_replicates(
                    params_dict, 
                    trial_struct, 
                    agent, 
                    n_reps=args.n_sim_reps
                )
                
                # Process results
                try:
                    if not results:  # Check if we have any results
                        logging.warning(f"No valid simulation results for subject {subj}")
                        pbar.update(1)
                        continue
                        
                    df_sim = pd.DataFrame(results)
                    
                    # Ensure we have the required columns for stats calculation
                    required_cols = ['choice', 'rt', 'frame', 'cond', 'prob']
                    missing_cols = [col for col in required_cols if col not in df_sim.columns]
                    if missing_cols:
                        logging.warning(f"Missing required columns {missing_cols} in simulation results for subject {subj}")
                        pbar.update(1)
                        continue
                        
                    # Ensure numeric types for calculations
                    df_sim['choice'] = pd.to_numeric(df_sim['choice'], errors='coerce')
                    df_sim['rt'] = pd.to_numeric(df_sim['rt'], errors='coerce')
                    df_sim['prob'] = pd.to_numeric(df_sim['prob'], errors='coerce')
                    
                    # Drop any rows with invalid data
                    df_sim = df_sim.dropna(subset=['choice', 'rt', 'prob'])
                    
                    if df_sim.empty:
                        logging.warning(f"No valid data after cleaning for subject {subj}")
                        pbar.update(1)
                        continue
                        
                    # Calculate summary statistics
                    all_stats = calculate_summary_stats(df_sim, ROBERTS_SUMMARY_STAT_KEYS)
                    sim_stats_list.append(all_stats)
                    
                except Exception as e:
                    logging.error(f"Error processing simulation results for subject {subj}: {e}", exc_info=True)
                    pbar.update(1)
                    continue
                    
                # Combine results
                df_sim = pd.DataFrame(sim_stats_list)
                
                # Store results with the correct structure for PPC analysis
                ppc_results[subj] = {
                    'observed_summary_stats': observed_summary_stats,
                    'simulated_summary_stats_list': sim_stats_list,  # List of dicts from simulate_n_replicates
                    'params_used_for_sim': params_dict,
                    'trial_data': df_subj.to_dict('records')  # Keep original data for reference
                }
                
                processed_count += 1
                pbar.update(1)
                
            except Exception as e:
                logging.error(f"Error processing subject {subj}: {str(e)}", exc_info=args.debug)
                pbar.update(1)
                continue
        
        logging.info(f"Successfully processed {processed_count}/{total_subjects} subjects")
        
        # Save results
        if ppc_results:
            output_file = args.output_dir / 'ppc_results.json'
            with open(output_file, 'w') as f:
                json.dump(ppc_results, f, indent=2)
            logging.info(f"Saved PPC results to {output_file}")
            
            # Generate and save plots
            plot_ppc_distributions(ppc_results, args.output_dir)
            
            # Calculate and print coverage statistics
            coverage_df = calculate_detailed_coverage(ppc_results)
            if not coverage_df.empty:
                coverage_file = args.output_dir / 'coverage_statistics.csv'
                coverage_df.to_csv(coverage_file, index=False)
                logging.info(f"Saved coverage statistics to {coverage_file}")
                
                # Print summary of coverage statistics
                agg_coverage = coverage_df.groupby('statistic').agg({
                    'coverage': 'mean',
                    'z_score': 'mean',
                    'p_value': lambda x: (x < 0.05).mean()
                }).reset_index()
                
                logging.info("\nCoverage Statistics Summary:")
                logging.info("-" * 80)
                logging.info(f"{'Statistic':<30} {'Coverage':<15} {'Mean |Z|':<15} 'p<0.05'")
                logging.info("-" * 80)
                for _, row in agg_coverage.iterrows():
                    logging.info(f"{row['statistic']:<30} {row['coverage']:<15.3f} {row['z_score']:<15.3f} {row['p_value']:.3f}")
                logging.info("-" * 80)
        else:
            logging.warning("No valid results to save")
            
    logging.info(f"Successfully processed {processed_count}/{total_subjects} subjects")

    # Check for empty results before calculating coverage
    if not ppc_results:
        logging.warning("No subjects were successfully processed. Exiting early.")
        return
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_file = args.output_dir / 'ppc_results.json'
    with open(output_file, 'w') as f:
        json.dump(ppc_results, f, indent=2)
    logging.info(f"Saved PPC results to {output_file}")
    
    # Generate and save plots
    plot_ppc_distributions(ppc_results, args.output_dir)
    
    # Calculate and save coverage statistics
    coverage_df = calculate_detailed_coverage(ppc_results)
    if not coverage_df.empty:
        coverage_file = args.output_dir / 'coverage_statistics.csv'
        coverage_df.to_csv(coverage_file, index=False)
        logging.info(f"Saved coverage statistics to {coverage_file}")
        
        # Print summary of coverage statistics
        agg_coverage = coverage_df.groupby('statistic').agg({
            'coverage': 'mean',
            'z_score': 'mean',
            'p_value': lambda x: (x < 0.05).mean()
        }).reset_index()
        
        logging.info("\nFinal Coverage Statistics Summary:")
        logging.info("-" * 80)
        logging.info(f"{'Statistic':<30} {'Coverage':<15} {'Mean |Z|':<15} 'p<0.05'")
        logging.info("-" * 80)
        for _, row in agg_coverage.iterrows():
            logging.info(f"{row['statistic']:<30} {row['coverage']:<15.3f} {row['z_score']:<15.3f} {row['p_value']:.3f}")
        logging.info("-" * 80)
    
    return ppc_results


def test_agent_variability(agent, params_dict):
    """Test the agent's stochastic behavior with given parameters.
    
    Args:
        agent: An instance of MVNESAgent
        params_dict: Dictionary of model parameters
    """
    trial_input = {
        "prob": 0.35,
        "frame": "gain",
        "cond": "ntc",
        "trialType": "target"
    }

    def trial_to_inputs(trial):
        """Convert trial dictionary to agent inputs."""
        salience_input = trial["prob"]
        norm_input = 1.0 if trial["frame"].lower() == "gain" else -1.0
        return salience_input, norm_input

    salience_input, norm_input = trial_to_inputs(trial_input)

    print("\n=== Testing Agent Variability ===")
    print("Inputs:")
    print(f"  salience_input = {salience_input}")
    print(f"  norm_input     = {norm_input}")
    print("\nParameters:")
    for k, v in params_dict.items():
        print(f"  {k}: {v:.4f}")

    # Run multiple trials
    choices, rts = [], []
    for _ in range(100):
        out = agent.run_mvnes_trial(
            salience_input=salience_input,
            norm_input=norm_input,
            params=params_dict
        )
        choices.append(out["choice"])
        rts.append(out["rt"])

    # Analyze results
    from collections import Counter
    choice_counts = Counter(choices)
    rt_mean = np.mean(rts)
    rt_std = np.std(rts)
    
    print("\nResults (100 trials):")
    print(f"Choice distribution: {dict(choice_counts)}")
    print(f"RT - Mean: {rt_mean:.4f}, Std: {rt_std:.4f}")
    
    # Diagnostic checks
    print("\nDiagnostics:")
    if len(choice_counts) == 1:
        print("❌ Agent is deterministic - always makes the same choice")
    else:
        print("✅ Agent shows stochastic choice behavior")
        
    if rt_std < 0.1:
        print(f"❌ Low RT variability (std = {rt_std:.4f}) - check noise parameters")
    else:
        print(f"✅ Good RT variability (std = {rt_std:.4f})")
    
    # Check if choices align with expected framing effect
    if choice_counts.get(1, 0) > choice_counts.get(0, 0):
        print("⚠️  Agent tends to gamble in gain frame - check norm_input sign")
    else:
        print("✅ Choice pattern aligns with expected framing effect")


def test_with_subject_params():
    """Test agent with parameters from a specific subject."""
    # Example parameters from subject 115 with all required parameters for MVNESAgent
    params_dict = {
        # Core parameters from empirical fits
        "v_norm": 0.6363,      # Normalization value (not directly used in agent)
        "a_0": 1.3871,         # Initial threshold
        "w_s_eff": 1.4529,     # Effective salience weight (will be mapped to w_s)
        "t_0": 0.3457,         # Non-decision time
        "alpha_gain": 0.6875,  # Gain frame modulation
        
        # Required parameters for MVNESAgent
        'w_s': 1.4529,         # Salience weight (using w_s_eff value)
        'w_n': 1.0,            # Norm weight (default)
        'threshold_a': 1.3871,  # Decision threshold (using a_0 value)
        't': 0.3457,           # Non-decision time (using t_0 value)
        'noise_std_dev': 0.1,   # Noise standard deviation
        'dt': 0.01,            # Simulation time step
        'max_time': 2.0,       # Maximum simulation time
        'affect_stress': False  # Stress condition flag
    }
    
    agent = MVNESAgent()
    test_agent_variability(agent, params_dict)


if __name__ == "__main__":  # pragma: no cover
    # Uncomment to run agent variability test
    # test_with_subject_params()
    main()
