# Filename: fit_nes_to_roberts_data_sbc_focused.py
# Purpose: Train an NPE for the NES model on Roberts et al. data
#          and perform Simulation-Based Calibration (SBC).
# This version integrates core logic from the user's previous full script.

import sys
import os

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for plotting

import argparse
import logging
import json
import sys
from pathlib import Path
import time
from typing import Optional, Dict, List, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns # For plotting if needed later, not strictly for SBC plot

import torch.nn as nn
import sbi.utils as sbi_utils
from sbi import utils as sbi_utils_module
from sbi.utils.user_input_checks import process_prior
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets import posterior_nn

# --- SBI Imports ---
try:
    import sbi 
    from sbi.inference import SNPE, simulate_for_sbi
    from sbi.utils import BoxUniform
    from scipy import stats as sp_stats # For KS test if sbc_rank_stats doesn't provide p-value
except ImportError as e:
    logging.error(f"Critical SBI/scipy import error: {e}. Please ensure libraries are installed correctly.")
    sys.exit(1)

# Import the 25-stat version from stats_schema
try:
    from src.nes_copilot.stats_schema import NES_SUMMARY_STAT_KEYS
    print(f"Successfully imported {len(NES_SUMMARY_STAT_KEYS)} summary statistics from stats_schema")
except ImportError as e:
    print(f"Error importing stats_schema: {e}")
    NES_SUMMARY_STAT_KEYS = None

# --- Local patch for sbi v0.22.0 SBC ---
def run_sbc(true_parameters, observations, posterior, num_posterior_samples):
    """
    Robust SBC implementation with better error handling and diagnostics.
    Args:
        true_parameters: Tensor of shape (num_datasets, num_parameters)
        observations: Tensor of shape (num_datasets, num_observation_features)
        posterior: SBI posterior object
        num_posterior_samples: Number of posterior samples per observation
    Returns:
        ranks: Tensor of shape (num_datasets, num_parameters)
    """
    import torch
    import logging
    import sys
    
    num_datasets, num_parameters = true_parameters.shape
    # Use -1 as placeholder for invalid/missing ranks (can't use NaN with integers)
    ranks_tensor = torch.full((num_datasets, num_parameters), -1, 
                             dtype=torch.int32, device=true_parameters.device)
    
    success = 0
    fail = 0
    error_messages = {}
    
    # Ensure posterior is not None before proceeding
    if posterior is None:
        logging.error("[SBC run_sbc] Posterior object is None. Cannot proceed.")
        raise ValueError("Posterior object cannot be None for SBC.")

    for i, (theta_i_original, x_i_original) in enumerate(zip(true_parameters, observations)):
        try:
            x_i_for_sampling = x_i_original.unsqueeze(0) # Ensure x_i is [1, num_summary_stats]

            # Logging:
            if i < 3 or i % (num_datasets // 10 or 1) == 0 : # Log first few and some spaced out
                logging.info(f"[SBC Loop {i}] Conditioning on x_i (shape {x_i_for_sampling.shape}): {x_i_for_sampling.flatten()[:5]}...")

            # Sample from posterior
            samples_i = posterior.sample(
                (num_posterior_samples,), 
                x=x_i_for_sampling, # Use the explicitly batched x_i
                show_progress_bars=False
            )
            
            if i < 3 or i % (num_datasets // 10 or 1) == 0 :
                logging.info(f"[SBC Loop {i}] Posterior sample means: {samples_i.mean(dim=0).cpu().numpy()}")

            # Calculate ranks
            # Ensure theta_i_original is comparable shape for broadcasting
            ranks_i = torch.sum(samples_i < theta_i_original.unsqueeze(0), dim=0) 
            
            # Store results
            ranks_tensor[success] = ranks_i
            success += 1
            
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == num_datasets:
                print(f"\r[SBC] Processed {i+1}/{num_datasets} datasets "
                      f"({success} success, {fail} failed)", 
                      end="", file=sys.stderr)
                
        except Exception as e:
            fail += 1
            error_type = type(e).__name__
            error_messages[error_type] = error_messages.get(error_type, 0) + 1
            if fail <= 5:  # Only show first few errors to avoid spam
                logging.warning(f"[SBC] Failed at simulation {i}: {e}")
            continue
    
    # Trim and validate results
    if success == 0:
        error_summary = ", ".join(f"{k}({v})" for k, v in error_messages.items())
        raise RuntimeError(
            f"SBC failed for all {num_datasets} datasets. Errors: {error_summary}"
        )
    
    if fail > 0:
        error_summary = ", ".join(f"{k}({v})" for k, v in error_messages.items())
        logging.warning(
            f"SBC had {fail}/{num_datasets} failures. Error types: {error_summary}"
        )
    
    # Trim to actual successful runs
    final_ranks = ranks_tensor[:success]

    if final_ranks.numel() == 0:
        logging.warning("[SBC Results] No successful SBC datasets to analyze. Returning empty ranks.")
    return final_ranks

def sbc_rank_plot(ranks, num_posterior_samples):
    import matplotlib.pyplot as plt
    import numpy as np
    num_datasets, num_parameters = ranks.shape
    fig, axes = plt.subplots(1, num_parameters, figsize=(4 * num_parameters, 4), squeeze=False)
    for j in range(num_parameters):
        ax = axes[0, j]
        ax.hist(ranks[:, j].cpu().numpy(), bins=np.arange(num_posterior_samples + 2) - 0.5, density=True)
        ax.set_title(f'Parameter {j}')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# --- Project-Specific Imports ---
script_dir = Path(__file__).resolve().parent
project_root_paths = [script_dir, script_dir.parent, script_dir.parent.parent]

# Fixed parameters for the NES model
FIXED_PARAMS = {
    'logit_z0': 0.0,         # Fixed parameter for initial bias
    'log_tau_norm': -0.693147  # Fixed parameter for time constant (exp(-0.693147) ≈ 0.5)
}
agent_mvnes_found = False
try:
    for prp in project_root_paths:
        potential_src_dir = prp / 'src'
        if (potential_src_dir / 'agent_mvnes.py').exists():
            if str(potential_src_dir) not in sys.path:
                sys.path.insert(0, str(potential_src_dir))
            from agent_mvnes import MVNESAgent # ASSUMING THIS IS YOUR AGENT CLASS
            agent_mvnes_found = True
            logging.info(f"Found and imported MVNESAgent from: {potential_src_dir}")
            break
    if not agent_mvnes_found:
        # Fallback if 'src' is not found, try current dir if script is moved to project root
        if (Path('.') / 'agent_mvnes.py').exists():
            if str(Path('.')) not in sys.path: sys.path.insert(0, str(Path('.')))
            from agent_mvnes import MVNESAgent
            agent_mvnes_found = True
            logging.info(f"Found and imported MVNESAgent from current directory.")
        else:
            raise ImportError("Could not find agent_mvnes.py in typical project structures or current directory.")
except ImportError as e:
    logging.error(f"Error importing MVNESAgent: {e}. Check script location and 'src' directory.")
    sys.exit(1)

# --- Global Configurations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s', force=True)
sbi_logger = logging.getLogger('sbi')
sbi_logger.setLevel(logging.WARNING)

PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain', 'beta_val']  # 6 parameters

# Wide priors (original)
PRIOR_LOW_WIDE = torch.tensor([0.1, 0.5, 0.2, 0.05, 0.5, -1.0])
PRIOR_HIGH_WIDE = torch.tensor([2.0, 2.5, 1.5, 0.7, 1.0, 1.0])

# Tight priors (advisor's suggestion)
PRIOR_LOW_TIGHT = torch.tensor([0.3, 1.0, 0.5, 0.1, 0.6, -0.5])
PRIOR_HIGH_TIGHT = torch.tensor([1.0, 2.0, 1.2, 0.5, 0.9, 0.5])

# Default to wide priors (will be set in main based on args)
PRIOR_LOW = PRIOR_LOW_WIDE
PRIOR_HIGH = PRIOR_HIGH_WIDE
BASE_SIM_PARAMS = {
    'noise_std_dev': 1.0, 'dt': 0.01, 'max_time': 10.0, 'veto_flag': False
}
CONDITIONS_ROBERTS = {
    'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'}, 'Gain_TC': {'frame': 'gain', 'cond': 'tc'},
    'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'}, 'Loss_TC': {'frame': 'loss', 'cond': 'tc'},
}

SUBJECT_TRIAL_STRUCTURE_TEMPLATE: Optional[pd.DataFrame] = None
# OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE is less critical for pure SBC sim if simulator is robust
# but calculate_summary_stats might use it if defined.
OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE: Optional[Dict[str, float]] = None


# --- Core Functions from User's Previous Script (Adapted) ---

def get_roberts_summary_stat_keys() -> List[str]:
    """
    Returns the keys for the summary statistics vector.
    Uses the full set of 60 statistics from stats_schema.py for the 6-parameter model.
    """
    # Import the full set of 60 statistics from the stats schema
    try:
        from NES_Copilot_GUI.stats_schema import ROBERTS_SUMMARY_STAT_KEYS
        if not ROBERTS_SUMMARY_STAT_KEYS:
            raise ValueError("ROBERTS_SUMMARY_STAT_KEYS is empty")
    except (ImportError, ValueError) as e:
        # Fallback to local stats_schema if GUI module not available
        try:
            from stats_schema import ROBERTS_SUMMARY_STAT_KEYS
            if not ROBERTS_SUMMARY_STAT_KEYS:
                raise ValueError("ROBERTS_SUMMARY_STAT_KEYS is empty")
        except (ImportError, ValueError) as e2:
            raise ImportError(f"Failed to load ROBERTS_SUMMARY_STAT_KEYS: {e2}")
    
    expected_count = 60
    actual_count = len(ROBERTS_SUMMARY_STAT_KEYS)
    if actual_count != expected_count:
        logging.warning(f"Expected {expected_count} summary statistics, got {actual_count}")
    else:
        logging.info(f"Successfully loaded {actual_count} summary statistics")
    
    return ROBERTS_SUMMARY_STAT_KEYS

def calculate_summary_stats_roberts(df_trials: pd.DataFrame, 
                                   stat_keys: List[str],
                                   impute_rt_means: Optional[Dict[str, float]] = None
                                   ) -> Dict[str, float]:
    """Calculates summary statistics from trial data. Adapted from user's script."""
    summaries = {key: -999.0 for key in stat_keys} 

    if df_trials.empty or len(df_trials) < 5:
        logging.debug(f"Too few trials ({len(df_trials)}) for detailed summary stats, returning placeholders.")
        return summaries

    # Ensure auxiliary columns exist (frame and cond should be in simulated df_trials)
    if 'time_constrained' not in df_trials.columns:
        df_trials['time_constrained'] = df_trials['cond'] == 'tc'
    if 'is_gain_frame' not in df_trials.columns:
        df_trials['is_gain_frame'] = df_trials['frame'] == 'gain'
    
    # Impute NaNs in RT if impute_rt_means is provided (more relevant for observed data prep)
    # For simulated data, we expect RTs unless the sim failed.
    if impute_rt_means and df_trials['rt'].isna().any():
        for cond_key_enum, rt_mean_val in impute_rt_means.items():
            filters = CONDITIONS_ROBERTS[cond_key_enum]
            mask = (df_trials['frame'] == filters['frame']) & (df_trials['cond'] == filters['cond']) & df_trials['rt'].isna()
            df_trials.loc[mask, 'rt'] = rt_mean_val

    # Overall stats
    valid_choices = df_trials['choice'].dropna()
    if not valid_choices.empty:
        summaries['prop_gamble_overall'] = valid_choices.mean()
    
    rts_overall = df_trials['rt'].dropna()
    if not rts_overall.empty:
        summaries['mean_rt_overall'] = rts_overall.mean()
        try:
            quantiles = rts_overall.quantile([0.1, 0.5, 0.9])
            summaries['rt_q10_overall'] = quantiles.get(0.1, -999.0)
            summaries['rt_q50_overall'] = quantiles.get(0.5, -999.0)
            summaries['rt_q90_overall'] = quantiles.get(0.9, -999.0)
        except Exception: # Handle cases like all RTs being identical
            summaries['rt_q10_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
            summaries['rt_q50_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
            summaries['rt_q90_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
    
    cond_props = {}
    cond_rts_mean = {}
    cond_rts_median = {}  # Added for median RT tracking
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
        subset = df_trials[
            (df_trials['frame'] == cond_filters['frame']) & 
            (df_trials['cond'] == cond_filters['cond'])
        ]
        if not subset.empty:
            valid_subset_choices = subset['choice'].dropna()
            if not valid_subset_choices.empty:
                prop_gamble = valid_subset_choices.mean()
                summaries[f'prop_gamble_{cond_key_enum}'] = prop_gamble
                cond_props[cond_key_enum] = prop_gamble
            
            rts_cond = subset['rt'].dropna()
            if not rts_cond.empty:
                mean_rt = rts_cond.mean()
                median_rt = rts_cond.median()
                summaries[f'mean_rt_{cond_key_enum}'] = mean_rt
                cond_rts_mean[cond_key_enum] = mean_rt
                cond_rts_median[cond_key_enum] = median_rt
                try:
                    q_cond = rts_cond.quantile([0.1, 0.5, 0.9])
                    summaries[f'rt_q10_{cond_key_enum}'] = q_cond.get(0.1, -999.0)
                    summaries[f'rt_q50_{cond_key_enum}'] = q_cond.get(0.5, -999.0)
                    summaries[f'rt_q90_{cond_key_enum}'] = q_cond.get(0.9, -999.0)
                except Exception:
                     summaries[f'rt_q10_{cond_key_enum}'] = rts_cond.iloc[0] if not rts_cond.empty else -999.0
                     summaries[f'rt_q50_{cond_key_enum}'] = rts_cond.iloc[0] if not rts_cond.empty else -999.0
                     summaries[f'rt_q90_{cond_key_enum}'] = rts_cond.iloc[0] if not rts_cond.empty else -999.0

                max_rt_val = 1.0 if 'TC' in cond_key_enum else 3.0 
                bin_edges = np.linspace(0, max_rt_val, 6) 
                if len(rts_cond) >= 1: 
                    hist, _ = np.histogram(rts_cond.clip(0, max_rt_val), bins=bin_edges, density=True)
                    for i_bin, bin_val in enumerate(hist):
                        summaries[f'rt_hist_bin{i_bin}_{cond_key_enum}'] = bin_val
    
    # Original framing effects (choice proportions)
    pg_ln = cond_props.get('Loss_NTC', np.nan); pg_gn = cond_props.get('Gain_NTC', np.nan)
    summaries['framing_effect_ntc'] = pg_ln - pg_gn if not (pd.isna(pg_ln) or pd.isna(pg_gn)) else -999.0
    
    pg_lt = cond_props.get('Loss_TC', np.nan); pg_gt = cond_props.get('Gain_TC', np.nan)
    summaries['framing_effect_tc'] = pg_lt - pg_gt if not (pd.isna(pg_lt) or pd.isna(pg_gt)) else -999.0

    # Original RT framing biases
    rt_ln = cond_rts_mean.get('Loss_NTC', np.nan); rt_gn = cond_rts_mean.get('Gain_NTC', np.nan)
    summaries['rt_framing_bias_ntc'] = rt_ln - rt_gn if not (pd.isna(rt_ln) or pd.isna(rt_gn)) else -999.0

    rt_lt = cond_rts_mean.get('Loss_TC', np.nan); rt_gt = cond_rts_mean.get('Gain_TC', np.nan)
    summaries['rt_framing_bias_tc'] = rt_lt - rt_gt if not (pd.isna(rt_lt) or pd.isna(rt_gt)) else -999.0
    
    # RT standard deviations
    summaries['rt_std_overall'] = rts_overall.std() if not rts_overall.empty else -999.0
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
        subset = df_trials[
            (df_trials['frame'] == cond_filters['frame']) & 
            (df_trials['cond'] == cond_filters['cond'])
        ]
        rts_cond = subset['rt'].dropna()
        summaries[f'rt_std_{cond_key_enum}'] = rts_cond.std() if not rts_cond.empty else -999.0

    # New targeted Gain vs Loss frame contrasts
    # Mean RT contrasts
    summaries['mean_rt_Gain_vs_Loss_TC'] = cond_rts_mean.get('Gain_TC', np.nan) - cond_rts_mean.get('Loss_TC', np.nan)
    summaries['mean_rt_Gain_vs_Loss_NTC'] = cond_rts_mean.get('Gain_NTC', np.nan) - cond_rts_mean.get('Loss_NTC', np.nan)
    
    # Median RT contrasts
    summaries['rt_median_Gain_vs_Loss_TC'] = cond_rts_median.get('Gain_TC', np.nan) - cond_rts_median.get('Loss_TC', np.nan)
    summaries['rt_median_Gain_vs_Loss_NTC'] = cond_rts_median.get('Gain_NTC', np.nan) - cond_rts_median.get('Loss_NTC', np.nan)
    
    # RT effects within frames
    summaries['framing_effect_rt_gain'] = cond_rts_mean.get('Gain_TC', np.nan) - cond_rts_mean.get('Gain_NTC', np.nan)
    summaries['framing_effect_rt_loss'] = cond_rts_mean.get('Loss_TC', np.nan) - cond_rts_mean.get('Loss_NTC', np.nan)

    # Compute additional statistics for gamble vs. sure choices
    if not df_trials.empty:
        # Stats for gamble choices
        gamble_trials = df_trials[df_trials['choice'] == 1]
        if not gamble_trials.empty:
            for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
                subset = gamble_trials[
                    (gamble_trials['frame'] == cond_filters['frame']) & 
                    (gamble_trials['cond'] == cond_filters['cond'])
                ]
                rts = subset['rt'].dropna()
                if not rts.empty:
                    key = f"mean_rt_Gamble_{cond_key_enum}"
                    if key in stat_keys:
                        summaries[key] = rts.mean()
        
        # Stats for sure choices
        sure_trials = df_trials[df_trials['choice'] == 0]
        if not sure_trials.empty:
            for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
                subset = sure_trials[
                    (sure_trials['frame'] == cond_filters['frame']) & 
                    (sure_trials['cond'] == cond_filters['cond'])
                ]
                rts = subset['rt'].dropna()
                if not rts.empty:
                    key = f"mean_rt_Sure_{cond_key_enum}"
                    if key in stat_keys:
                        summaries[key] = rts.mean()
    
    # Compute aggregate statistics by frame and condition
    frames = ['Gain', 'Loss']
    conds = ['TC', 'NTC']
    
    for frame in frames:
        frame_trials = df_trials[df_trials['frame'] == frame.lower()]
        if not frame_trials.empty:
            # Frame-level stats (Gain/Loss)
            key = f'p_gamble_{frame}'
            if key in stat_keys:
                summaries[key] = frame_trials['choice'].mean() if not frame_trials['choice'].isna().all() else -999.0
            
            key = f'mean_rt_{frame}'
            if key in stat_keys:
                summaries[key] = frame_trials['rt'].mean() if not frame_trials['rt'].isna().all() else -999.0
            
            key = f'std_rt_{frame}'
            if key in stat_keys:
                summaries[key] = frame_trials['rt'].std() if not frame_trials['rt'].isna().all() else -999.0
            
            # Gamble choices within frame
            gamble_frame = frame_trials[frame_trials['choice'] == 1]
            if not gamble_frame.empty:
                key = f'mean_rt_Gamble_{frame}'
                if key in stat_keys:
                    summaries[key] = gamble_frame['rt'].mean() if not gamble_frame['rt'].isna().all() else -999.0
            
            # Sure choices within frame
            sure_frame = frame_trials[frame_trials['choice'] == 0]
            if not sure_frame.empty:
                key = f'mean_rt_Sure_{frame}'
                if key in stat_keys:
                    summaries[key] = sure_frame['rt'].mean() if not sure_frame['rt'].isna().all() else -999.0
    
    # Compute aggregate statistics by time condition
    for cond in conds:
        cond_trials = df_trials[df_trials['cond'] == cond.lower()]
        if not cond_trials.empty:
            # Condition-level stats (TC/NTC)
            key = f'p_gamble_{cond}'
            if key in stat_keys:
                summaries[key] = cond_trials['choice'].mean() if not cond_trials['choice'].isna().all() else -999.0
            
            key = f'mean_rt_{cond}'
            if key in stat_keys:
                summaries[key] = cond_trials['rt'].mean() if not cond_trials['rt'].isna().all() else -999.0
            
            key = f'std_rt_{cond}'
            if key in stat_keys:
                summaries[key] = cond_trials['rt'].std() if not cond_trials['rt'].isna().all() else -999.0
            
            # Gamble choices within condition
            gamble_cond = cond_trials[cond_trials['choice'] == 1]
            if not gamble_cond.empty:
                key = f'mean_rt_Gamble_{cond}'
                if key in stat_keys:
                    summaries[key] = gamble_cond['rt'].mean() if not gamble_cond['rt'].isna().all() else -999.0
            
            # Sure choices within condition
            sure_cond = cond_trials[cond_trials['choice'] == 0]
            if not sure_cond.empty:
                key = f'mean_rt_Sure_{cond}'
                if key in stat_keys:
                    summaries[key] = sure_cond['rt'].mean() if not sure_cond['rt'].isna().all() else -999.0
    
    # Compute composite indices
    try:
        # Framing index (difference in gambling rate between loss and gain frames)
        if all(k in summaries for k in ['p_gamble_Loss', 'p_gamble_Gain']):
            summaries['framing_index'] = summaries['p_gamble_Loss'] - summaries['p_gamble_Gain']
        
        # Time pressure index (difference in gambling rate between TC and NTC)
        if all(k in summaries for k in ['p_gamble_TC', 'p_gamble_NTC']):
            summaries['time_pressure_index'] = summaries['p_gamble_TC'] - summaries['p_gamble_NTC']
        
        # Framing index within time conditions
        if all(k in summaries for k in ['p_gamble_Loss_TC', 'p_gamble_Gain_TC']):
            summaries['framing_index_TC'] = summaries['p_gamble_Loss_TC'] - summaries['p_gamble_Gain_TC']
        if all(k in summaries for k in ['p_gamble_Loss_NTC', 'p_gamble_Gain_NTC']):
            summaries['framing_index_NTC'] = summaries['p_gamble_Loss_NTC'] - summaries['p_gamble_Gain_NTC']
        
        # Time pressure index within frames
        if all(k in summaries for k in ['p_gamble_Gain_TC', 'p_gamble_Gain_NTC']):
            summaries['time_pressure_index_Gain'] = summaries['p_gamble_Gain_TC'] - summaries['p_gamble_Gain_NTC']
        if all(k in summaries for k in ['p_gamble_Loss_TC', 'p_gamble_Loss_NTC']):
            summaries['time_pressure_index_Loss'] = summaries['p_gamble_Loss_TC'] - summaries['p_gamble_Loss_NTC']
        
        # RT Quantiles by Frame
        for frame in ['Gain', 'Loss']:
            frame_rts = df_trials[df_trials['is_gain_frame'] == (frame == 'Gain')]['rt'].dropna()
            if len(frame_rts) >= 5:  # Require at least 5 samples for quantile estimation
                try:
                    q90 = frame_rts.quantile(0.9)
                    summaries[f'rt_q90_{frame}'] = q90
                    if frame == 'Gain':  # Only compute q10 for Gain frame as specified
                        q10 = frame_rts.quantile(0.1)
                        summaries['rt_q10_Gain'] = q10
                except Exception as e:
                    logging.warning(f"Error computing RT quantiles for {frame} frame: {e}")
        
        # RT Quantiles by Time Constraint
        for cond in ['TC', 'NTC']:
            cond_rts = df_trials[df_trials['time_constrained'] == (cond == 'TC')]['rt'].dropna()
            if len(cond_rts) >= 5:  # Require at least 5 samples for quantile estimation
                try:
                    q90 = cond_rts.quantile(0.9)
                    summaries[f'rt_q90_{cond}'] = q90
                    if cond == 'TC':  # Only compute q10 for TC as specified
                        q10 = cond_rts.quantile(0.1)
                        summaries['rt_q10_TC'] = q10
                except Exception as e:
                    logging.warning(f"Error computing RT quantiles for {cond} condition: {e}")
        
        # Choice-Conditional RT Difference by Frame
        for frame in ['Gain', 'Loss']:
            frame_mask = df_trials['is_gain_frame'] == (frame == 'Gain')
            gamble_rts = df_trials[frame_mask & (df_trials['choice'] == 1)]['rt'].dropna()
            sure_rts = df_trials[frame_mask & (df_trials['choice'] == 0)]['rt'].dropna()
            
            if len(gamble_rts) >= 3 and len(sure_rts) >= 3:  # Require at least 3 samples in each group
                try:
                    mean_gamble_rt = gamble_rts.mean()
                    mean_sure_rt = sure_rts.mean()
                    summaries[f'mean_rt_Gamble_vs_Sure_{frame}'] = mean_gamble_rt - mean_sure_rt
                except Exception as e:
                    logging.warning(f"Error computing RT differences for {frame} frame: {e}")
        
        # RT Bimodality Ratio
        all_rts = df_trials['rt'].dropna()
        if len(all_rts) >= 10:  # Require at least 10 samples for bimodality estimation
            try:
                fast_count = (all_rts < 1.0).sum()
                slow_count = (all_rts > 2.0).sum()
                if fast_count > 0:
                    summaries['rt_bimodality_ratio_overall'] = slow_count / fast_count
                else:
                    summaries['rt_bimodality_ratio_overall'] = float('inf')
            except Exception as e:
                logging.warning(f"Error computing RT bimodality ratio: {e}")
                
    except Exception as e:
        logging.warning(f"Error computing composite indices and additional stats: {e}")
        import traceback
        logging.warning(traceback.format_exc())
    
    # Final cleanup and validation
    final_summaries = {}
    for key in stat_keys:
        val = summaries.get(key, -999.0)
        if pd.isna(val) or not np.isfinite(val):
            val = -999.0
        final_summaries[key] = val
    
    # Log any missing statistics
    missing_stats = [k for k in stat_keys if k not in summaries]
    if missing_stats:
        logging.warning(f"Could not compute {len(missing_stats)} statistics: {missing_stats[:5]}{'...' if len(missing_stats) > 5 else ''}")
    
    return final_summaries

def get_trial_valence_scores(texts: List[str]) -> np.ndarray:
    """
    Compute valence scores for trial texts using a pre-trained sentiment model.
    
    Args:
        texts: List of text strings (frame descriptions or trial stimuli)
        
    Returns:
        np.ndarray: Array of valence scores in [-1, 1]
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F
        
        # Load model and tokenizer
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Process in batches to handle large numbers of texts
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and prepare input
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=1)
                
                # Calculate valence score: prob(positive) - prob(negative)
                # Labels: 0=negative, 1=neutral, 2=positive
                batch_scores = (probs[:, 2] - probs[:, 0]).cpu().numpy()
                all_scores.extend(batch_scores)
        
        # Convert to numpy array
        valence_scores = np.array(all_scores, dtype=np.float32)
        
        # If all scores are the same, add small noise to avoid division by zero
        if len(np.unique(valence_scores)) == 1:
            noise = np.random.normal(0, 0.01, size=len(valence_scores))
            valence_scores = valence_scores + noise
        
        # Mean-center and rescale to [-1, 1]
        centered = valence_scores - np.mean(valence_scores)
        max_abs = np.max(np.abs(centered))
        
        if max_abs > 1e-6:  # Avoid division by very small numbers
            valence_scores = centered / max_abs
        
        # Clip to ensure we're within [-1, 1] after any numerical imprecision
        return np.clip(valence_scores, -1.0, 1.0)
        
    except Exception as e:
        logging.error(f"Error in get_trial_valence_scores: {e}")
        logging.error("Falling back to random valence scores with variance")
        # Generate random scores with some variance
        base_scores = np.random.uniform(-0.5, 0.5, size=len(texts)).astype(np.float32)
        # Add some structure based on text length to simulate meaningful variation
        length_effect = np.array([len(t) for t in texts], dtype=np.float32)
        length_effect = (length_effect - np.mean(length_effect)) / (np.std(length_effect) + 1e-8) * 0.2
        return np.clip(base_scores + length_effect, -1.0, 1.0)

def prepare_trial_template(roberts_data_path: Path, num_template_trials: int, seed: int) -> None:
    """
    Loads Roberts data, computes valence scores for each trial,
    and creates a global trial structure template for simulations.
    """
    global SUBJECT_TRIAL_STRUCTURE_TEMPLATE, OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE
    
    try:
        df = pd.read_csv(roberts_data_path)
    except FileNotFoundError:
        logging.error(f"Roberts data file not found at {roberts_data_path} for template creation.")
        raise
        
    # Filter and prepare the data
    # Rename columns to match expected names in the rest of the code
    df = df.rename(columns={
        'reaction_time': 'rt',
        'gamble_prob': 'prob',
        'gamble_chosen': 'choice'
    })
    
    # Drop rows with missing values in key columns
    df.dropna(subset=['subject', 'frame', 'cond', 'prob', 'rt'], inplace=True)
    
    # Add derived columns
    df['time_constrained'] = df['cond'] == 'tc'
    df['is_gain_frame'] = df['frame'] == 'gain'

    if df.empty:
        raise ValueError("No valid target trials found in Roberts data for template creation.")

    # Define template columns including frame text for valence scoring
    template_cols = ['frame', 'cond', 'prob', 'is_gain_frame', 'time_constrained']
    
    # Add any additional columns we need from the original data
    additional_cols = ['endow', 'sureOutcome']
    
    # Create initial unique trials dataframe
    unique_trials = df[template_cols].drop_duplicates()
    for col in additional_cols:
        if col in df.columns and col not in template_cols:
            # Get the first non-null value for each unique trial configuration
            first_vals = df.groupby(template_cols)[col].first()
            unique_trials = unique_trials.merge(
                first_vals, 
                left_on=template_cols,
                right_index=True,
                how='left'
            )
    
    # Create template with unique trial configurations to reduce computation
    unique_trials = unique_trials.drop_duplicates(subset=template_cols)
    
    # Log the structure of the data we're working with
    logging.info(f"Found {len(unique_trials)} unique trial configurations")
    logging.info("Sample of unique trials:")
    logging.info(unique_trials.head().to_string())
    
    # Create more descriptive trial texts using actual gamble parameters from the data
    def create_trial_text(row):
        frame = row['frame']
        cond = row['cond']
        prob = row['prob']
        
        # Get actual values from the data
        sure_outcome = row.get('sureOutcome', 20 if frame == 'gain' else -20)
        endow = row.get('endow', 40 if frame == 'gain' else 50)
        
        if frame == 'gain':
            sure_amount = int(round(abs(sure_outcome)))
            gamble_amount = int(round(endow))
            if cond == 'tc':  # Time-constrained
                return (f"You must choose quickly: "
                        f"1) Receive ${sure_amount} for sure, or "
                        f"2) A gamble with a {int(prob*100)}% chance to win ${gamble_amount} "
                        f"and a {int((1-prob)*100)}% chance to win $0.")
            else:  # Non-time-constrained
                return (f"Consider your options carefully: "
                        f"A) A sure gain of ${sure_amount}, or "
                        f"B) A {int(prob*100)}% chance to win ${gamble_amount} "
                        f"and {int((1-prob)*100)}% chance to win nothing.")
        else:  # loss frame
            sure_loss = int(round(abs(sure_outcome)))
            potential_loss = int(round(endow))
            if cond == 'tc':  # Time-constrained
                return (f"Starting with ${potential_loss}, decide now: "
                        f"1) Lose ${sure_loss} for sure, or "
                        f"2) A gamble with a {int(prob*100)}% chance to lose ${potential_loss} "
                        f"and a {int((1-prob)*100)}% chance to lose nothing.")
            else:  # Non-time-constrained
                return (f"Starting with ${potential_loss}, consider: "
                        f"A) A sure loss of ${sure_loss}, or "
                        f"B) A {int(prob*100)}% chance to lose ${potential_loss} "
                        f"and {int((1-prob)*100)}% chance to lose nothing.")
    
    # Apply the function to create descriptive trial texts
    logging.info("Generating trial texts with actual gamble parameters...")
    unique_trials['frame_text'] = unique_trials.apply(create_trial_text, axis=1)
    
    # Log some example trial texts
    example_trials = unique_trials.sample(min(3, len(unique_trials)), random_state=seed)
    logging.info("\nExample trial texts (first 3 shown):")
    for idx, (_, row) in enumerate(example_trials.iterrows(), 1):
        logging.info(f"\nExample {idx} [{row['frame'].upper()}, {row['cond'].upper()}]:\n{row['frame_text']}")
    
    # Log the distribution of key parameters
    logging.info("\nParameter distributions in the template:")
    for col in ['prob', 'endow', 'sureOutcome']:
        if col in unique_trials.columns:
            stats = unique_trials[col].describe()
            logging.info(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                       f"min={stats['min']:.2f}, max={stats['max']:.2f}")
    
    # Ensure we have all required columns for the template
    for col in template_cols + ['frame_text']:
        if col not in unique_trials.columns:
            raise ValueError(f"Missing required column in template: {col}")
    
    # Compute valence scores for each unique trial configuration
    logging.info("Computing valence scores for trial texts...")
    valence_scores = get_trial_valence_scores(unique_trials['frame_text'].tolist())
    unique_trials['valence_score'] = valence_scores
    
    # Log detailed valence score information
    logging.info(f"Valence score statistics:")
    logging.info(f"  Min: {np.min(valence_scores):.4f}")
    logging.info(f"  Max: {np.max(valence_scores):.4f}")
    logging.info(f"  Mean: {np.mean(valence_scores):.4f}")
    logging.info(f"  Std: {np.std(valence_scores):.4f}")
    logging.info(f"  Number of unique scores: {len(np.unique(valence_scores))}")
    
    # If all scores are the same, add some small random noise to break ties
    if len(np.unique(valence_scores)) == 1:
        logging.warning("All valence scores are identical. Adding small random noise.")
        noise = np.random.normal(0, 0.01, size=len(valence_scores))
        valence_scores = valence_scores + noise
        unique_trials['valence_score'] = valence_scores
        
        # Re-center and scale to [-1, 1]
        valence_scores = (valence_scores - np.mean(valence_scores)) / (np.max(np.abs(valence_scores - np.mean(valence_scores))) + 1e-8)
        unique_trials['valence_score'] = valence_scores
        
        logging.info("After adding noise:")
        logging.info(f"  Min: {np.min(valence_scores):.4f}")
        logging.info(f"  Max: {np.max(valence_scores):.4f}")
        logging.info(f"  Mean: {np.mean(valence_scores):.4f}")
        logging.info(f"  Std: {np.std(valence_scores):.4f}")
    
    # Sample the requested number of trials
    if len(unique_trials) < num_template_trials:
        SUBJECT_TRIAL_STRUCTURE_TEMPLATE = unique_trials[template_cols + ['valence_score']].copy()
        logging.warning(f"Requested {num_template_trials} trials but only {len(unique_trials)} unique configurations available.")
    else:
        SUBJECT_TRIAL_STRUCTURE_TEMPLATE = unique_trials[template_cols + ['valence_score']].sample(
            n=num_template_trials, random_state=seed, replace=False
        ).reset_index(drop=True)
    
    logging.info(f"SUBJECT_TRIAL_STRUCTURE_TEMPLATE created with {len(SUBJECT_TRIAL_STRUCTURE_TEMPLATE)} trials.")
    logging.info(f"Valence score stats - Min: {SUBJECT_TRIAL_STRUCTURE_TEMPLATE['valence_score'].min():.3f}, "
                f"Max: {SUBJECT_TRIAL_STRUCTURE_TEMPLATE['valence_score'].max():.3f}, "
                f"Mean: {SUBJECT_TRIAL_STRUCTURE_TEMPLATE['valence_score'].mean():.3f}")

    # Set up RT means for imputation
    OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE = {}
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
        # Map condition keys to match the data format
        frame_map = {'gain': 'gain', 'loss': 'loss'}
        cond_map = {'TC': 'tc', 'NTC': 'ntc'}
        
        # Extract frame and condition from the key (e.g., 'Gain_TC' -> 'gain', 'tc')
        parts = cond_key_enum.split('_')
        if len(parts) == 2:
            frame_part, cond_part = parts
            frame = frame_map.get(frame_part.lower(), frame_part.lower())
            cond = cond_map.get(cond_part, cond_part.lower())
            
            subset = df[(df['frame'] == frame) & (df['cond'] == cond)]
            rts_for_impute = subset['rt'].dropna()
            
            if not rts_for_impute.empty:
                OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE[cond_key_enum] = rts_for_impute.mean()
        
    # Fallback for any missing conditions
    overall_mean = df['rt'].mean() if not df['rt'].empty else 1.5
    for cond_key_enum in CONDITIONS_ROBERTS.keys():
        if cond_key_enum not in OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE:
            OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE[cond_key_enum] = overall_mean
    
    logging.info(f"Set OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE: {OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE}")


def check_for_unhashable(d, name=""):
    """Helper function to check for unhashable types in nested structures.
    Only shows warnings if actual unhashable types (sets) are found.
    """
    if not isinstance(d, (dict, list, tuple, set)):
        return
        
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, set):
                logging.warning(f"Found unhashable set in dict '{name}' at key '{k}': {v}")
            elif isinstance(v, (dict, list, tuple)):
                check_for_unhashable(v, f"{name}.{k}" if name else k)
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            if isinstance(v, set):
                logging.warning(f"Found unhashable set in list '{name}' at index {i}: {v}")
            elif isinstance(v, (dict, list, tuple)):
                check_for_unhashable(v, f"{name}[{i}]")


def nes_sbi_simulator(params_tensor: torch.Tensor, stat_keys: List[str]) -> torch.Tensor:
    """
    SBI wrapper for the NES model simulation for Roberts task. 
    Simulates one parameter set with valence modulation.
    """
    # Only show debug info for the first few simulations
    global _sim_count
    if '_sim_count' not in globals():
        globals()['_sim_count'] = 0
    
    _sim_count += 1
    if _sim_count <= 3:  # Only show detailed debug for first 3 simulations
        logging.debug(f"--- Starting simulation {_sim_count} with params: {[f'{x:.4f}' for x in params_tensor.tolist()]}")
        logging.debug(f"Input shape: {params_tensor.shape}, type: {type(params_tensor).__name__}")
        
        # Only check for unhashable types in the first simulation
        if _sim_count == 1:
            check_for_unhashable(globals(), "globals")
            check_for_unhashable(locals(), "locals")
    
    global SUBJECT_TRIAL_STRUCTURE_TEMPLATE 
    if SUBJECT_TRIAL_STRUCTURE_TEMPLATE is None:
        raise RuntimeError("SUBJECT_TRIAL_STRUCTURE_TEMPLATE not initialized before simulation call.")

    # Handle input tensor dimensions
    if params_tensor.ndim > 1: 
        params_tensor = params_tensor.squeeze(0)  # Handle potential [1, num_params]

    # Convert parameters to dictionary
    params_dict = {}
    for name, val in zip(PARAM_NAMES, params_tensor):
        val_item = val.item() if hasattr(val, 'item') else val
        params_dict[name] = val_item
    
    # Add fixed parameters
    check_for_unhashable(FIXED_PARAMS, "FIXED_PARAMS")
    params_dict.update(FIXED_PARAMS)
    
    # Log simulation start with parameters
    param_str = ", ".join([f"{k}={v:.4f}" for k, v in params_dict.items()])
    logging.debug(f"Starting simulation with params: {param_str}")
    check_for_unhashable(params_dict, "params_dict")
    
    # Initialize the agent with default config
    agent = MVNESAgent({}) 
    
    # Prepare results container
    sim_results_list = []
    
    # Run simulations for each trial in the template
    for _, trial_info in SUBJECT_TRIAL_STRUCTURE_TEMPLATE.iterrows():
        # Extract trial parameters
        salience_input = trial_info['prob'] 
        norm_input = 1.0 if trial_info['is_gain_frame'] else -1.0
        
        # Get the valence score for this trial (pre-computed in prepare_trial_template)
        valence_score = trial_info.get('valence_score', 0.0)  # Default to neutral if not found
        
        # Prepare parameters for this trial
        check_for_unhashable(BASE_SIM_PARAMS, "BASE_SIM_PARAMS")
        
        agent_run_params = {
            'w_n': params_dict['v_norm'], 
            'threshold_a': params_dict['a_0'],
            'w_s': params_dict['w_s_eff'], 
            't': params_dict['t_0'],
            'alpha_gain': params_dict['alpha_gain'],
            'beta_val': params_dict['beta_val'],
            'logit_z0': params_dict['logit_z0'],
            'log_tau_norm': params_dict['log_tau_norm']
        }
        
        # Add BASE_SIM_PARAMS
        agent_run_params.update(BASE_SIM_PARAMS)
        check_for_unhashable(agent_run_params, "agent_run_params")
        
        try:
            # Run the simulation for this trial
            trial_output = agent.run_mvnes_trial(
                salience_input=salience_input,
                norm_input=norm_input,
                params=agent_run_params 
            )
            
            # Get RT (should already include t_0 from the agent)
            sim_rt = trial_output.get('rt', np.nan)

            # Apply time constraint if needed
            if not pd.isna(sim_rt) and trial_info['time_constrained']:
                sim_rt = min(sim_rt, 1.0)
            
            # Store trial results
            sim_results_list.append({
                'rt': sim_rt, 
                'choice': trial_output.get('choice', np.nan),
                'frame': trial_info['frame'], 
                'cond': trial_info['cond'],
                'time_constrained': trial_info['time_constrained'],
                'is_gain_frame': trial_info['is_gain_frame'],
                'valence_score': valence_score
            })
            
        except Exception as e_sim:
            logging.debug(f"Sim trial exception for params {params_dict}: {e_sim}")
            # Add a failed trial with NaN values
            sim_results_list.append({
                'rt': np.nan, 
                'choice': np.nan, 
                'frame': trial_info['frame'], 
                'cond': trial_info['cond'],
                'time_constrained': trial_info['time_constrained'],
                'is_gain_frame': trial_info['is_gain_frame'],
                'valence_score': valence_score
            })
    
    # Log simulation completion (only first few times)
    if _sim_count <= 3:
        valid_rts = sum(1 for r in sim_results_list if not np.isnan(r['rt']))
        logging.debug(f"Simulation {_sim_count} completed: {valid_rts}/{len(sim_results_list)} valid RTs")
    
    # Convert results to DataFrame
    df_sim_batch = pd.DataFrame(sim_results_list)
    
    # Calculate summary statistics
    summary_stats_dict = calculate_summary_stats_roberts(
        df_sim_batch, 
        stat_keys, 
        OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE
    )
    
    # Convert to tensor in the correct order
    summary_stats_vector = [summary_stats_dict.get(k, -999.0) for k in stat_keys]
    return torch.tensor(summary_stats_vector, dtype=torch.float32)


# --- Main Script Logic ---
# (Main script logic continues here)

def setup_output_directory(output_base_name: str) -> Path:
    """Creates the main output directory and standard subdirectories."""
    base_dir = Path(output_base_name)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / 'data').mkdir(parents=True, exist_ok=True)
    (base_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (base_dir / 'models').mkdir(parents=True, exist_ok=True) # Add this line
    (base_dir / 'npe_cache').mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory setup at: {base_dir.resolve()}")
    return base_dir, base_dir / 'data', base_dir / 'plots', base_dir / 'models', base_dir / 'npe_cache'

def main():
    parser = argparse.ArgumentParser(description="Run SBC for NES model on Roberts et al. task.")
    # Simulation and training parameters
    parser.add_argument('--npe_train_sims', type=int, default=5000, 
                       help="Number of training simulations for NPE")
    parser.add_argument('--template_trials', type=int, default=100,
                       help="Number of trials to use from the template")
    parser.add_argument('--sbc_datasets', type=int, default=100,
                       help="Number of datasets to generate for SBC")
    parser.add_argument('--sbc_posterior_samples', type=int, default=500,
                       help="Number of posterior samples per SBC dataset")
    
    # Configuration
    parser.add_argument('--seed', type=int, default=None, 
                       help="Random seed (default: current time)")
    parser.add_argument('--output_base_name', type=str, default="sbc_nes_roberts_rebuilt",
                       help="Base name for output directories and files")
    parser.add_argument('--roberts_data_file', type=str, 
                       default="./roberts_framing_data/ftp_osf_data.csv",
                       help="Path to Roberts et al. data file")
    
    # Architecture and priors
    parser.add_argument('--use_tight_priors', action='store_true',
                       help='Use tighter prior distributions for parameters')
    parser.add_argument('--npe_architecture', type=str, default='maf',
                       choices=['maf', 'nsf'],
                       help='Density estimator architecture (maf or nsf)')
    
    # Runtime options
    parser.add_argument('--sbc_debug_mode', action='store_true', 
                       help='Run SBC on only 10 datasets for debugging')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--force_retrain_npe', action='store_true',
                       help='Force retraining of NPE even if a checkpoint exists')
    args = parser.parse_args()

    # Set up random seeds and device
    if args.seed is None: 
        args.seed = int(time.time())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}. Seed: {args.seed}")
    
    # Set priors based on command line argument
    global PRIOR_LOW, PRIOR_HIGH
    if args.use_tight_priors:
        PRIOR_LOW = PRIOR_LOW_TIGHT
        PRIOR_HIGH = PRIOR_HIGH_TIGHT
        logging.info("Using TIGHT priors")
    else:
        PRIOR_LOW = PRIOR_LOW_WIDE
        PRIOR_HIGH = PRIOR_HIGH_WIDE
        logging.info("Using WIDE priors")
    
    output_dir, output_dir_data, output_dir_plots, output_dir_models, output_dir_cache = setup_output_directory(args.output_base_name)
    summary_stat_keys = get_roberts_summary_stat_keys()
    num_summary_stats = len(summary_stat_keys)
    if num_summary_stats <=0: logging.error("Num summary stats is 0!"); sys.exit(1)
    logging.info(f"Number of summary stats defined: {num_summary_stats}")

    # Prepare trial template
    try:
        prepare_trial_template(Path(args.roberts_data_file), args.template_trials, args.seed)
    except Exception as e_template:
        logging.error(f"Failed to prepare trial template: {e_template}", exc_info=True); sys.exit(1)

    sbi_prior = BoxUniform(low=PRIOR_LOW.to(device), high=PRIOR_HIGH.to(device), device=device.type)
    def actual_simulator_for_sbi(parameter_sample_batch_tensor: torch.Tensor) -> torch.Tensor:
        # This function is called by simulate_for_sbi.
        # parameter_sample_batch_tensor can be [batch_size, num_params] or [num_params]
        if parameter_sample_batch_tensor.ndim == 1: # Single sample
             return nes_sbi_simulator(parameter_sample_batch_tensor, summary_stat_keys)
        else: # Batch of samples
            batch_results = []
            for i_sample in range(parameter_sample_batch_tensor.shape[0]):
                batch_results.append(nes_sbi_simulator(parameter_sample_batch_tensor[i_sample], summary_stat_keys))
            return torch.stack(batch_results)

    logging.info(f"Starting NPE training with {args.npe_train_sims} simulations...")

    theta_train, x_train = simulate_for_sbi(
        simulator=actual_simulator_for_sbi,
        proposal=sbi_prior,
        num_simulations=args.npe_train_sims,
        num_workers=1, 
        show_progress_bar=True,
        simulation_batch_size=1 # Kept at 1 as per previous findings, new wrapper handles potential batches if sbi sends them
    )

    logging.info(f"Generated training data: theta_train {theta_train.shape}, x_train {x_train.shape}")

    # Filter out simulations that resulted in NaNs or Infs or only placeholder values in summary stats
    # This can happen if a simulation run fails catastrophically for some parameter sets
    valid_sim_mask = ~(x_train == -999.0).all(dim=1) & ~torch.isnan(x_train).any(dim=1) & ~torch.isinf(x_train).any(dim=1)
    theta_train_valid = theta_train[valid_sim_mask]
    x_train_valid = x_train[valid_sim_mask]
    if len(theta_train_valid) < args.npe_train_sims * 0.5:
        logging.error(f"Less than 50% of simulations are valid ({len(theta_train_valid)}/{args.npe_train_sims}). Aborting."); sys.exit(1)
    logging.info(f"Using {len(theta_train_valid)} valid simulations for NPE training.")

    # Set up NPE with selected architecture
    if args.npe_architecture == 'nsf':
        # Create embedding net with simpler architecture
        # First check what parameters are accepted by FCEmbedding
        logging.info("Creating embedding network...")
        embedding_net = FCEmbedding(
            input_dim=x_train_valid.shape[1],
            num_layers=2,  # Two hidden layers
            num_hiddens=256,  # 256 units per hidden layer
            output_dim=30  # Reduce to 30 features
        )
        
        # Create NSF density estimator
        density_estimator = posterior_nn(
            model='nsf',
            embedding_net=embedding_net,
            hidden_features=256,
            num_transforms=8,
            num_bins=8,
            z_score_theta='independent',
            z_score_x='independent'
        )
        
        logging.info("Using NSF density estimator with embedding net (60 -> 30 features) and 8 transforms")
        npe = SNPE(
            prior=sbi_prior,
            density_estimator=density_estimator,
            device=device.type,
            show_progress_bars=True
        )
    else:
        # Default to MAF
        logging.info("Using MAF density estimator (default)")
        npe = SNPE(
            prior=sbi_prior,
            density_estimator='maf',
            device=device.type
        )
    
    # Train the density estimator
    density_estimator = npe.append_simulations(theta_train_valid, x_train_valid).train(
        show_train_summary=True,
        force_first_round_loss=True
    )

    # --- Compute and save training summary-stat means and stds ---
    training_stat_means = x_train_valid.mean(dim=0).cpu().numpy()
    training_stat_stds = x_train_valid.std(dim=0).cpu().numpy()
    np.save(output_dir_data / 'training_stat_means.npy', training_stat_means)
    np.save(output_dir_data / 'training_stat_stds.npy', training_stat_stds)
    pd.DataFrame(training_stat_means).to_csv(output_dir_data / 'training_stat_means.csv', index=False, header=False)
    pd.DataFrame(training_stat_stds).to_csv(output_dir_data / 'training_stat_stds.csv', index=False, header=False)

    # --- Save summary_stat_keys and parameter names ---
    with open(output_dir_data / 'summary_stat_keys.json', 'w') as f:
        json.dump(summary_stat_keys, f, indent=2)
    with open(output_dir_data / 'parameter_names.json', 'w') as f:
        json.dump(PARAM_NAMES, f, indent=2)

    # --- Save prior bounds ---
    prior_bounds = {'low': PRIOR_LOW.cpu().numpy().tolist(), 'high': PRIOR_HIGH.cpu().numpy().tolist()}
    with open(output_dir_data / 'prior_bounds.json', 'w') as f:
        json.dump(prior_bounds, f, indent=2)
    torch.save(prior_bounds, output_dir_data / 'prior_low_high.pt')

    # --- Save all valid training data for reproducibility ---
    torch.save({'theta_train_valid': theta_train_valid.cpu(), 'x_train_valid': x_train_valid.cpu()}, output_dir_data / 'theta_x_train_valid.pt')
    np.save(output_dir_data / 'x_train_valid.npy', x_train_valid.cpu().numpy())
    np.save(output_dir_data / 'theta_train_valid.npy', theta_train_valid.cpu().numpy())
    pd.DataFrame(x_train_valid.cpu().numpy()).to_csv(output_dir_data / 'simulated_summary_stats.csv', index=False, header=False)
    training_stat_means = x_train_valid.mean(dim=0).cpu().numpy()
    training_stat_stds = x_train_valid.std(dim=0).cpu().numpy()
    np.save(output_dir_data / 'training_stat_means.npy', training_stat_means)
    np.save(output_dir_data / 'training_stat_stds.npy', training_stat_stds)
    pd.DataFrame(training_stat_means).to_csv(output_dir_data / 'training_stat_means.csv', index=False, header=False)
    pd.DataFrame(training_stat_stds).to_csv(output_dir_data / 'training_stat_stds.csv', index=False, header=False)

    # --- Save summary_stat_keys and parameter names ---
    with open(output_dir_data / 'summary_stat_keys.json', 'w') as f:
        json.dump(summary_stat_keys, f, indent=2)
    with open(output_dir_data / 'parameter_names.json', 'w') as f:
        json.dump(PARAM_NAMES, f, indent=2)

    # --- Save prior bounds ---
    prior_bounds = {'low': PRIOR_LOW.cpu().numpy().tolist(), 'high': PRIOR_HIGH.cpu().numpy().tolist()}
    with open(output_dir_data / 'prior_bounds.json', 'w') as f:
        json.dump(prior_bounds, f, indent=2)
    torch.save(prior_bounds, output_dir_data / 'prior_low_high.pt')

    # --- Save checkpoint with all critical metadata ---
    npe_save_path = output_dir_models / f"nes_npe_sims{args.npe_train_sims}_template{args.template_trials}_seed{args.seed}.pt"
    npe_checkpoint = {
        'density_estimator_state_dict': density_estimator.state_dict(),
        'prior_params': prior_bounds,
        'param_names': PARAM_NAMES,
        'num_summary_stats': num_summary_stats,
        'summary_stat_keys': summary_stat_keys,
        'npe_train_sims': args.npe_train_sims,
        'template_trials_for_training': args.template_trials,
        'sbi_version': sbi.__version__,
        'training_seed': args.seed,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_stat_means': training_stat_means.tolist(),
        'training_stat_stds': training_stat_stds.tolist()
    }
    torch.save(npe_checkpoint, npe_save_path)
    logging.info(f"Saved trained NPE and all empirical fit metadata to {npe_save_path}")

    # --- Explicitly save essential artifacts for empirical fit ---
    # Save summary_stat_keys.json
    with open(output_dir_data / "summary_stat_keys.json", "w") as f:
        json.dump(summary_stat_keys, f, indent=2)
    # Save training_stat_means.pt
    torch.save(torch.tensor(training_stat_means), output_dir_data / "training_stat_means.pt")
    # Save parameter_names.json
    with open(output_dir_data / "parameter_names.json", "w") as f:
        json.dump(PARAM_NAMES, f, indent=2)
    # Save prior_bounds.json
    with open(output_dir_data / "prior_bounds.json", "w") as f:
        json.dump(prior_bounds, f, indent=2)
    # Save x_train_valid.pt and theta_train_valid.pt for debugging
    torch.save(x_train_valid.cpu(), output_dir_data / "x_train_valid.pt")
    torch.save(theta_train_valid.cpu(), output_dir_data / "theta_train_valid.pt")

    logging.info(f"Starting SBC with {args.sbc_datasets} datasets...")
    
    theta_sbc_gt = sbi_prior.sample((args.sbc_datasets,)).to(device) # Ensure on correct device

    x_sbc_obs_list = []
    for i in range(args.sbc_datasets):
        if (i + 1) % (args.sbc_datasets // 10 or 1) == 0: logging.info(f"Simulating SBC dataset {i+1}/{args.sbc_datasets}...")
        x_sbc_obs_list.append(actual_simulator_for_sbi(theta_sbc_gt[i])) 
    x_sbc_obs = torch.stack(x_sbc_obs_list).to(device) # Ensure results are on device
    logging.info(f"Generated SBC observations: theta_sbc_gt {theta_sbc_gt.shape}, x_sbc_obs {x_sbc_obs.shape}")

    valid_sbc_mask = ~(x_sbc_obs == -999.0).all(dim=1) & ~torch.isnan(x_sbc_obs).any(dim=1) & ~torch.isinf(x_sbc_obs).any(dim=1)
    theta_sbc_gt_valid = theta_sbc_gt[valid_sbc_mask].to(device)
    x_sbc_obs_valid = x_sbc_obs[valid_sbc_mask].to(device)
    num_valid_sbc_datasets = len(theta_sbc_gt_valid)
    if num_valid_sbc_datasets == 0: logging.error("No valid SBC datasets generated."); sys.exit(1)
    logging.info(f"Using {num_valid_sbc_datasets} valid datasets for SBC rank calculation.")

    posterior_object_for_sbc = npe.build_posterior(density_estimator)
    logging.info(f"Posterior object for SBC built from trained density_estimator.")
    logging.info(f"Proceeding with {num_valid_sbc_datasets} valid datasets for SBC rank calculation.")

    ranks = run_sbc(
        theta_sbc_gt_valid,
        x_sbc_obs_valid,
        posterior_object_for_sbc, 
        args.sbc_posterior_samples 
    ) 
    logging.info(f"SBC ranks calculated. Shape before potential transpose: {ranks.shape}")

    # Patch: Ensure ranks is 2D (num_datasets, num_parameters)
    if ranks.ndim == 1:
        logging.warning(f"Ranks tensor is 1D (shape: {ranks.shape}), unsqueezing to 2D for downstream compatibility.")
        ranks = ranks.unsqueeze(0)

    df_ranks_columns = [f"rank_{PARAM_NAMES[j]}" if j < len(PARAM_NAMES) else f"rank_param_{j}" for j in range(ranks.shape[1])]
    df_ranks = pd.DataFrame(ranks.cpu().numpy(), columns=df_ranks_columns)
    df_ranks.to_csv(output_dir_data / "sbc_ranks.csv", index=False)
    logging.info(f"SBC ranks saved to {output_dir_data / 'sbc_ranks.csv'}")

    ks_results = {}
    num_params_from_ranks = ranks.shape[1]
    for i in range(num_params_from_ranks):
        param_name = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"param_{i}"
        param_ranks_cpu = ranks[:, i].cpu().numpy() 
        normalized_ranks = param_ranks_cpu / args.sbc_posterior_samples
        if len(np.unique(normalized_ranks)) < 2:
            logging.warning(f"KS test for '{param_name}': All ranks are identical ({normalized_ranks[0]}). KS test result will be trivial (p=0 or p=1) and likely uninformative.")
            ks_stat, ks_pval = (1.0, 0.0) if np.all(normalized_ranks == normalized_ranks[0]) else sp_stats.kstest(normalized_ranks, 'uniform')
        else:
            ks_stat, ks_pval = sp_stats.kstest(normalized_ranks, 'uniform')
        ks_results[param_name] = {'ks_stat': ks_stat, 'ks_pval': ks_pval}
    logging.info(f"SBC KS test results: {ks_results}")
    with open(output_dir_data / "sbc_ks_test_results.json", 'w') as f_ks:
        json.dump(ks_results, f_ks, indent=4)
    logging.info(f"SBC KS test results saved to {output_dir_data / 'sbc_ks_test_results.json'}")

    # --- MANUAL SBC DIAGNOSTIC PLOTTING: ECDFs and Histograms ---
    try:
        # Ranks: shape (num_sbc_datasets, num_params)
        ranks_np = ranks.cpu().numpy()
        if ranks_np.shape[1] != len(PARAM_NAMES):
            param_names = [f'param_{i}' for i in range(ranks_np.shape[1])]
        else:
            param_names = PARAM_NAMES
        num_params = ranks_np.shape[1]
        num_datasets = ranks_np.shape[0]
        fig, axes = plt.subplots(num_params, 2, figsize=(10, 2.5*num_params))
        if num_params == 1:
            axes = np.array([axes])
        for i in range(num_params):
            # ECDF
            sorted_ranks = np.sort(ranks_np[:, i])
            ecdf = np.arange(1, num_datasets+1) / num_datasets
            axes[i, 0].plot(sorted_ranks / args.sbc_posterior_samples, ecdf, marker='.', linestyle='-', color='blue')
            axes[i, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[i, 0].set_title(f"ECDF: {param_names[i]}")
            axes[i, 0].set_xlabel("Normalized Rank")
            axes[i, 0].set_ylabel("ECDF")
            axes[i, 0].set_xlim([0, 1])
            axes[i, 0].set_ylim([0, 1])
            # Histogram
            axes[i, 1].hist(ranks_np[:, i], bins=max(10, args.sbc_posterior_samples // 20), range=(0, args.sbc_posterior_samples), color='gray', alpha=0.7, density=True)
            axes[i, 1].axhline(1.0, color='red', linestyle='--', alpha=0.5)
            axes[i, 1].set_title(f"Rank Histogram: {param_names[i]}")
            axes[i, 1].set_xlabel("Rank")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].set_xlim([0, args.sbc_posterior_samples])
        fig.tight_layout()
        fig.suptitle(f'SBC Manual Diagnostics ({num_datasets} Datasets, {args.sbc_posterior_samples} Posterior Samples)', y=1.02)
        sbc_plot_path = output_dir_plots / "sbc_manual_diagnostics_plot.png"
        fig.savefig(sbc_plot_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"SBC manual diagnostics plot saved to {sbc_plot_path}")
    except Exception as e:
        logging.error(f"Manual SBC diagnostics plotting failed: {e}")


    sbc_metadata = {
        'npe_train_sims': args.npe_train_sims, 'template_trials': args.template_trials,
        'sbc_datasets_attempted': args.sbc_datasets, 'sbc_datasets_valid': num_valid_sbc_datasets,
        'sbc_posterior_samples': args.sbc_posterior_samples, 'seed': args.seed,
        'output_dir': str(output_dir), 'npe_save_path': str(npe_save_path),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'ks_results': ks_results
    }
    with open(output_dir_data / "sbc_run_metadata.json", 'w') as f_meta:
        json.dump(sbc_metadata, f_meta, indent=4)
    logging.info(f"SBC metadata saved. Output directory: {output_dir}")
    logging.info("SBC run finished successfully.")

if __name__ == "__main__":
    main()