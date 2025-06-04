#!/usr/bin/env python3

"""Run Posterior Predictive Checks (PPCs) for the 6-parameter NES model.

This script is designed for --empirical_fits_only mode, using mean fitted parameters
to simulate data and compare with observed summary statistics.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # For specific matplotlib/torch issues
import sys
from datetime import datetime
import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# from sbi.inference.posteriors import DirectPosterior as Posterior # Not used in empirical_fits_only
from tqdm import tqdm

# Define the 6 parameters used in the NES model (match fitted_params_file)
PARAM_NAMES_FITTED = ['v_norm_mean', 'a_0_mean', 'w_s_eff_mean', 't_0_mean', 'alpha_gain_mean', 'beta_val_mean']
# Corresponding agent parameter names if different (agent uses these for its attributes/config)
PARAM_MAP_AGENT = {
    'v_norm_mean': 'w_n',       # Norm weight in agent
    'a_0_mean': 'threshold_a',  # Base threshold in agent
    'w_s_eff_mean': 'w_s',      # Salience weight in agent
    't_0_mean': 't',            # Non-decision time in agent
    'alpha_gain_mean': 'alpha_gain',
    'beta_val_mean': 'beta_val'
}


# --- Local project imports ---
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"  # SRC_DIR is 'C:\Users\jesse\Hegemonikon Project\NES_Copilot\src'

if not SRC_DIR.is_dir():
    # This part is less likely to be hit now we know the structure
    logging.error(f"CRITICAL: src/ directory not found at {SRC_DIR}. Cannot proceed.")
    sys.exit(1)  # Exit if src isn't there
else:
    # src directory exists, add it to path to make 'nes_copilot' package findable
    if str(SRC_DIR) not in sys.path:
        # Debug prints for path resolution - commented out for cleaner output
        # print(f"DEBUG SCRIPT: Added {SRC_DIR} to sys.path.")
        # print(f"DEBUG SCRIPT: Attempting to import from package 'nes_copilot' within {SRC_DIR}")
        sys.path.insert(0, str(SRC_DIR))

try:
    # Import using the package structure nes_copilot.module_name
    from nes_copilot.agent_mvnes import MVNESAgent
    from nes_copilot.roberts_stats import calculate_summary_stats_roberts as calculate_summary_stats, ROBERTS_SUMMARY_STAT_KEYS, validate_summary_stats
    from nes_copilot.config_manager_fixed import FixedConfigManager as ConfigManager
    # You might also need this if it's used and in that folder:
    # from nes_copilot.config_manager_fixed import FixedConfigManager as ConfigManager

    # Debug prints for imports - commented out for cleaner output
    # print("DEBUG SCRIPT: Successfully imported modules from 'nes_copilot' package within src/.")
    # print(f"DEBUG SCRIPT: Current working directory: {os.getcwd()}")
    # print(f"DEBUG SCRIPT: Python path: {sys.path}")
    # print(f"DEBUG SCRIPT: Attempting to import from 'nes_copilot'...")
    pass

except ImportError as e:
    logging.error(f"Failed to import from nes_copilot package: {e}")
    logging.error(f"Current sys.path: {sys.path}")
    logging.error(f"Contents of {SRC_DIR}: {[f.name for f in SRC_DIR.iterdir() if f.is_file() or f.is_dir()]}")
    raise

# Verify imports (optional but good)
if 'MVNESAgent' not in locals():
    raise ImportError("MVNESAgent failed to import.")
if 'ROBERTS_SUMMARY_STAT_KEYS' not in locals():
    raise ImportError("ROBERTS_SUMMARY_STAT_KEYS failed to import.")
if 'calculate_summary_stats' not in locals():
    raise ImportError("calculate_summary_stats failed to import.")
# --- End Local project imports ---

# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_ppc",
        description="Posterior Predictive Checks for the NES model using empirical fits.",
    )
    parser.add_argument("--fitted_params_file", type=Path, required=True, help="CSV with empirical posterior summaries (mean values).")
    parser.add_argument("--roberts_data_file", type=Path, required=True, help="Raw Roberts data CSV (trial-level).")
    parser.add_argument("--output_dir", type=Path, default=Path("ppc_outputs"), help="Output directory for PPC results.")
    parser.add_argument("--subject_ids", type=str, default=None, help="Comma-separated subject IDs to analyse (default: all).")
    parser.add_argument("--n_sim_reps", type=int, default=100, help="Number of simulation replicates FOR EACH pseudo-posterior parameter draw.")
    parser.add_argument("--n_pseudo_posterior_samples", type=int, default=50, help="# of pseudo-posterior parameter sets to draw per subject. Set to 0 or 1 to use only mean parameters (with n_sim_reps).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (torch & numpy).")
    parser.add_argument("--threshold_scale", type=float, default=1.0, help="Scaling factor for the threshold parameter (a_0).")
    parser.add_argument("--ppc_version", type=str, default="empirical_ppc", help="Version identifier for output directory.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage (relevant if torch operations were on GPU).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    # --empirical_fits_only is assumed for this version of the script.
    # --npe_file, num_posterior_draws, num_simulations_per_draw are omitted as they relate to NPE-based PPCs.
    return parser.parse_args(argv)

def setup_logging(debug: bool = False) -> None:
    """Configure root logger for console output."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(asctime)s %(name)s - %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) # Quieten matplotlib
    logging.info(f"Logging level set to {logging.getLevelName(log_level)}")


def get_trial_structure(df_subj_cleaned: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Converts a preprocessed and cleaned subject DataFrame to a list of trial dictionaries.
    Assumes df_subj_cleaned already contains all necessary columns with correct types.
    """
    required_cols = [
        'prob', 'is_gain_frame', 'frame', 'cond', 
        'time_constrained', 'valence_score', 'norm_category_for_trial'
    ]
    missing = [col for col in required_cols if col not in df_subj_cleaned.columns]
    if missing:
        raise ValueError(
            f"Cleaned DataFrame for get_trial_structure is missing columns: {missing}. "
            f"Available columns: {df_subj_cleaned.columns.tolist()}"
        )
    
    # Ensure correct types before converting to dicts
    df_copy = df_subj_cleaned[required_cols].copy()
    df_copy['prob'] = pd.to_numeric(df_copy['prob'], errors='coerce')
    df_copy['valence_score'] = pd.to_numeric(df_copy['valence_score'], errors='coerce')
    df_copy['is_gain_frame'] = df_copy['is_gain_frame'].astype(bool)
    df_copy['time_constrained'] = df_copy['time_constrained'].astype(bool)
    
    # Fill any NaNs that might have been introduced by coerce, though data should be clean by now
    if df_copy['prob'].isnull().any():
        logging.warning("NaNs found in 'prob' column within get_trial_structure; filling with mean.")
        df_copy['prob'].fillna(df_copy['prob'].mean(), inplace=True)
    if df_copy['valence_score'].isnull().any():
        logging.warning("NaNs found in 'valence_score' column within get_trial_structure; filling with 0.")
        df_copy['valence_score'].fillna(0.0, inplace=True)

    return df_copy.to_dict('records')


def simulate_one_full_dataset(
    agent: MVNESAgent,                  # Agent instance with its parameters already set
    trial_struct: Sequence[Dict],
    sim_control_params: Dict[str, Any]  # For dt, noise_std_dev, max_time (trial-specific)
) -> Dict[str, Any] | None:
    """
    Run MVNESAgent across a subject's trial structure and compute summary stats.
    The agent uses its own internal parameters (self.w_s, self.threshold_a, etc.).
    The 'sim_control_params' are passed to the agent's run_mvnes_trial 'params' argument.
    """
    simulated_rows = []
    if not trial_struct:
        # Detailed debug logging - uncomment only when needed
        # logging.debug("Simulate_one_full_dataset received parameters:")
        # for param_name_key, value in params.items():
        #     logging.debug(f"  {param_name_key}: {value}")
        logging.info(f"Simulating {len(trial_struct)} trials.")
        # logging.debug(f"Agent DDM params (from agent.config): w_s={agent.config.get('w_s'):.3f}, w_n={agent.config.get('w_n'):.3f}, threshold_a={agent.config.get('threshold_a'):.3f}, t={agent.config.get('t'):.3f}, alpha_gain={agent.config.get('alpha_gain'):.3f}, beta_val={agent.config.get('beta_val'):.3f}")

    for i, trial_info in enumerate(trial_struct):
        try:
            prob = float(trial_info['prob'])
            is_gain_frame = bool(trial_info['is_gain_frame'])
            frame_str_val = str(trial_info['frame'])
            cond_str_val = str(trial_info['cond'])
            time_constrained = bool(trial_info['time_constrained'])
            valence_score = float(trial_info['valence_score'])
            norm_category_str = str(trial_info['norm_category_for_trial'])

            # Calculate norm_category_code early
            category_mapping = {'default': 0, 'fairness': 1, 'altruism': 2, 
                             'reciprocity': 3, 'trust': 4, 'cooperation': 5}
            norm_category_code = category_mapping.get(norm_category_str.lower(), 0)
            
            salience_input = prob
            norm_input_val = 1.0 if is_gain_frame else -1.0

            # Extensive trial logging - uncomment only when debugging specific trials
            # if i < 1:  # Log first trial details extensively
            #     logging.debug(f"  Trial {i}: prob={prob:.2f}, ...")
            #     logging.debug(f"  Agent call params dict for trial {i}: {current_trial_run_params}"), "
                            # f"time_constrained={time_constrained}, "
                            # f"valence_score={valence_score:.2f}, "
                            # f"norm_category_code={norm_category_code}")

            # Call the agent's run_mvnes_trial method with the correct signature
            choice, rt = agent.run_mvnes_trial(
                is_gain_frame=is_gain_frame,
                time_constrained=time_constrained,
                valence_score_trial=valence_score,
                norm_category_for_trial=norm_category_code
            )

            simulated_rows.append({
                'trial': i + 1,
                'prob': prob,
                'frame': frame_str_val,
                'cond': cond_str_val,
                'rt': rt,
                'choice': choice
            })
        except Exception as e:
            logging.error(f"Error in simulate_one_full_dataset trial {i}: {e}", exc_info=True)
            logging.error(f"Problematic trial_info: {trial_info}")
            # To avoid cascading errors, re-raise if critical, or return None/partial
            # For PPC, it's often better to get partial data than none.
            # However, if it's a systematic error, it should be fixed.
            # For now, let's allow it to skip problematic trials within a simulation.
            continue 
            
    if not simulated_rows:
        logging.warning("No rows were successfully simulated.")
        return None
        
    df_sim = pd.DataFrame(simulated_rows)
    try:
        stats_dict = calculate_summary_stats(df_sim, ROBERTS_SUMMARY_STAT_KEYS)
        return stats_dict
    except Exception as e:
        logging.error(f"Error calculating summary stats for simulated data: {e}", exc_info=True)
        return None


def simulate_n_replicates(
    agent: MVNESAgent,
    trial_struct: Sequence[Dict],
    base_sim_control_params: Dict[str, Any], # e.g. dt, noise_std_dev
    n_reps: int
) -> List[Dict[str, Any]]:
    """Run multiple simulations with the same agent (and its set parameters)."""
    all_replicate_stats = []
    for _ in tqdm(range(n_reps), desc="Simulating replicates", leave=False):
        stats = simulate_one_full_dataset(agent, trial_struct, base_sim_control_params)
        if stats:
            all_replicate_stats.append(stats)
    return all_replicate_stats


def plot_ppc_distributions(
    ppc_results: Dict[str, Any], 
    output_dir: Path,
    subject_ids_to_plot: Optional[List[str]] = None,
    stats_to_plot: Optional[List[str]] = None
) -> None:
    """Generate PPC plots for each subject and summary statistic."""
    if subject_ids_to_plot is None:
        subject_ids_to_plot = list(ppc_results.keys())
    
    # Use a subset of the most important statistics for plotting
    default_stats = [
        'p_gamble_All', 'mean_rt_All', 'std_rt_All',
        'p_gamble_Gain', 'p_gamble_Loss',
        'mean_rt_Gain', 'mean_rt_Loss',
        'framing_effect_ntc', 'framing_effect_tc'
    ]
    # Only use default stats that exist in the observed data
    if ppc_results and subject_ids_to_plot and subject_ids_to_plot[0] in ppc_results:
        observed_keys = set(ppc_results[subject_ids_to_plot[0]].get('observed_summary_stats', {}).keys())
        default_stats = [s for s in default_stats if s in observed_keys]
    
    stats_to_plot = stats_to_plot if stats_to_plot is not None else default_stats
    
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for subj_id in tqdm(subject_ids_to_plot, desc="Generating PPC plots"):
        if subj_id not in ppc_results:
            logging.warning(f"No PPC results found for subject {subj_id} to plot.")
            continue
            
        subj_data = ppc_results[subj_id]
        obs_stats = subj_data.get('observed_summary_stats')
        sim_stats_list = subj_data.get('simulated_summary_stats_list')
        
        if not obs_stats or not sim_stats_list:
            logging.warning(f"Missing observed or simulated stats for subject {subj_id}.")
            continue
            
        for stat_key in stats_to_plot:
            if stat_key not in obs_stats:
                logging.debug(f"Statistic '{stat_key}' not in observed_stats for subject {subj_id}.")
                continue
            
            obs_val = obs_stats[stat_key]
            sim_vals_for_stat = [s.get(stat_key) for s in sim_stats_list if s and stat_key in s]
            sim_vals_for_stat = [s for s in sim_vals_for_stat if pd.notna(s)]

            if not sim_vals_for_stat:
                logging.debug(f"No valid simulated values for stat '{stat_key}' for subject {subj_id}.")
                continue
            
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(sim_vals_for_stat, kde=True, stat='density', color='skyblue', edgecolor='white', bins=min(30, max(5, len(sim_vals_for_stat)//10)))
                plt.axvline(obs_val, color='red', linestyle='--', linewidth=2, label=f'Observed: {obs_val:.3f}')
                
                lower_ci = np.percentile(sim_vals_for_stat, 2.5)
                upper_ci = np.percentile(sim_vals_for_stat, 97.5)
                plt.axvspan(lower_ci, upper_ci, alpha=0.2, color='gray', label=f'95% CI [{lower_ci:.3f}, {upper_ci:.3f}]')
                
                plt.title(f'Subject {subj_id} - {stat_key}')
                plt.xlabel('Statistic Value')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_dir / f"subj_{subj_id}_{stat_key}.png")
                plt.close()
            except Exception as e:
                logging.warning(f"Error plotting {stat_key} for subject {subj_id}: {e}")
                plt.close('all')


def calculate_detailed_coverage(ppc_results: Dict[str, Any], min_sim_reps: int = 5) -> pd.DataFrame:
    """
    Calculate detailed coverage statistics for each subject and statistic.
    
    Args:
        ppc_results: Dictionary containing PPC results for each subject
        min_sim_reps: Minimum number of simulation replicates required to calculate statistics
        
    Returns:
        DataFrame with coverage statistics for each subject and statistic
    """
    rows = []
    for subj_id, data in ppc_results.items():
        obs_stats = data.get('observed_summary_stats', {})
        sim_stats_list = data.get('simulated_summary_stats_list', [])
        
        if not obs_stats or not sim_stats_list:
            logging.debug(f"Skipping subject {subj_id}: missing observed or simulated stats")
            continue
            
        for stat_key, obs_val in obs_stats.items():
            if pd.isna(obs_val):
                logging.debug(f"Skipping {stat_key} for subject {subj_id}: observed value is NaN")
                continue

            # Get all non-NaN simulated values for this statistic
            sim_vals_for_stat = [s.get(stat_key) for s in sim_stats_list if s and stat_key in s]
            sim_vals_for_stat = [s for s in sim_vals_for_stat if pd.notna(s)]
            n_sims = len(sim_vals_for_stat)
            
            # Skip if not enough simulations
            if n_sims < min_sim_reps:
                logging.debug(
                    f"Insufficient simulations ({n_sims}) for {stat_key} in subject {subj_id}. "
                    f"Need at least {min_sim_reps} for reliable statistics."
                )
                rows.append({
                    'subject_id': subj_id, 'stat_key': stat_key, 'observed': obs_val,
                    'simulated_mean': np.nan, 'simulated_std': np.nan,
                    'covered_95': np.nan, 'ci_95_lower': np.nan, 'ci_95_upper': np.nan,
                    'n_simulations_for_stat': n_sims
                })
                continue
            
            try:
                # Calculate statistics
                sim_mean = np.nanmean(sim_vals_for_stat)
                sim_std = np.nanstd(sim_vals_for_stat, ddof=1) if n_sims > 1 else np.nan
                
                # Calculate percentiles only if we have enough data
                if n_sims >= 5:  # Minimum for meaningful percentiles
                    ci_95_lower = np.nanpercentile(sim_vals_for_stat, 2.5)
                    ci_95_upper = np.nanpercentile(sim_vals_for_stat, 97.5)
                    covered_95 = int(ci_95_lower <= obs_val <= ci_95_upper)
                else:
                    ci_95_lower = ci_95_upper = np.nan
                    covered_95 = np.nan
                
                rows.append({
                    'subject_id': subj_id, 'stat_key': stat_key, 'observed': obs_val,
                    'simulated_mean': sim_mean, 'simulated_std': sim_std,
                    'covered_95': covered_95, 'ci_95_lower': ci_95_lower, 'ci_95_upper': ci_95_upper,
                    'n_simulations_for_stat': n_sims
                })
                
            except Exception as e:
                logging.warning(f"Error calculating coverage for {stat_key} in subject {subj_id}: {e}")
                rows.append({
                    'subject_id': subj_id, 'stat_key': stat_key, 'observed': obs_val,
                    'simulated_mean': np.nan, 'simulated_std': np.nan,
                    'covered_95': np.nan, 'ci_95_lower': np.nan, 'ci_95_upper': np.nan,
                    'n_simulations_for_stat': n_sims
                })
            
    if not rows:
        logging.warning("No data for coverage calculation.")
        return pd.DataFrame()
    return pd.DataFrame(rows)

def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for running PPCs."""
    args = parse_args(argv)
    setup_logging(debug=args.debug)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.output_dir = Path(str(args.output_dir) + f'_{args.ppc_version}')
    logging.info(f"Run mode: Empirical Fits Only. Output will be saved to: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / 'run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Load empirical fits
    logging.info(f"Loading empirical fits from {args.fitted_params_file}")
    df_fits = pd.read_csv(args.fitted_params_file)
    if 'subject' not in df_fits.columns and 'subject_id' in df_fits.columns:
        df_fits.rename(columns={'subject_id': 'subject'}, inplace=True)
    df_fits['subject'] = df_fits['subject'].astype(str).str.strip()
    logging.info(f"Loaded {len(df_fits)} subjects' fitted parameters.")

    # Load and preprocess empirical behavioral data
    if not args.roberts_data_file or not args.roberts_data_file.exists():
        raise FileNotFoundError(f"Roberts data file not found: {args.roberts_data_file}")
    logging.info(f"Loading Roberts behavioral data from {args.roberts_data_file}")
    df_raw_empirical = pd.read_csv(args.roberts_data_file)
    df_raw_empirical.columns = [col.lower().replace(' ', '_') for col in df_raw_empirical.columns]
    
    # Basic preprocessing specific to Roberts data
    df_empirical_processed = df_raw_empirical[df_raw_empirical['trialtype'].str.lower() == 'target'].copy()
    df_empirical_processed['subject'] = df_empirical_processed['subject'].astype(str).str.strip()
    df_empirical_processed['choice'] = pd.to_numeric(df_empirical_processed['choice'], errors='coerce').fillna(0).astype(int)
    df_empirical_processed['rt'] = pd.to_numeric(df_empirical_processed['rt'], errors='coerce')
    df_empirical_processed['prob'] = pd.to_numeric(df_empirical_processed['prob'], errors='coerce')
    df_empirical_processed['is_gain_frame'] = df_empirical_processed['frame'].str.lower() == 'gain'
    df_empirical_processed['time_constrained'] = df_empirical_processed['cond'].str.lower() == 'tc'
    if 'valence_score' not in df_empirical_processed.columns: # Add if missing
        df_empirical_processed['valence_score'] = 0.0 
    if 'norm_category_for_trial' not in df_empirical_processed.columns:
        df_empirical_processed['norm_category_for_trial'] = 'default'
    
    all_empirical_subjects = sorted(df_empirical_processed['subject'].unique())
    fitted_subjects = set(df_fits['subject'].unique())
    
    subjects_to_run = sorted(list(set(all_empirical_subjects) & fitted_subjects))
    if args.subject_ids:
        requested_subjects = {s.strip() for s in args.subject_ids.split(",")}
        subjects_to_run = sorted(list(requested_subjects.intersection(subjects_to_run)))
        missing_ids = requested_subjects - set(subjects_to_run)
        if missing_ids:
            logging.warning(f"Requested subject IDs not found or missing fits/data: {missing_ids}")

    if not subjects_to_run:
        raise ValueError("No valid subjects to process with both data and fits.")
    logging.info(f"Processing {len(subjects_to_run)} subjects: {subjects_to_run[:5]}...")

    ppc_results: Dict[str, Any] = {}
    processed_count = 0  # Initialize counter for successfully processed subjects
    
    # Ensure RNG is seeded once at the start of main()
    # np.random.seed(args.seed) # This is already done earlier

    with tqdm(total=len(subjects_to_run), desc="Processing subjects", unit="subject") as pbar:
        for subj_id in subjects_to_run: # Simplified tqdm
            logging.info(f"--- Subject {subj_id} ---")
            subj_fits_row = df_fits[df_fits['subject'] == subj_id].iloc[0]

            all_sim_stats_for_subject = [] # Collects stats from all pseudo-posterior samples for this subject

            # Determine number of parameter sets to use for this subject
            num_param_sets_to_simulate = 1
            if args.n_pseudo_posterior_samples > 1:
                num_param_sets_to_simulate = args.n_pseudo_posterior_samples
                logging.info(f"Generating {num_param_sets_to_simulate} pseudo-posterior parameter samples for subject {subj_id}.")
            else:
                logging.info(f"Using mean fitted parameters for subject {subj_id}.")

            # Prepare subject's empirical trial data (moved before parameter sampling loop)
            df_subj_empirical = df_empirical_processed[df_empirical_processed['subject'] == subj_id].copy()
            if df_subj_empirical.empty:
                logging.warning(f"No empirical trial data for subject {subj_id}. Skipping.")
                pbar.update(1)
                continue
            
            # Clean and prepare trial_data for get_trial_structure & observed stats
            if df_subj_empirical['rt'].isnull().any():
                rt_mean = df_subj_empirical['rt'].mean()
                df_subj_empirical = df_subj_empirical.copy()  # Avoid SettingWithCopyWarning
                df_subj_empirical.loc[:, 'rt'] = df_subj_empirical['rt'].fillna(rt_mean)
                logging.warning(f"Filled {df_subj_empirical['rt'].isnull().sum()} NaN RTs with mean {rt_mean} for subject {subj_id}")
            
            # Drop rows if critical columns (prob, choice) are NaN AFTER attempting to fill RT
            df_subj_empirical.dropna(subset=['prob', 'choice', 'rt'], inplace=True)
            if df_subj_empirical.empty:
                logging.warning(f"No valid trials after NaN drop for subject {subj_id}. Skipping.")
                pbar.update(1)
                continue

            # Prepare trial structure and observed stats (once per subject)
            try:
                trial_struct = get_trial_structure(df_subj_empirical)
                observed_summary_stats = calculate_summary_stats(df_subj_empirical, ROBERTS_SUMMARY_STAT_KEYS)
            except Exception as e:
                logging.error(f"Error preparing data or obs stats for subject {subj_id}: {e}", exc_info=True)
                pbar.update(1)
                continue
                
            if not trial_struct or not observed_summary_stats:
                logging.warning(f"Empty trial_struct or obs_stats for subject {subj_id}")
                pbar.update(1)
                continue

            for i_param_sample in range(num_param_sets_to_simulate):
                agent_init_kwargs = {}
                
                # Sample or get mean for each of the 6 core DDM parameters
                for fit_param_csv_key in PARAM_NAMES_FITTED:  # e.g., 'v_norm_mean', 'a_0_mean', ...
                    base_name_for_sampling = fit_param_csv_key.replace('_mean', '')  # 'v_norm', 'a_0', ...
                    mean_val = float(subj_fits_row[fit_param_csv_key])
                    
                    if args.n_pseudo_posterior_samples > 1:
                        std_col_csv_key = fit_param_csv_key.replace('_mean', '_std')
                        std_val = abs(mean_val * 0.1) if mean_val != 0 else 0.01  # Default if not found
                        if std_col_csv_key in subj_fits_row and pd.notna(subj_fits_row[std_col_csv_key]):
                            s = float(subj_fits_row[std_col_csv_key])
                            std_val = max(s, 1e-6) if s > 0 else (abs(mean_val * 0.1) if mean_val != 0 else 0.01)
                        else:
                            logging.warning(f"Std dev for {fit_param_csv_key} not found/invalid, using 10% of mean.")
                        
                        sampled_value = np.random.normal(loc=mean_val, scale=std_val)
                    else:
                        sampled_value = mean_val
                        
                    # Apply constraints (clipping)
                    if base_name_for_sampling == "v_norm":
                        sampled_value = np.clip(sampled_value, 0.1, 5.0)
                    elif base_name_for_sampling == "a_0":
                        sampled_value = np.clip(sampled_value, 0.5, 3.0)
                    elif base_name_for_sampling == "w_s_eff":
                        sampled_value = np.clip(sampled_value, 0.3, 0.7)
                    elif base_name_for_sampling == "t_0":
                        sampled_value = np.clip(sampled_value, 0.1, 0.5)
                        if sampled_value < 0.01:
                            sampled_value = 0.01  # Hard minimum
                    elif base_name_for_sampling == "alpha_gain":
                        sampled_value = np.clip(sampled_value, -1.0, 1.0)
                    elif base_name_for_sampling == "beta_val":
                        sampled_value = np.clip(sampled_value, -1.0, 1.0)
                        
                    # Assign to agent_init_kwargs using the agent's __init__ parameter names
                    if base_name_for_sampling == "v_norm":
                        agent_init_kwargs['v_norm'] = sampled_value
                    elif base_name_for_sampling == "a_0":
                        agent_init_kwargs['a_0'] = sampled_value  # Will be scaled after sampling & clipping
                    elif base_name_for_sampling == "w_s_eff":
                        agent_init_kwargs['w_s_eff'] = sampled_value
                    elif base_name_for_sampling == "t_0":
                        agent_init_kwargs['t_0'] = sampled_value
                    elif base_name_for_sampling == "alpha_gain":
                        agent_init_kwargs['alpha_gain'] = sampled_value
                    elif base_name_for_sampling == "beta_val":
                        agent_init_kwargs['beta_val'] = sampled_value
                
                # Apply threshold scaling to the a_0 that will go into init
                agent_init_kwargs['a_0'] *= args.threshold_scale
                
                # Add fixed parameters for agent __init__
                agent_init_kwargs['logit_z0'] = 0.0
                agent_init_kwargs['log_tau_norm'] = -0.693147
                
                try:
                    agent = MVNESAgent(**agent_init_kwargs)
                except ValueError as e_val:
                    logging.error(f"  Skipping sample {i_param_sample+1} due to invalid param during agent init: {e_val}")
                    logging.error(f"    Problematic init_kwargs: {agent_init_kwargs}")
                    continue
                
                # Prepare the 'sim_control_params' dictionary for simulate_one_full_dataset
                # This will be passed as 'params' to agent.run_mvnes_trial
                sim_params_for_trial_runner = {
                    'w_s': agent.w_s_eff,  # Use the sampled w_s_eff directly
                    'w_n': agent.v_norm,
                    'threshold_a': agent.a_0,
                    't': agent.t_0,
                    'alpha_gain': agent.alpha_gain,
                    'beta_val': agent.beta_val,
                    'logit_z0': agent.logit_z0,
                    'log_tau_norm': agent.log_tau_norm,
                    'meta_cognitive_on': False,
                    'noise_std_dev': 1.0,
                    'dt': 0.01
                }
                
                # Run simulations for this parameter set
                sim_reps_stats = simulate_n_replicates(
                    agent=agent,
                    trial_struct=trial_struct,
                    base_sim_control_params=sim_params_for_trial_runner,
                    n_reps=args.n_sim_reps
                )
                all_sim_stats_for_subject.extend(sim_reps_stats)
            
            # After processing all pseudo-posterior samples (or the single mean set) for the subject
            if not all_sim_stats_for_subject:
                logging.warning(f"No valid simulation results collected for subject {subj_id} after all samples/reps.")
                pbar.update(1)
                continue # to the next subject
                
            ppc_results[subj_id] = {
                'observed_summary_stats': observed_summary_stats,
                'simulated_summary_stats_list': all_sim_stats_for_subject,
                'params_reference': "sampled_from_mean_std" if args.n_pseudo_posterior_samples > 1 else "mean_values",
                'n_pseudo_posterior_samples_used': num_param_sets_to_simulate,
                'n_reps_per_sample': args.n_sim_reps
            }
            processed_count += 1
            pbar.update(1)

    logging.info(f"Successfully processed {processed_count}/{len(subjects_to_run)} subjects.")

    if not ppc_results:
        logging.warning("No subjects were successfully processed. Exiting.")
        return

    output_file = args.output_dir / 'ppc_results.json'
    with open(output_file, 'w') as f:
        json.dump(ppc_results, f, indent=2)
    logging.info(f"Saved PPC results to {output_file}")
    
    plot_ppc_distributions(ppc_results, args.output_dir, subjects_to_run)
    
    coverage_df = calculate_detailed_coverage(ppc_results)
    if not coverage_df.empty:
        coverage_file = args.output_dir / 'coverage_statistics.csv'
        coverage_df.to_csv(coverage_file, index=False)
        logging.info(f"Saved coverage statistics to {coverage_file}")
        
        # Print summary (example)
        if 'covered_95' in coverage_df.columns:
            avg_coverage = coverage_df.groupby('stat_key')['covered_95'].mean()
            logging.info("\nAverage 95% Coverage Per Statistic:\n" + avg_coverage.to_string())
    else:
        logging.info("No coverage data to save or display.")

    logging.info("PPC script finished.")


if __name__ == "__main__":
    main()