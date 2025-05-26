# Filename: run_meta_ppc.py
# Purpose: Perform simulations for Posterior Predictive Checks (PPC)
#          across different meta-cognitive monitoring modes.

import argparse
import logging
import json
import sys
from pathlib import Path
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# --- Project-Specific Imports ---
# Attempt to import from the project structure
script_dir = Path(__file__).resolve().parent
# Common potential locations for the 'src' and 'fit_script_dir'
# Assuming this script might be in the root, or in a 'scripts' folder, etc.
project_root_paths = [
    script_dir, 
    script_dir.parent, # If script is in a 'scripts' or 'empirical_fitting' subdir
    script_dir.parent.parent # If script is in a deeper subdir
]

agent_mvnes_found = False
fit_script_imports_found = False

for prp in project_root_paths:
    potential_src_dir = prp / 'src'
    potential_fit_script_dir = prp # Assuming fit_nes_to_roberts_data_sbc_focused.py is in root or a sibling dir like 'empirical_fitting'
    
    if not agent_mvnes_found and (potential_src_dir / 'agent_mvnes.py').exists():
        if str(potential_src_dir) not in sys.path:
            sys.path.insert(0, str(potential_src_dir))
        try:
            from agent_mvnes import MVNESAgent
            agent_mvnes_found = True
            logging.info(f"Found and imported MVNESAgent from: {potential_src_dir}")
        except ImportError as e:
            logging.warning(f"Attempted to import MVNESAgent from {potential_src_dir} but failed: {e}")
            if str(potential_src_dir) in sys.path: sys.path.remove(str(potential_src_dir))


    # Try to import from fit_nes_to_roberts_data_sbc_focused.py
    # This is a bit unusual, usually common components are in a shared lib.
    # For now, let's assume it's findable and add its directory to path if needed.
    # Best practice would be to refactor shared components.
    # We need to find where fit_nes_to_roberts_data_sbc_focused.py is.
    # Common locations: project root, or a directory like 'empirical_fitting'
    
    # Try current project root path first for the fitting script
    fit_script_file = prp / 'fit_nes_to_roberts_data_sbc_focused.py'
    if not fit_script_imports_found and fit_script_file.exists():
        fit_script_dir_str = str(prp)
        if fit_script_dir_str not in sys.path:
            sys.path.insert(0, fit_script_dir_str)
        try:
            from fit_nes_to_roberts_data_sbc_focused import (
                prepare_trial_template, calculate_summary_stats_roberts,
                get_roberts_summary_stat_keys, CONDITIONS_ROBERTS,
                BASE_SIM_PARAMS, PARAM_NAMES, PRIOR_LOW, PRIOR_HIGH,
                SUBJECT_TRIAL_STRUCTURE_TEMPLATE, OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE
            )
            fit_script_imports_found = True
            logging.info(f"Successfully imported components from fit_nes_to_roberts_data_sbc_focused.py found in: {fit_script_dir_str}")
        except ImportError as e:
            logging.warning(f"Attempted to import from fit_nes_to_roberts_data_sbc_focused.py in {fit_script_dir_str} but failed: {e}")
            if fit_script_dir_str in sys.path and fit_script_dir_str != str(potential_src_dir) : sys.path.remove(fit_script_dir_str)
    
    if agent_mvnes_found and fit_script_imports_found:
        break

# Fallback if not found in typical project structures relative to this script
if not agent_mvnes_found:
    try: # Try assuming 'src' is a direct subdir of cwd if script is in root
        if str(Path.cwd() / 'src') not in sys.path and (Path.cwd() / 'src' / 'agent_mvnes.py').exists():
             sys.path.insert(0, str(Path.cwd() / 'src'))
        from agent_mvnes import MVNESAgent
        agent_mvnes_found = True
        logging.info("Found and imported MVNESAgent from ./src (relative to CWD).")
    except ImportError as e:
        logging.error(f"MVNESAgent import failed from ./src: {e}. Ensure it's in the Python path.")
        sys.exit(1)

if not fit_script_imports_found:
    try: # Try assuming fit_nes_to_roberts_data_sbc_focused.py is in CWD
        if str(Path.cwd()) not in sys.path and (Path.cwd() / 'fit_nes_to_roberts_data_sbc_focused.py').exists():
            sys.path.insert(0, str(Path.cwd()))
        from fit_nes_to_roberts_data_sbc_focused import (
            prepare_trial_template, calculate_summary_stats_roberts,
            get_roberts_summary_stat_keys, CONDITIONS_ROBERTS,
            BASE_SIM_PARAMS, PARAM_NAMES, PRIOR_LOW, PRIOR_HIGH,
            SUBJECT_TRIAL_STRUCTURE_TEMPLATE, OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE
        )
        fit_script_imports_found = True
        logging.info("Successfully imported components from fit_nes_to_roberts_data_sbc_focused.py found in CWD.")
    except ImportError as e:
        logging.error(f"Import from fit_nes_to_roberts_data_sbc_focused.py failed from CWD: {e}. Ensure it's in the Python path.")
        sys.exit(1)


# --- Global Configurations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', force=True)
PPC_MODES = ["no_meta", "meta_observe_only", "meta_tune_active"]
# NUM_PPC_SIMULATIONS_PER_MODE will be an argparse option
# NUM_TEMPLATE_TRIALS_FOR_PPC will be an argparse option

def main():
    parser = argparse.ArgumentParser(description="Run PPC for NES model with meta-cognitive monitoring.")
    parser.add_argument('--num_sims_per_mode', type=int, default=100, help="Number of full DDM simulations per PPC mode.")
    parser.add_argument('--num_template_trials', type=int, default=100, help="Number of trials per DDM simulation (from template).")
    parser.add_argument('--seed', type=int, default=None, help="Random seed.")
    parser.add_argument('--output_dir', type=str, default="ppc_meta_results", help="Directory to save PPC results.")
    parser.add_argument('--roberts_data_file', type=str, default="./Roberts_Framing_Data/ftp_osf_data.csv", help="Path to Roberts et al. data for trial template.")
    parser.add_argument('--force_cpu', action='store_true', help="Force CPU usage even if CUDA is available.")
    
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}. Seed: {args.seed}")

    # Setup output directory
    output_dir_path = Path(args.output_dir)
    output_data_path = output_dir_path / "data"
    output_plots_path = output_dir_path / "plots" # Placeholder for future plots
    output_data_path.mkdir(parents=True, exist_ok=True)
    output_plots_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory setup at: {output_dir_path.resolve()}")

    # Prepare trial template (relies on global SUBJECT_TRIAL_STRUCTURE_TEMPLATE)
    try:
        prepare_trial_template(Path(args.roberts_data_file), args.num_template_trials, args.seed)
        if SUBJECT_TRIAL_STRUCTURE_TEMPLATE is None or SUBJECT_TRIAL_STRUCTURE_TEMPLATE.empty:
            raise ValueError("SUBJECT_TRIAL_STRUCTURE_TEMPLATE is empty after prepare_trial_template call.")
    except Exception as e_template:
        logging.error(f"Failed to prepare trial template: {e_template}", exc_info=True)
        sys.exit(1)
    
    summary_stat_keys = get_roberts_summary_stat_keys()

    # Define Representative Parameter Set (using prior means)
    # Ensure PARAM_NAMES, PRIOR_LOW, PRIOR_HIGH from the fit script include all necessary params
    if len(PARAM_NAMES) != len(PRIOR_LOW) or len(PARAM_NAMES) != len(PRIOR_HIGH):
        logging.error("Mismatch in lengths of PARAM_NAMES, PRIOR_LOW, PRIOR_HIGH. Check imports/definitions.")
        sys.exit(1)
        
    prior_means_tensor = (PRIOR_LOW + PRIOR_HIGH) / 2.0
    param_set_dict = {name: val.item() for name, val in zip(PARAM_NAMES, prior_means_tensor)}
    
    # Ensure all required base params for agent.run_mvnes_trial are present
    # BASE_SIM_PARAMS usually contains dt, noise_std_dev, max_time
    # The agent config will also have dt.
    # Critical params like w_s, w_n, threshold_a, t, alpha_gain, beta_val, log_taus should be in PARAM_NAMES
    logging.info(f"Using representative parameter set (prior means): {param_set_dict}")

    # Data Collection Structures
    all_trial_data = defaultdict(list)
    all_summary_stats = defaultdict(list)
    all_meta_logs = defaultdict(list)

    # Initialize Agent (config can be expanded if more defaults are needed by agent's __init__)
    agent_config = {
        'dt': BASE_SIM_PARAMS.get('dt', 0.01),
        # Add other necessary agent-specific config defaults if MVNESAgent relies on them
        # e.g., meta_monitor_interval_ms, etc., if not passed via `params` in run_mvnes_trial
        # and if their defaults in MVNESAgent.__init__ are not desired.
        # For this PPC, we control meta flags via params to run_mvnes_trial.
    }
    agent = MVNESAgent(config=agent_config)

    # Simulation Loop
    for mode in PPC_MODES:
        logging.info(f"Starting PPC simulations for mode: {mode}")
        
        enable_meta_monitor_flag = False
        enable_meta_tuning_flag = False

        if mode == "no_meta":
            enable_meta_monitor_flag = False
            enable_meta_tuning_flag = False
        elif mode == "meta_observe_only":
            enable_meta_monitor_flag = True
            enable_meta_tuning_flag = False
        elif mode == "meta_tune_active":
            enable_meta_monitor_flag = True
            enable_meta_tuning_flag = True

        logging.info(f"Mode '{mode}': Monitor={enable_meta_monitor_flag}, Tuning={enable_meta_tuning_flag}")

        for sim_idx in range(args.num_sims_per_mode):
            if (sim_idx + 1) % (args.num_sims_per_mode // 10 or 1) == 0:
                 logging.info(f"  Simulation {sim_idx + 1}/{args.num_sims_per_mode} for mode '{mode}'...")

            current_simulation_trial_results = []
            current_simulation_meta_logs = []

            for trial_idx, trial_info in SUBJECT_TRIAL_STRUCTURE_TEMPLATE.iterrows():
                salience_input = trial_info['prob']
                norm_input = 1.0 if trial_info['is_gain_frame'] else -1.0
                
                agent_run_params = {
                    **param_set_dict, # Contains w_s, w_n, threshold_a, t, alpha_gain, beta_val, log_taus
                    **BASE_SIM_PARAMS, # Contains dt, noise_std_dev, max_time
                    'valence_score_trial': trial_info['valence_score'],
                    'norm_type': trial_info['norm_type'],
                    'enable_meta_monitor': enable_meta_monitor_flag,
                    'enable_meta_tuning': enable_meta_tuning_flag,
                    'frame': trial_info['frame'], # For frame-dependent threshold/drift adjustments in agent
                    'subj_id': f"ppc_sim_{sim_idx}", # For potential debugging prints in agent
                    'trial_idx': trial_idx,
                    'debug': False # Set to True for verbose agent debug prints
                }
                
                trial_output = agent.run_mvnes_trial(
                    salience_input=salience_input,
                    norm_input=norm_input,
                    params=agent_run_params
                )
                
                trial_data_to_store = {
                    'sim_idx': sim_idx,
                    'trial_idx_template': trial_idx,
                    'rt': trial_output.get('rt', np.nan),
                    'choice': trial_output.get('choice', np.nan),
                    'frame': trial_info['frame'],
                    'cond': trial_info['cond'],
                    'norm_type': trial_info['norm_type'],
                    'valence_score': trial_info['valence_score'],
                    'timeout': trial_output.get('timeout', True)
                }
                current_simulation_trial_results.append(trial_data_to_store)
                current_simulation_meta_logs.extend(trial_output.get('log_meta_events', []))
            
            df_sim_batch = pd.DataFrame(current_simulation_trial_results)
            # OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE should be populated by prepare_trial_template
            summary_stats = calculate_summary_stats_roberts(df_sim_batch, summary_stat_keys, OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE)
            
            all_trial_data[mode].append(df_sim_batch)
            all_summary_stats[mode].append(summary_stats)
            all_meta_logs[mode].append(current_simulation_meta_logs) # list of lists of strings

    # Save Data
    logging.info("Saving PPC simulation data...")
    for mode in PPC_MODES:
        # Save trial data
        if all_trial_data[mode]:
            df_all_trials_mode = pd.concat(all_trial_data[mode], ignore_index=True)
            trial_data_filename = output_data_path / f"ppc_trial_data_{mode}_sims{args.num_sims_per_mode}_trials{args.num_template_trials}.csv.gz"
            df_all_trials_mode.to_csv(trial_data_filename, index=False, compression="gzip")
            logging.info(f"Saved trial data for mode '{mode}' to {trial_data_filename}")

        # Save summary stats
        if all_summary_stats[mode]:
            df_summary_stats_mode = pd.DataFrame(all_summary_stats[mode])
            summary_stats_filename = output_data_path / f"ppc_summary_stats_{mode}_sims{args.num_sims_per_mode}_trials{args.num_template_trials}.csv"
            df_summary_stats_mode.to_csv(summary_stats_filename, index=False)
            logging.info(f"Saved summary stats for mode '{mode}' to {summary_stats_filename}")

        # Save meta logs
        if all_meta_logs[mode]:
            meta_logs_filename = output_data_path / f"ppc_meta_logs_{mode}_sims{args.num_sims_per_mode}_trials{args.num_template_trials}.json"
            # Flatten the list of lists of logs for easier JSON storage if desired, or save as is.
            # For now, save as a list of lists (each inner list is one simulation's meta logs)
            with open(meta_logs_filename, 'w') as f:
                json.dump(all_meta_logs[mode], f, indent=2)
            logging.info(f"Saved meta logs for mode '{mode}' to {meta_logs_filename}")
            
    # --- PPC Plotting Section (Conceptual) ---
    # The saved data (CSVs and JSON) should be loaded into a Jupyter Notebook
    # or a separate Python script for detailed PPC plotting.
    # This typically involves:
    # 1. Histograms of observed summary statistics vs. distributions of simulated summary statistics.
    # 2. Comparing RT distributions (observed vs. simulated) per condition.
    # 3. Comparing choice proportions (observed vs. simulated) per condition.
    # 4. Analyzing meta-log data for patterns in threshold tuning.
    logging.info("PPC data generation complete. Further analysis and plotting should be done in a dedicated analysis script or notebook.")

if __name__ == "__main__":
    main()
