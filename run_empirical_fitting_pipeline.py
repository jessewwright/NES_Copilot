#!/usr/bin/env python3
"""
NES Co-Pilot: Empirical Fitting Pipeline

This script orchestrates the empirical fitting of the 6-parameter NES model
to the Roberts et al. dataset using a pre-trained and validated NPE.
It leverages the NES Co-Pilot modules.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml # For potentially loading a master config for this specific workflow

# Assuming NES_Copilot is in PYTHONPATH or structured as a package
# This will depend on how you run it (e.g., from NES_Copilot parent dir with `python -m NES_Copilot.run_empirical_fitting_pipeline`)
# Or if NES_Copilot/ is added to sys.path
try:
    # If running from Hegemonikon_Project and NES_Copilot is a subdir
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'NES_Copilot'))
    from nes_copilot.config_manager_fixed import FixedConfigManager as ConfigManager # Use your fixed version
    from nes_copilot.data_prep_module import DataPrepModule 
    from nes_copilot.summary_stats_module import SummaryStatsModule 
    from nes_copilot.empirical_fit_module import EmpiricalFitModule # This is the NEW module we're effectively designing
    from nes_copilot.checkpoint_manager import CheckpointManager # For loading the NPE
    from nes_copilot.logging_manager import setup_logging # Assuming you have this
    from nes_copilot.stats_schema import ROBERTS_SUMMARY_STAT_KEYS, validate_summary_stats # Central schema
except ImportError as e:
    print(f"Error importing NES Co-Pilot modules: {e}. Ensure NES_Copilot is in PYTHONPATH or script is run correctly.")
    sys.exit(1)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Empirical Fitting for 6-Parameter NES Model")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the master YAML configuration file for this empirical fitting run.')
    parser.add_argument('--npe_checkpoint_file', type=str, default=None,
                        help='(Override) Path to the specific pre-trained NPE .pt file to use.')
    parser.add_argument('--output_dir_override', type=str, default=None,
                        help='(Override) Master output directory for this run.')
    parser.add_argument('--num_posterior_samples', type=int, default=None,
                        help='(Override) Number of posterior samples per subject.')
    parser.add_argument('--seed', type=int, default=None, help='(Override) Random seed.')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage.')
    
    args = parser.parse_args()

    # 1. Initialize Configuration
    # ConfigManager should load the YAML specified by --config
    # This YAML will define paths, parameters for this specific fitting workflow
    try:
        config_manager = ConfigManager(args.config)
        cfg = config_manager.get_master_config() # Get the full config dict
    except Exception as e:
        logger.error(f"Failed to load or parse configuration from {args.config}: {e}", exc_info=True)
        sys.exit(1)

    # Override config values with command-line arguments if provided
    # Output directory setup
    output_dir_base = args.output_dir_override if args.output_dir_override else cfg.get('output_dir_base', './empirical_fits')
    run_name = cfg.get('experiment', {}).get('name', 'nes_empirical_fit')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir_base) / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging setup (assuming logging_manager.setup_logging can take a file path)
    log_file_path = output_dir / f"empirical_fit_log_{timestamp}.log"
    setup_logging(log_file_path=log_file_path, console_level_str=cfg.get('logging',{}).get('level','INFO'))
    logger.info(f"Empirical Fitting Run Initialized. Output will be in: {output_dir}")
    logger.info(f"Using configuration file: {args.config}")

    # Seed and Device
    seed = args.seed if args.seed is not None else cfg.get('experiment', {}).get('random_seed', int(time.time()))
    np.random.seed(seed); torch.manual_seed(seed)
    device_name = 'cpu' if args.force_cpu else cfg.get('compute_device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logger.info(f"Using device: {device}. Seed: {seed}")

    # Get relevant paths and parameters from config
    npe_checkpoint_path = Path(args.npe_checkpoint_file if args.npe_checkpoint_file else cfg['npe_checkpoint_to_load'])
    roberts_data_path = Path(cfg['data_paths']['roberts_raw_csv'])
    num_posterior_samples_per_subject = args.num_posterior_samples if args.num_posterior_samples is not None else cfg['fitting_settings']['num_posterior_samples']

    # 2. Initialize Core Modules Needed for Fitting
    # data_prep_module = DataPrepModule(config_manager) # If DataPrepModule handles schema and path via config
    # summary_stats_module = SummaryStatsModule(config_manager) # If SummaryStatsModule handles schema via config
    # For simplicity now, directly use functions/schema assuming they are imported
    
    # Ensure summary stats schema is validated
    try:
        validate_summary_stats() # From stats_schema.py
        stat_keys = list(ROBERTS_SUMMARY_STAT_KEYS) # From stats_schema.py
        logger.info(f"Using {len(stat_keys)} summary statistics from stats_schema.")
    except Exception as e:
        logger.error(f"Failed to load or validate summary stats schema: {e}", exc_info=True)
        sys.exit(1)

    # This is where the new/refactored EmpiricalFitModule would come in.
    # For now, let's integrate the logic directly here, assuming EmpiricalFitModule
    # would encapsulate something similar.
    
    # 3. Load the Pre-trained 5-Parameter NPE
    # The load_npe_posterior_object function needs to be available,
    # either in this script or imported from a utility module (e.g., checkpoint_manager.py or npe_training_module.py)
    # Let's assume a robust loading function exists (adapted from your PPC script)
    
    # Define prior that MATCHES the loaded NPE's training
    # These should ideally also come from the loaded NPE's metadata or be in the config for this run.
    # For the 6-parameter model:
    current_param_names = ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain', 'beta_val']
    current_prior_low = torch.tensor([0.1,  0.5,  0.2,  0.05, 0.5, -0.5], device=device) # Example TIGHT prior
    current_prior_high = torch.tensor([1.5, 1.8, 1.2, 0.5, 0.9, 0.5], device=device)  # Example TIGHT prior
    
    # Check if config overrides these
    model_params_config = cfg.get('model', {}).get('parameters', {})
    if model_params_config:
        current_prior_low_list = []
        current_prior_high_list = []
        temp_param_names = []
        for p_name in current_param_names: # Ensure order
            if p_name in model_params_config:
                temp_param_names.append(p_name)
                current_prior_low_list.append(model_params_config[p_name][0])
                current_prior_high_list.append(model_params_config[p_name][1])
            else:
                logger.warning(f"Parameter {p_name} not found in config model.parameters. Using hardcoded script defaults.")
        if len(temp_param_names) == len(current_param_names): # All params found in config
             current_prior_low = torch.tensor(current_prior_low_list, device=device, dtype=torch.float32)
             current_prior_high = torch.tensor(current_prior_high_list, device=device, dtype=torch.float32)
             logger.info("Loaded prior bounds from configuration file.")
        else:
             logger.warning("Mismatch between PARAM_NAMES and config parameters. Using hardcoded script priors.")


    sbi_prior_for_loading = BoxUniform(low=current_prior_low, high=current_prior_high, device=device)

    try:
        # This function needs to be robust as developed previously
        # For example, copy load_npe_posterior_object from your working run_ppc_nes_roberts.py
        from nes_copilot.checkpoint_manager import load_npe_posterior_object_from_checkpoint # Assuming you create this
        loaded_npe_posterior_object = load_npe_posterior_object_from_checkpoint(
            npe_checkpoint_path, 
            sbi_prior_for_loading, # Pass the prior it was trained with
            device,
            expected_param_names=current_param_names, # For validation
            expected_num_summary_stats=len(stat_keys) # For validation
        )
    except Exception as e:
        logger.error(f"Failed to load NPE from {npe_checkpoint_path}. Exiting. Error: {e}", exc_info=True)
        sys.exit(1)

    # 4. Load and Preprocess Empirical Data
    try:
        df_all_empirical_data = pd.read_csv(roberts_data_path)
        # Standard preprocessing
        if 'trialType' in df_all_empirical_data.columns:
            df_all_empirical_data = df_all_empirical_data[df_all_empirical_data['trialType'] == 'target'].copy()
        for col_numeric in ['rt', 'choice', 'prob', 'sureOutcome', 'endow']: # Add sureOutcome, endow if needed for valence
            if col_numeric in df_all_empirical_data.columns:
                df_all_empirical_data[col_numeric] = pd.to_numeric(df_all_empirical_data[col_numeric], errors='coerce')
        df_all_empirical_data.dropna(subset=['subject','frame','cond','rt','choice','prob'], inplace=True)
        df_all_empirical_data['time_constrained'] = df_all_empirical_data['cond'] == 'tc'
        df_all_empirical_data['is_gain_frame'] = df_all_empirical_data['frame'] == 'gain'
        
        # Add VALENCE SCORES and NORM CATEGORY to empirical data for MVNESAgent consistency
        # This assumes valence_processor.py is available and has get_trial_valence_scores_for_df
        try:
            from nes_copilot.valence_processor import get_trial_valence_scores_for_df # You'll need to create this
            df_all_empirical_data = get_trial_valence_scores_for_df(df_all_empirical_data) # Adds 'valence_score' column
        except ImportError:
            logger.warning("ValenceProcessor not found, valence_score will be default (0).")
            df_all_empirical_data['valence_score'] = 0.0 
        df_all_empirical_data['norm_category_for_trial'] = 'default' # For consistency with agent

        logger.info(f"Loaded and preprocessed empirical data: {len(df_all_empirical_data)} target trials.")
    except Exception as e:
        logger.error(f"Failed to load or preprocess empirical data: {e}", exc_info=True)
        sys.exit(1)

    # 5. Fit Each Subject
    all_subject_fit_results = []
    subject_ids = sorted(df_all_empirical_data['subject'].unique())

    for subj_id in tqdm(subject_ids, desc="Fitting Empirical Subjects"):
        logger.info(f"--- Processing Subject {subj_id} for Empirical Fit ---")
        df_subj_empirical = df_all_empirical_data[df_all_empirical_data['subject'] == subj_id]
        if df_subj_empirical.empty:
            logging.warning(f"No empirical data for subject {subj_id}. Skipping."); continue
        
        # Calculate observed summary statistics for this subject
        # We need a robust calculate_summary_stats that uses stats_schema
        # Assuming calculate_summary_stats_roberts is available and uses schema
        try:
            from nes_copilot.summary_stats_module import calculate_summary_stats_roberts # If this is the one
            obs_stats_dict_actual = calculate_summary_stats_roberts(df_subj_empirical, stat_keys)
        except ImportError: # Fallback if not in a module structure
            from stats_schema import calculate_summary_stats_roberts_fallback as calculate_summary_stats # placeholder name
            obs_stats_dict_actual = calculate_summary_stats(df_subj_empirical, stat_keys)


        obs_stats_vector_actual = [obs_stats_dict_actual.get(k, -999.0) for k in stat_keys]
        obs_stats_tensor_for_conditioning = torch.tensor(obs_stats_vector_actual, dtype=torch.float32).to(device)

        if torch.isnan(obs_stats_tensor_for_conditioning).any() or torch.isinf(obs_stats_tensor_for_conditioning).any():
            logging.warning(f"Subject {subj_id} has NaN/Inf in summary stats. Skipping. Stats: {obs_stats_vector_actual[:5]}")
            # Add placeholder result
            subject_result = {'subject_id': subj_id, 'error': 'NaN/Inf in summary_stats'}
            for p_name in current_param_names:
                subject_result[f'{p_name}_mean'] = np.nan
            all_subject_fit_results.append(subject_result)
            continue
            
        subject_result = {'subject_id': subj_id}
        try:
            with torch.no_grad():
                posterior_samples = loaded_npe_posterior_object.sample(
                    (num_posterior_samples_per_subject,),
                    x=obs_stats_tensor_for_conditioning.unsqueeze(0), # Add batch dimension
                    show_progress_bars=False
                ).cpu()
            
            for i_param, p_name in enumerate(current_param_names):
                subject_result[f'{p_name}_mean'] = posterior_samples[:, i_param].mean().item()
                subject_result[f'{p_name}_median'] = posterior_samples[:, i_param].median().item()
                subject_result[f'{p_name}_std'] = posterior_samples[:, i_param].std().item()
            subject_result['error'] = None

        except Exception as e_fit:
            logging.error(f"Empirical fitting failed for subject {subj_id}: {e_fit}", exc_info=True)
            subject_result['error'] = str(e_fit)
            for p_name in current_param_names: subject_result[f'{p_name}_mean'] = np.nan
        
        all_subject_fit_results.append(subject_result)

    # 6. Save Aggregate Results
    df_final_fits = pd.DataFrame(all_subject_fit_results)
    fit_results_csv_path = output_dir / f"empirical_fitting_results_{len(current_param_names)}params_seed{seed}.csv"
    df_final_fits.to_csv(fit_results_csv_path, index=False, float_format='%.6f')
    logging.info(f"Saved all subject empirical fitting results to: {fit_results_csv_path}")

    # (Optional: Add basic plotting of parameter distributions across subjects here)

    logging.info("Empirical fitting pipeline finished successfully.")


if __name__ == "__main__":
    main()