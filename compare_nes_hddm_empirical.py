import pandas as pd
import numpy as np
import arviz as az
import hddm # For HDDM-ext fitting
# import sbi # If using sbi directly for NES NPE posteriors
# Add any other imports needed for your NES fitting (e.g., torch)
import pickle
import time
import logging
from pathlib import Path

# --- Configuration ---
EMPIRICAL_DATA_FILE = 'path/to/your/n45_framing_data.csv' # REPLACE
OUTPUT_DIR = Path('./model_comparison_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MCMC/NPE Sampling parameters (general guidance, adjust as per your methods)
N_SAMPLES_POSTERIOR = 2000 # Number of posterior samples to draw/use per subject
N_CHAINS_MCMC = 4          # For MCMC methods like HDDM
N_BURN_MCMC = 1000         # For MCMC methods

# --- Logging Setup ---
log_file = OUTPUT_DIR / 'model_comparison.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
)

# --- Helper Functions ---
def get_log_likelihood_pointwise(model_type, posterior_samples_dict, data_df):
    """
    Calculate pointwise log-likelihood for each posterior sample and data point.
    This is the most crucial and model-specific part.
    You need to implement the likelihood function for both NES and HDDM-ext.

    Args:
        model_type (str): 'nes' or 'hddm_ext'
        posterior_samples_dict (dict): 
            Keys are parameter names, values are numpy arrays of shape 
            (n_chains, n_draws_per_chain, n_subjects_or_trials).
            Or, for NPE, it might be (n_draws, n_parameters) for a single subject,
            and this function would be called per subject.
        data_df (pd.DataFrame): The empirical data for which to calculate likelihood.
                                This might be data for ONE subject if fitting sequentially.

    Returns:
        np.ndarray: Pointwise log-likelihood array of shape (n_chains*n_draws, n_trials)
    """
    logging.info(f"Calculating pointwise log-likelihood for {model_type}...")
    # This function needs to iterate through each posterior sample (parameter set)
    # and for each parameter set, calculate the log probability of each observed trial (rt, choice)
    # given those parameters and the respective model's likelihood function (e.g., Wiener likelihood).

    # Example structure (VERY SIMPLIFIED - REPLACE WITH ACTUAL LIKELIHOODS)
    n_posterior_draws_total = posterior_samples_dict['a'].shape[0] * posterior_samples_dict['a'].shape[1] # chains * draws
    n_trials = len(data_df)
    log_lik_pointwise = np.zeros((n_posterior_draws_total, n_trials))

    # Flatten posterior samples for easier iteration if they are (chains, draws, ...)
    flat_samples = {}
    for param, values in posterior_samples_dict.items():
        if values.ndim == 3: # (chains, draws, subjects_if_hierarchical_group_param)
             # This part needs careful handling if params are hierarchical at group level
             # For now, assume we are dealing with subject-specific posteriors flattened
            flat_samples[param] = values.reshape(-1, values.shape[-1] if values.ndim > 2 else 1)
        elif values.ndim == 2: # (draws, params_for_one_subject)
            flat_samples[param] = values
        else: # (draws,) for single param
            flat_samples[param] = values.reshape(-1,1)


    for i in range(n_posterior_draws_total):
        current_params = {param: samples[i] for param, samples in flat_samples.items()}
        
        for trial_idx, trial_data in enumerate(data_df.itertuples()):
            rt_obs = trial_data.rt
            choice_obs = trial_data.response # Assuming 'response' is 0 or 1

            if model_type == 'nes':
                # --- !!! YOUR NES LIKELIHOOD CALCULATION HERE !!! ---
                # This would involve taking current_params for NES (w_n_eff, a, t, w_s_eff, etc.)
                # and calculating the log_pdf of the Wiener distribution (or your NES equivalent)
                # for the observed rt and choice, given trial covariates (e.g., conflict level).
                # You might use hddm.wfpt.wiener_like_rldf if your NES is DDM-based.
                # log_p_trial = calculate_nes_log_likelihood_for_trial(current_params, trial_data)
                log_p_trial = -5 # Placeholder - high penalty (low likelihood)
                # Example using HDDM's likelihood (adjust params and inputs)
                # try:
                #     # Ensure params are mapped correctly for hddm.wfpt.wiener_like_rldf
                #     # It expects v, sv, a, z, sz, t, st
                #     # You'll need to compute v based on NES logic and current_params
                #     v_trial = nes_calculate_drift(current_params, trial_data.condition) # Placeholder
                #     log_p_trial = hddm.wfpt.wiener_like_rldf(
                #         rt_obs, choice_obs, v_trial, current_params['a'], current_params['t'], 
                #         current_params.get('z', 0.5), 0, 0, 0, 1e-5 # sv, sz, st, err
                #     )
                # except OverflowError: # Catch potential numerical issues
                #    log_p_trial = -np.inf
                log_lik_pointwise[i, trial_idx] = log_p_trial

            elif model_type == 'hddm_ext':
                # --- !!! YOUR HDDM-EXT LIKELIHOOD CALCULATION HERE !!! ---
                # Similar to NES, but using parameters from HDDM-ext.
                # If HDDM-ext is fitted with hddm.HDDMRegressor, you could potentially
                # adapt parts of its internal likelihood calculation or use wfpt.
                # log_p_trial = calculate_hddm_ext_log_likelihood_for_trial(current_params, trial_data)
                log_p_trial = -6 # Placeholder
                # Example using HDDM's likelihood (adjust params and inputs)
                # try:
                #     # v_trial needs to be calculated from HDDM's regression (v_Intercept + v_condition_effect)
                #     v_trial = hddm_calculate_drift(current_params, trial_data.frame_condition) # Placeholder
                #     log_p_trial = hddm.wfpt.wiener_like_rldf(
                #         rt_obs, choice_obs, v_trial, current_params['a'], current_params['t'],
                #         current_params.get('z', 0.5), 0, 0, 0, 1e-5 # sv, sz, st, err
                #     )
                # except OverflowError:
                #    log_p_trial = -np.inf
                log_lik_pointwise[i, trial_idx] = log_p_trial
            else:
                raise ValueError("Unknown model_type")
    
    # Ensure no -inf values that ArviZ dislikes for loo/waic calculations, replace with large negative
    log_lik_pointwise[np.isneginf(log_lik_pointwise)] = -1e100 
    return log_lik_pointwise

# --- Model Fitting Functions (PLACEHOLDERS - REPLACE WITH YOUR ACTUAL CODE) ---

def fit_nes_model(subject_data, subject_id):
    """
    Fits the NES model to a single subject's data using NPE.
    Returns an ArviZ InferenceData object.
    """
    logging.info(f"Fitting NES model for subject {subject_id}...")
    # 1. Load your trained NPE model
    #    npe_posterior_estimator = load_my_trained_npe_model() # REPLACE
    
    # 2. Calculate summary statistics for the subject_data
    #    summary_stats_obs = calculate_nes_summary_stats(subject_data) # REPLACE
    
    # 3. Draw posterior samples from the NPE
    #    posterior_samples_nes = npe_posterior_estimator.sample((N_SAMPLES_POSTERIOR,), 
    #                                                          x=summary_stats_obs, 
    #                                                          show_progress_bars=False)
    #    posterior_samples_nes_np = posterior_samples_nes.numpy() # Convert to numpy
    
    # --- !!! THIS IS A MAJOR PLACEHOLDER - REPLACE !!! ---
    # Create dummy posterior samples for NES for demonstration
    # These should be (n_draws, n_parameters)
    # Parameters: w_n_eff, a, t, w_s_eff (and others if in your minimal NES)
    # Assuming 4 parameters for NES as per your abstract
    dummy_posterior_nes = {
        'w_n_eff': np.random.normal(1.0, 0.3, (1, N_SAMPLES_POSTERIOR)), # (chain, draw)
        'a': np.random.normal(1.0, 0.2, (1, N_SAMPLES_POSTERIOR)),
        't': np.random.normal(0.3, 0.1, (1, N_SAMPLES_POSTERIOR)),
        'w_s_eff': np.random.normal(0.7, 0.2, (1, N_SAMPLES_POSTERIOR))
    }
    # Reshape to (chain, draw, params) for ArviZ if needed, or handle dict of (chain, draw)
    # ArviZ prefers a dictionary where keys are param names and values are (chains, draws, *dims)

    # 4. Calculate pointwise log-likelihood (MOST IMPORTANT STEP)
    #    log_likelihood_nes = get_log_likelihood_pointwise(
    #        'nes', 
    #        dummy_posterior_nes, # Should be posterior_samples_nes_np in a suitable dict format
    #        subject_data
    #    )
    # For dummy run, create dummy log_likelihood
    log_likelihood_nes = np.random.normal(-5, 1, (N_SAMPLES_POSTERIOR, len(subject_data)))
    log_likelihood_nes = log_likelihood_nes.reshape(1, N_SAMPLES_POSTERIOR, len(subject_data)) # (chain, draw, trial)
    # --- END MAJOR PLACEHOLDER ---

    # 5. Create ArviZ InferenceData object
    # Posterior samples need to be in shape (n_chains, n_draws, n_params_dims...)
    # For ArviZ, it's best if posterior_samples_nes_np is a dict:
    # {'param1': array[chain, draw, ...], 'param2': array[chain, draw, ...]}
    
    idata_nes = az.from_dict(
        posterior=dummy_posterior_nes, # Should be your actual posterior samples dictionary
        log_likelihood={'y': log_likelihood_nes}, # 'y' is a common name for observations
        dims={'w_n_eff': ['subject_dim'], 'a':['subject_dim'], 't':['subject_dim'], 'w_s_eff':['subject_dim']}, # Example if params have a subject dim
        coords={'subject_dim': [subject_id]}
    )
    logging.info(f"NES fitting complete for subject {subject_id}.")
    return idata_nes


def fit_hddm_ext_model(subject_data, subject_id):
    """
    Fits the extended HDDM model to a single subject's data.
    Returns an ArviZ InferenceData object.
    """
    logging.info(f"Fitting HDDM-ext model for subject {subject_id}...")
    # 1. Define your HDDM-ext model formula and parameters
    #    model_formula_hddm = "v ~ 1 + frame_condition" # Example
    #    include_hddm = ['a', 't', 'z'] # Example
    
    # 2. Fit HDDM model
    #    m_hddm = hddm.HDDMRegressor(subject_data, model_formula_hddm, include=include_hddm, ...)
    #    m_hddm.sample(N_SAMPLES_POSTERIOR + N_BURN_MCMC, burn=N_BURN_MCMC, chains=N_CHAINS_MCMC)
    
    # --- !!! THIS IS A MAJOR PLACEHOLDER - REPLACE !!! ---
    # Create dummy posterior samples for HDDM-ext
    # HDDM traces are usually (draws, ) per parameter. Need to collect per chain if N_CHAINS_MCMC > 1
    # Or get traces directly from m_hddm.model.nodes_db into ArviZ format
    dummy_posterior_hddm = {
        'a': np.random.normal(1.2, 0.2, (N_CHAINS_MCMC, N_SAMPLES_POSTERIOR)),
        't': np.random.normal(0.25, 0.1, (N_CHAINS_MCMC, N_SAMPLES_POSTERIOR)),
        'v_Intercept': np.random.normal(0.5, 0.3, (N_CHAINS_MCMC, N_SAMPLES_POSTERIOR)),
        'v_frame_condition[T.loss]': np.random.normal(-0.2, 0.2, (N_CHAINS_MCMC, N_SAMPLES_POSTERIOR)) # Example
    }
    # For dummy run, create dummy log_likelihood
    log_likelihood_hddm = np.random.normal(-6, 1, (N_CHAINS_MCMC * N_SAMPLES_POSTERIOR, len(subject_data)))
    log_likelihood_hddm = log_likelihood_hddm.reshape(N_CHAINS_MCMC, N_SAMPLES_POSTERIOR, len(subject_data))
    # --- END MAJOR PLACEHOLDER ---

    # 3. Convert HDDM traces to ArviZ InferenceData
    #    This can be done using az.from_pymc3(trace=m_hddm.get_traces()) if PyMC3 based
    #    Or manually construct the dictionary for az.from_dict
    #    Crucially, you need the POINTWISE log likelihood for WAIC/LOO.
    #    HDDM models store this if you sample with `m.sample(..., store_post_pred_like=True)`
    #    or you calculate it manually using get_log_likelihood_pointwise.

    #    If m_hddm.gen_post_pred_dict(groupby=['condition_col']) works,
    #    you might find 'logp' in its output for observed data.
    #    Otherwise, manual calculation is needed.

    # log_likelihood_hddm = get_log_likelihood_pointwise(
    #     'hddm_ext', 
    #     dummy_posterior_hddm, # Should be actual HDDM posterior samples dict
    #     subject_data
    # )

    idata_hddm = az.from_dict(
        posterior=dummy_posterior_hddm,
        log_likelihood={'y': log_likelihood_hddm},
        # Add dims and coords if parameters are per subject or condition
    )
    logging.info(f"HDDM-ext fitting complete for subject {subject_id}.")
    return idata_hddm

# --- Main Script ---
def main():
    logging.info("Starting model comparison: NES vs HDDM-ext.")
    
    # 1. Load Data
    try:
        data_all_subjects = pd.read_csv(EMPIRICAL_DATA_FILE)
        logging.info(f"Loaded empirical data: {len(data_all_subjects)} total trials, {data_all_subjects['subject_id'].nunique()} subjects.")
    except FileNotFoundError:
        logging.error(f"Empirical data file not found: {EMPIRICAL_DATA_FILE}")
        return

    all_subjects_idata_nes = {}
    all_subjects_idata_hddm = {}

    # Iterate over subjects (or fit hierarchically if your functions support it)
    # For WAIC/LOO, ArviZ expects pointwise log-likelihood, often computed per subject then concatenated
    # or from a hierarchical model that provides it.
    # This script structure assumes fitting per subject and then combining.

    for subj_id in data_all_subjects['subject_id'].unique():
        logging.info(f"Processing subject: {subj_id}")
        subject_data_df = data_all_subjects[data_all_subjects['subject_id'] == subj_id]

        # Fit NES
        try:
            idata_nes_subj = fit_nes_model(subject_data_df, subj_id)
            all_subjects_idata_nes[subj_id] = idata_nes_subj
            idata_nes_subj.to_netcdf(OUTPUT_DIR / f"idata_nes_subj_{subj_id}.nc")
            logging.info(f"Saved NES InferenceData for subject {subj_id}")
        except Exception as e:
            logging.error(f"Failed to fit NES for subject {subj_id}: {e}", exc_info=True)

        # Fit HDDM-ext
        try:
            idata_hddm_subj = fit_hddm_ext_model(subject_data_df, subj_id)
            all_subjects_idata_hddm[subj_id] = idata_hddm_subj
            idata_hddm_subj.to_netcdf(OUTPUT_DIR / f"idata_hddm_ext_subj_{subj_id}.nc")
            logging.info(f"Saved HDDM-ext InferenceData for subject {subj_id}")
        except Exception as e:
            logging.error(f"Failed to fit HDDM-ext for subject {subj_id}: {e}", exc_info=True)

    # If fitting was done per subject, concatenate InferenceData objects
    if all_subjects_idata_nes:
        # This needs careful handling of coordinates if subject is a dim in posteriors
        # For now, let's assume we want to compare models overall, summing/averaging ICs
        logging.info("Concatenating InferenceData objects across subjects is complex and depends on model structure.")
        logging.info("For now, we will compute ICs per subject and then average/sum if appropriate.")
        # idata_nes_all = az.concat(list(all_subjects_idata_nes.values()), dim="subject_coord_name") # Needs a common coord
        # idata_hddm_all = az.concat(list(all_subjects_idata_hddm.values()), dim="subject_coord_name")
    else:
        logging.error("No NES models were successfully fitted.")
        return
    if not all_subjects_idata_hddm:
        logging.error("No HDDM-ext models were successfully fitted.")
        return
        
    # 3. Compute WAIC and LOO (per subject for now)
    waic_results = []
    loo_results = []

    for subj_id in data_all_subjects['subject_id'].unique():
        subj_res = {'subject_id': subj_id}
        if subj_id in all_subjects_idata_nes:
            idata_nes = all_subjects_idata_nes[subj_id]
            subj_res['waic_nes'] = az.waic(idata_nes, pointwise=False, scale="log").waic # d_waic for sum
            subj_res['loo_nes'] = az.loo(idata_nes, pointwise=False, scale="log").loo     # elpd_loo for sum
        else:
            subj_res['waic_nes'] = np.nan
            subj_res['loo_nes'] = np.nan

        if subj_id in all_subjects_idata_hddm:
            idata_hddm = all_subjects_idata_hddm[subj_id]
            subj_res['waic_hddm'] = az.waic(idata_hddm, pointwise=False, scale="log").waic
            subj_res['loo_hddm'] = az.loo(idata_hddm, pointwise=False, scale="log").loo
        else:
            subj_res['waic_hddm'] = np.nan
            subj_res['loo_hddm'] = np.nan
        
        waic_results.append(subj_res) # Actually these are individual ICs, not differences yet
        loo_results.append(subj_res) # Using same dict structure

    df_waic = pd.DataFrame(waic_results)
    df_loo = pd.DataFrame(loo_results)

    # Summing ICs (common for non-hierarchical fits summed over subjects)
    # Note: For true hierarchical models, you'd typically get one IC for the whole model.
    # If your models are effectively fit per subject, summing log-likelihoods (and thus ICs) is one approach.
    total_waic_nes = df_waic['waic_nes'].sum()
    total_waic_hddm = df_waic['waic_hddm'].sum()
    total_loo_nes = df_loo['loo_nes'].sum() # This is sum of elpd_loo
    total_loo_hddm = df_loo['loo_hddm'].sum()

    logging.info(f"\n--- Overall Model Comparison (summed across subjects) ---")
    logging.info(f"Total WAIC NES: {total_waic_nes:.2f}")
    logging.info(f"Total WAIC HDDM-ext: {total_waic_hddm:.2f}")
    logging.info(f"Total ELPD_LOO NES: {total_loo_nes:.2f}") # Higher is better for ELPD
    logging.info(f"Total ELPD_LOO HDDM-ext: {total_loo_hddm:.2f}")

    # 4. Compute Differences and Bayes Factor
    # Note: ArviZ compare function is better for this if you have single InferenceData objects for full hierarchical models
    # For WAIC, lower is better. Delta_WAIC = WAIC_model - WAIC_best_model
    delta_waic = total_waic_nes - total_waic_hddm 
    logging.info(f"ΔWAIC (NES - HDDM-ext): {delta_waic:.2f} (Positive favors HDDM-ext, Negative favors NES)")

    # For LOO ELPD, higher is better. Delta_elpd = ELPD_model - ELPD_other_model
    delta_elpd_nes_vs_hddm = total_loo_nes - total_loo_hddm 
    logging.info(f"Δelpd_loo (NES - HDDM-ext): {delta_elpd_nes_vs_hddm:.2f} (Positive favors NES, Negative favors HDDM-ext)")
    
    # Standard error of delta_elpd is needed for robust comparison.
    # az.compare could give this if comparing two full models.
    # For summed LOOs, SE is more complex. For now, we proceed with the point estimate.

    if not np.isnan(delta_elpd_nes_vs_hddm):
        # Bayes Factor BF₁₀ (NES vs HDDM-ext) ≈ exp(Δelpd_loo / 2) -- this is unconventional and approximate.
        # A more standard BF approach uses marginal likelihoods.
        # The exp(Δelpd) is more related to relative likelihood of models.
        # Vehtari et al. suggest caution with BFs from LOO.
        # However, if requested: BF_nes_hddm = exp(delta_elpd_nes_vs_hddm)
        # The prompt suggests exp(Δelpd / 2)
        bf_approx_nes_vs_hddm = np.exp(delta_elpd_nes_vs_hddm / 2)
        logging.info(f"Approximate Bayes Factor (BF_NES_HDDM_ext) from Δelpd_loo/2: {bf_approx_nes_vs_hddm:.2f}")
        logging.warning("Note: Bayes Factors derived from LOO ELPD differences are approximate and should be interpreted with caution.")
    else:
        logging.warning("Could not calculate approximate Bayes Factor due to NaN Δelpd_loo.")


    # Save comparison summary
    summary_comp = {
        'total_waic_nes': [total_waic_nes],
        'total_waic_hddm': [total_waic_hddm],
        'delta_waic_nes_minus_hddm': [delta_waic],
        'total_elpd_loo_nes': [total_loo_nes],
        'total_elpd_loo_hddm': [total_loo_hddm],
        'delta_elpd_loo_nes_minus_hddm': [delta_elpd_nes_vs_hddm],
        'approx_bf_nes_vs_hddm': [bf_approx_nes_vs_hddm if not np.isnan(delta_elpd_nes_vs_hddm) else np.nan]
    }
    df_summary_comp = pd.DataFrame(summary_comp)
    df_summary_comp.to_csv(OUTPUT_DIR / 'model_comparison_summary.csv', index=False)
    logging.info(f"Saved overall comparison summary to {OUTPUT_DIR / 'model_comparison_summary.csv'}")

    df_waic.to_csv(OUTPUT_DIR / 'waic_per_subject.csv', index=False)
    df_loo.to_csv(OUTPUT_DIR / 'loo_per_subject.csv', index=False)
    logging.info("Saved per-subject WAIC and LOO results.")


    # 5. (Optional) 10-fold Cross-Validation Outline
    logging.info("\n--- 10-Fold Cross-Validation (Outline) ---")
    logging.info("To implement 10-fold CV:")
    logging.info("1. Split your N=45 subject data into 10 folds (e.g., using sklearn.model_selection.KFold).")
    logging.info("2. For each fold:")
    logging.info("   a. Train NES on 9 folds, predict/calculate log-likelihood on the held-out fold.")
    logging.info("   b. Train HDDM-ext on 9 folds, predict/calculate log-likelihood on the held-out fold.")
    logging.info("3. Sum the out-of-sample log-likelihoods across folds for each model.")
    logging.info("4. Compare total out-of-sample log-likelihoods (higher is better).")
    logging.info("This is computationally intensive as it requires 10 full model fits for each model type.")

if __name__ == "__main__":
    main()