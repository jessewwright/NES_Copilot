import sys
import torch
import pandas as pd
import numpy as np
import arviz as az  # For WAIC/LOO calculation
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import argparse
import os
from sbi.inference import SNPE_C
from sbi.utils import BoxUniform
import torch.distributions as dist
from datetime import datetime

# Assuming sbi is installed and working
from sbi.inference.posteriors import DirectPosterior as Posterior 
# Or if your .pt file contains the full inference object:
# from sbi.inference import SNPE # or SNLE, AALR etc.

# Your agent and data loading/simulation functions will be needed
# For this example, I'll assume you have:
# - load_empirical_data_for_subject(subject_id) -> returns (choices, rts) or similar trial data
# - calculate_log_likelihood_of_trial(trial_data_point, params_dict, agent_instance) -> returns log_p_trial
#   This is the HARD part. For DDM, this might involve:
#   - For choice: Simulating many times with `params_dict` to get P(choice_observed | params_dict), then log.
#   - For RT: Simulating many RTs with `params_dict` to get a distribution, then evaluating P(rt_observed | params_dict) 
#     (e.g., using KDE on simulated RTs or assuming a known RT distribution like Wald/ex-Gaussian and fitting its parameters).
#   A simpler (but less accurate) approach for DDM might be to use the "first passage time density" if you can evaluate it.

# Placeholder for the hard part - this needs to be implemented based on your model
def calculate_pointwise_log_likelihood(observed_subject_data, posterior_samples_for_subject, agent_class, param_names_agent, num_sims=100):
    """
    Calculate a matrix of log-likelihoods: (n_posterior_samples, n_data_points_for_subject).
    This is a placeholder and needs careful implementation for a DDM.
    
    Args:
        observed_subject_data: List/array of observed trials for one subject. 
                               Each trial could be a tuple (choice, rt).
        posterior_samples_for_subject: NumPy array of shape (n_draws, n_params).
        agent_class: The MVNESAgent class.
        param_names_agent: List of parameter names in the order they appear in posterior_samples.

    Returns:
        log_lik_matrix: NumPy array of shape (n_draws, n_trials_for_subject).
    """
    n_draws = posterior_samples_for_subject.shape[0]
    n_trials = len(observed_subject_data)
    log_lik_matrix = np.zeros((n_draws, n_trials))
    
    print(f"Calculating log-likelihoods for {n_draws} samples and {n_trials} trials...")
    
    # Pre-allocate arrays for simulation results
    all_choices = np.zeros((n_draws, n_trials, num_sims), dtype=int)
    all_rts = np.zeros((n_draws, n_trials, num_sims))
    
    # First, run all simulations
    for s_idx in range(n_draws):
        params_array = posterior_samples_for_subject[s_idx, :]
        
        # Map parameters to agent's __init__
        agent_init_kwargs = {name: float(val) for name, val in zip(param_names_agent, params_array)}
        
        try:
            agent = agent_class(**agent_init_kwargs)
        except ValueError as e:
            print(f"Warning: Skipping posterior sample {s_idx} due to invalid parameters: {e}")
            log_lik_matrix[s_idx, :] = -np.inf
            continue
            
        for t_idx, trial_data in enumerate(observed_subject_data):
            # Run multiple simulations for this trial condition
            for sim in range(num_sims):
                try:
                    # Run the simulation with the current parameters and trial conditions
                    choice, rt = agent.run_mvnes_trial(
                        is_gain_frame=trial_data['is_gain_frame'],
                        time_constrained=trial_data['time_constrained'],
                        valence_score_trial=trial_data['valence_score'],
                        norm_category_for_trial=trial_data['norm_category_code']
                    )
                    all_choices[s_idx, t_idx, sim] = choice
                    all_rts[s_idx, t_idx, sim] = rt
                except Exception as e:
                    print(f"Error in simulation {sim} for sample {s_idx}, trial {t_idx}: {e}")
                    all_choices[s_idx, t_idx, sim] = -1  # Invalid choice
                    all_rts[s_idx, t_idx, sim] = 0.0
    
    # Now calculate log-likelihoods using the simulation results
    for s_idx in range(n_draws):
        if np.all(log_lik_matrix[s_idx, :] == -np.inf):
            continue  # Skip invalid samples
            
        for t_idx, trial_data in enumerate(observed_subject_data):
            obs_choice = trial_data['choice']
            obs_rt = trial_data['rt']
            
            # Calculate choice probability (fraction of simulations that match observed choice)
            matching_choices = (all_choices[s_idx, t_idx] == obs_choice).sum()
            p_choice = (matching_choices + 0.5) / (num_sims + 1)  # Add-1/2 smoothing
            log_p_choice = np.log(max(p_choice, 1e-9))
            
            # Calculate RT likelihood using KDE on simulated RTs
            sim_rts = all_rts[s_idx, t_idx]
            valid_rts = sim_rts[sim_rts > 0]  # Filter out any invalid simulations
            
            if len(valid_rts) > 10:  # Need enough samples for KDE
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(valid_rts)
                    rt_density = kde(obs_rt)[0]
                    log_p_rt = np.log(max(rt_density, 1e-9))
                except:
                    log_p_rt = np.log(1e-9)  # Fallback if KDE fails
            else:
                # Fallback: use normal approximation if not enough samples
                if len(valid_rts) > 1:
                    rt_mean = np.mean(valid_rts)
                    rt_std = max(np.std(valid_rts), 0.1)  # Minimum std to avoid division by zero
                    rt_likelihood = (1 / (rt_std * np.sqrt(2 * np.pi))) * \
                                  np.exp(-0.5 * ((obs_rt - rt_mean) / rt_std) ** 2)
                    log_p_rt = np.log(max(rt_likelihood, 1e-9))
                else:
                    log_p_rt = np.log(1e-9)  # Fallback if no valid RTs
            
            # Combine choice and RT log-likelihoods (assuming independence)
            log_lik_matrix[s_idx, t_idx] = log_p_choice + log_p_rt

    return log_lik_matrix


def load_posterior_samples(pt_file_path: Path, expected_param_names: List[str] = None, n_samples: int = 1000) -> Dict[str, np.ndarray]:
    """
    Load parameter samples from a .pt file containing an sbi density estimator.
    
    Args:
        pt_file_path: Path to the .pt file containing the trained sbi model
        expected_param_names: List of expected parameter names (for validation)
        n_samples: Number of samples to draw from the posterior
        
    Returns:
        Dictionary of parameter names to numpy arrays of samples
    """
    try:
        print(f"\nLoading model from: {pt_file_path}")
        checkpoint = torch.load(pt_file_path, map_location='cpu')
        
        # Print debug info about the checkpoint
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        print(f"Parameter names in checkpoint: {checkpoint.get('param_names')}")
        print(f"sbi version: {checkpoint.get('sbi_version', 'unknown')}")
        
        # Extract parameter names from checkpoint or use expected names
        param_names = checkpoint.get('param_names', expected_param_names)
        if param_names is None:
            raise ValueError("No parameter names found in checkpoint and none provided")
        
        # For sbi 0.22.0, we need to use the posterior_potential approach
        print("Using sbi 0.22.0 compatible loading method")
        
        # Create a dummy prior distribution
        prior_low = torch.ones(len(param_names)) * -10.0
        prior_high = torch.ones(len(param_names)) * 10.0
        prior = BoxUniform(low=prior_low, high=prior_high)
        
        # Initialize the inference object
        inference = SNPE_C(prior=prior, density_estimator='nsf')
        
        # Create a dummy observation
        num_summary_stats = checkpoint.get('num_summary_stats', 10)  # Adjust based on your data
        x_o = torch.zeros((1, num_summary_stats))
        
        # Load the posterior
        if 'density_estimator_state_dict' in checkpoint:
            print("Loading density estimator state dict...")
            
            # Create a dummy training round to initialize the network
            _ = inference.append_simulations(
                theta=torch.randn(10, len(param_names)),  # Dummy parameters
                x=torch.randn(10, num_summary_stats)      # Dummy observations
            )
            
            # Train for 0 epochs just to initialize the network
            _ = inference.train(max_num_epochs=0)
            
            # Load the state dict
            inference._neural_net.load_state_dict(checkpoint['density_estimator_state_dict'])
            
            # Build the posterior
            posterior = inference.build_posterior()
            
            # Generate samples from the posterior
            print(f"Generating {n_samples} samples from the posterior...")
            posterior_samples = posterior.sample((n_samples,), x=x_o, show_progress_bars=True)
            
            # Convert to numpy and create dictionary
            samples_np = posterior_samples.detach().cpu().numpy()
            samples_dict = {name: samples_np[:, i] for i, name in enumerate(param_names)}
            
            print(f"Generated samples with shape: {samples_np.shape}")
            print(f"Parameter names in samples: {list(samples_dict.keys())}")
            
            return samples_dict
        else:
            raise ValueError("No density_estimator_state_dict found in checkpoint")
        
    except Exception as e:
        print(f"Error loading or sampling from model {pt_file_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_model_file(pt_file_path: Path, model_param_names: list, 
                     all_subjects_trial_data: dict, 
                     agent_class_to_use: any,
                     n_posterior_draws: int = 200) -> dict:
    """
    Loads a model, generates posterior samples, calculates pointwise log-likelihoods,
    and computes WAIC/LOO.
    """
    print(f"\n--- Processing model: {pt_file_path.name} ---")
    print(f"Expected parameters: {model_param_names}")
    
    # Load parameter samples
    try:
        print(f"Loading model with expected parameters: {model_param_names}")
        param_samples = load_posterior_samples(
            pt_file_path, 
            expected_param_names=model_param_names, 
            n_samples=n_posterior_draws
        )
        
        if not param_samples:
            print(f"No valid parameter samples found in {pt_file_path.name}")
            return None
        
        print(f"Successfully loaded parameters: {list(param_samples.keys())}")
        
        # Check if we have all expected parameters
        missing_params = [p for p in model_param_names if p not in param_samples]
        if missing_params:
            print(f"Warning: Missing expected parameters: {missing_params}")
            print("Using available parameters instead.")
        
        # Use available parameters that match our expected names
        available_params = [p for p in model_param_names if p in param_samples]
        if not available_params:
            print(f"Error: No expected parameters found in the model file.")
            return None
            
        # Convert to the format expected by the rest of the code
        # (n_samples, n_params) array where columns are in model_param_names order
        posterior_samples = np.column_stack([param_samples[p] for p in available_params])
        print(f"Created parameter matrix with shape: {posterior_samples.shape}")
        
        # Save the samples for debugging
        output_dir = Path("debug_samples")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_file = output_dir / f"samples_{pt_file_path.stem}_{timestamp}.npy"
        np.save(sample_file, posterior_samples)
        print(f"Saved samples to {sample_file}")
        
    except Exception as e:
        print(f"Error processing {pt_file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # This will store arviz InferenceData objects for each subject
    all_subjects_idata = {}
    
    # Get a list of subjects present in the trial data
    subject_ids = list(all_subjects_trial_data.keys())
    if not subject_ids:
        print("No subject data provided.")
        return None

    all_waic_scores = {}
    all_loo_scores = {}
    
    for subj_id, subject_data in all_subjects_trial_data.items():
        print(f"Processing subject {subj_id}...")
        
        # Calculate pointwise log-likelihoods using the pre-loaded samples
        log_lik_matrix = calculate_pointwise_log_likelihood(
            observed_subject_data=subject_data,
            posterior_samples_for_subject=posterior_samples,
            agent_class=agent_class_to_use,
            param_names_agent=model_param_names
        )
        
        # Convert to ArviZ InferenceData format
        log_lik_matrix = np.expand_dims(log_lik_matrix, 0)  # Add chain dimension
        data = {"log_likelihood": log_lik_matrix}
        
        # Calculate WAIC and LOO
        try:
            idata = az.from_dict(
                data,
                log_likelihood={"log_lik": ["chain", "draw", "log_lik"]},
                coords={"chain": [0], "draw": np.arange(n_posterior_draws)},
                dims={"log_lik": ["chain", "draw", "log_lik"]},
            )
            
            waic_result = az.waic(idata, pointwise=True)
            loo_result = az.loo(idata, pointwise=True)
            
            all_waic_scores[subj_id] = waic_result.waic
            all_loo_scores[subj_id] = loo_result.loo
            print(f"  Subject {subj_id}: WAIC = {waic_result.waic:.2f}, LOO = {loo_result.loo:.2f}")
            
        except Exception as e:
            print(f"  Error calculating WAIC/LOO for subject {subj_id}: {e}")
            continue
    
    # Calculate total WAIC and LOO scores across all subjects
    total_waic = np.sum(list(all_waic_scores.values())) if all_waic_scores else np.nan
    total_loo = np.sum(list(all_loo_scores.values())) if all_loo_scores else np.nan
    
    print(f"\nModel {pt_file_path.name} - Overall (summed):")
    print(f"  Total WAIC: {total_waic:.2f}")
    print(f"  Total LOO: {total_loo:.2f}")
    
    return {
        "waic_per_subject": all_waic_scores, 
        "loo_per_subject": all_loo_scores, 
        "total_waic": total_waic, 
        "total_loo": total_loo,
        "model_file": str(pt_file_path.name)
    }

def load_and_prep_all_subject_data(roberts_data_file_path: Path, subject_ids: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Load and prepare trial data for all subjects from the Roberts dataset.
    
    Args:
        roberts_data_file_path: Path to the Roberts data CSV file
        subject_ids: Optional list of subject IDs to include (if None, include all)
        
    Returns:
        Dictionary mapping subject IDs to lists of trial data dictionaries
    """
    print(f"Loading data from {roberts_data_file_path}")
    df = pd.read_csv(roberts_data_file_path)
    
    # Print column names to help with debugging
    print("Available columns in the data:", df.columns.tolist())
    
    # Try to find the subject ID column (case insensitive)
    subject_col = next((col for col in df.columns if 'subj' in col.lower() or 'id' in col.lower()), None)
    if subject_col is None:
        print("Error: Could not find subject ID column in the data. Available columns:", df.columns.tolist())
        return {}
        
    print(f"Using '{subject_col}' as the subject ID column")
    
    # Try to find choice and RT columns
    choice_col = next((col for col in df.columns if 'choice' in col.lower() or 'resp' in col.lower()), None)
    rt_col = next((col for col in df.columns if 'rt' in col.lower() or 'response_time' in col.lower()), None)
    
    if not choice_col or not rt_col:
        print(f"Error: Could not find required columns. Choice col: {choice_col}, RT col: {rt_col}")
        return {}
    
    # Convert columns to appropriate types and handle missing values
    df[choice_col] = pd.to_numeric(df[choice_col], errors='coerce')
    df[rt_col] = pd.to_numeric(df[rt_col], errors='coerce')
    
    # Drop rows where either choice or RT is missing
    df = df.dropna(subset=[choice_col, rt_col])
    
    if df.empty:
        print("Error: No valid data after removing rows with missing values")
        return {}
    
    print(f"Found {len(df)} valid trials after removing rows with missing values")
    
    # Group by subject and convert to list of trial dictionaries
    all_subjects = {}
    for subj_id, subj_df in df.groupby(subject_col):
        subj_id = str(subj_id)
        if subject_ids is not None and subj_id not in subject_ids:
            continue
            
        trials = []
        for _, trial in subj_df.iterrows():
            try:
                trial_data = {
                    'choice': int(trial[choice_col]),
                    'rt': float(trial[rt_col])
                }
                
                # Add additional trial information
                trial_data.update({
                    'frame': trial.get('frame'),
                    'prob': trial.get('prob'),
                    'trial_type': trial.get('trialType'),
                    'timelimit': trial.get('timelimit')
                })
                
                trials.append(trial_data)
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping trial due to error: {e}")
                continue
            
        all_subjects[subj_id] = trials
        print(f"Loaded {len(trials)} trials for subject {subj_id}")
    
    print(f"Loaded data for {len(all_subjects)} subjects")
    return all_subjects

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate WAIC and LOO for NES models')
    parser.add_argument('--output-dir', type=str, default='waic_results',
                        help='Directory to save results')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated list of subject IDs to include')
    parser.add_argument('--draws', type=int, default=200,
                        help='Number of posterior draws per subject')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse subject IDs if provided
    subject_ids = args.subjects.split(',') if args.subjects else None
    
    # Define model configurations with exact paths
    model_files_and_params = {
        "6_param": {
            "path": Path(r"C:\Users\jesse\Hegemonikon Project\NES_Copilot\sbc_nes_6param_60SS_HybridPriors_nsf_60k\models\nes_npe_sims60000_template250_seed91.pt"),
            "params": ["v_norm", "a_0", "w_s_eff", "t_0", "alpha_gain", "beta_val"]
        },
        "4_param": {
            "path": Path(r"C:\Users\jesse\Hegemonikon Project\hegemonikon\sbc_nes_roberts_template250_v2\models\nes_npe_sims30000_template250_seed2031.pt"),
            "params": ["v_norm", "a_0", "w_s_eff", "t_0"]
        },
        "5_param_alpha": {
            "path": Path(r"C:\Users\jesse\Hegemonikon Project\hegemonikon\sbc_nes_roberts_5param_alpha_gain\models\nes_npe_sims30000_template250_seed2042.pt"),
            "params": ["v_norm", "a_0", "w_s_eff", "t_0", "alpha_gain"]
        }
    }

    # Load and prepare the data
    print("Loading and prepping all subject data...")
    roberts_data_path = Path(r"C:\\Users\\jesse\\Hegemonikon Project\\NES_Copilot\\Roberts_Framing_Data\\ftp_osf_data.csv")
    if not roberts_data_path.exists():
        print(f"Error: Data file not found at {roberts_data_path}")
        sys.exit(1)
        
    all_subjects_trial_data = load_and_prep_all_subject_data(
        roberts_data_path,
        subject_ids=subject_ids
    )
    print(f"Loaded data for {len(all_subjects_trial_data)} subjects.")
    
    if not all_subjects_trial_data:
        print("No subject data loaded. Exiting.")
        sys.exit(1)
    
    # Import MVNESAgent
    try:
        from nes_copilot.agent_mvnes import MVNESAgent
    except ImportError:
        print("Error: Could not import MVNESAgent. Make sure the module is in your PYTHONPATH.")
        sys.exit(1)
    
    # Process each model
    results_summary = {}
    for model_name, model_info in model_files_and_params.items():
        model_path = model_info["path"]
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            results_summary[model_name] = None
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"Parameters: {model_info['params']}")
        print(f"File: {model_path}")
        
        model_results = process_model_file(
            pt_file_path=model_path,
            model_param_names=model_info["params"],
            all_subjects_trial_data=all_subjects_trial_data,
            agent_class_to_use=MVNESAgent,
            n_posterior_draws=args.draws
        )
        
        if model_results:
            results_summary[model_name] = model_results
            
            # Save individual model results
            model_output = output_dir / f"{model_name}_results.json"
            with open(model_output, 'w') as f:
                json.dump({
                    'model': model_name,
                    'parameters': model_info['params'],
                    'waic': model_results.get('total_waic'),
                    'loo': model_results.get('total_loo')
                }, f, indent=2)
            print(f"Saved results to {model_output}")
    
    # Save summary of all models
    summary_output = output_dir / 'model_comparison_summary.json'
    summary = {
        model: {
            'waic': res.get('total_waic') if res else None,
            'loo': res.get('total_loo') if res else None,
            'n_subjects': len(res.get('waic_per_subject', {})) if res else 0
        }
        for model, res in results_summary.items()
    }
    
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'WAIC':>15} {'LOO':>15} {'N_Subjects':>12}")
    print("-"*60)
    
    for model, res in sorted(results_summary.items(), 
                            key=lambda x: x[1].get('total_waic', float('inf')) 
                                       if x[1] else float('inf')):
        if res:
            print(f"{model:<15} {res.get('total_waic', 'N/A'):>15.2f} "
                  f"{res.get('total_loo', 'N/A'):>15.2f} "
                  f"{len(res.get('waic_per_subject', {})):>12}")
        else:
            print(f"{model:<15} {'N/A':>15} {'N/A':>15} {'N/A':>12}")
    
    print(f"\nDetailed results saved to: {output_dir.absolute()}")