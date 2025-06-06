import os
import sys
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the current directory to the path so we can import our debug module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from debug_comparison import compare_models_robust

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = "results/model_comparison_empirical"
NPE_CHECKPOINT_PATH = "output/models/nes_npe_checkpoint.pt"
SUMMARY_STAT_KEYS_NES = ['rt_mean', 'rt_std', 'acc_mean', 'rt_10th_pctl', 'rt_90th_pctl']
NES_DYNAMIC_PARAM_NAMES = ['v_norm', 'a_0', 't_0', 'alpha_gain', 'beta_val']
FIXED_NES_PARAMS = {'logit_z0': 0.0, 'log_tau_norm': -0.7}
NES_DDM_BASE_CONFIG = {'w_s': 1.0, 'salience_input': 0.5, 'sv': 0.0, 'sz': 0.0, 'st': 0.0}

# Standardize sampling parameters
NUM_CHAINS = 2
NUM_SAMPLES = 1000  # Total samples per chain
BURN_IN = 500      # Burn-in samples for HDDM

# NES parameters
NES_NUM_SAMPLES = NUM_SAMPLES
NES_NUM_CHAINS = NUM_CHAINS

# HDDM parameters
HDDM_MODEL_CONFIG_MAIN = {
    'models': [],
    'include': ['v', 'a', 't'],
    'p_outlier': 0.05,
    'is_group_model': False
}
HDDM_NUM_SAMPLES = NUM_SAMPLES + BURN_IN  # HDDM includes burn-in in total samples
HDDM_BURN_IN = BURN_IN

def load_empirical_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load and prepare empirical data"""
    print(f"Loading empirical data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} trials from {df['subj_idx'].nunique()} subjects")
        return df
    except Exception as e:
        print(f"Error loading empirical data: {e}")
        return None

def fit_nes_model(subject_data: pd.DataFrame, npe_checkpoint_path: str, 
                  summary_stat_keys: List[str], nes_param_names: List[str], 
                  fixed_nes_params: Dict[str, float], nes_ddm_params_config: Dict[str, float],
                  num_samples: int = NES_NUM_SAMPLES, 
                  num_chains: int = NES_NUM_CHAINS) -> Optional[az.InferenceData]:
    """Fit NES model to subject data and return InferenceData"""
    try:
        print("\n--- Fitting NES model for subject data ---")
        print(f"Subject data shape: {subject_data.shape}")
        print(f"NPE checkpoint: {npe_checkpoint_path}")
        
        # Mock NES fitting - replace with actual NES fitting code
        n_trials = len(subject_data)
        n_samples_chain = num_samples // num_chains
        
        # Create mock posterior samples with standardized dimensions
        posterior_data = {}
        for param in NES_DYNAMIC_PARAM_NAMES:
            if param == 'v_norm':
                posterior_data[param] = np.random.normal(0.5, 0.2, (num_chains, n_samples_chain, 1))  # Add trial dimension
            elif param == 'a_0':
                posterior_data[param] = np.random.normal(1.5, 0.3, (num_chains, n_samples_chain, 1))
            elif param == 't_0':
                posterior_data[param] = np.random.normal(0.3, 0.1, (num_chains, n_samples_chain, 1))
            elif param == 'alpha_gain':
                posterior_data[param] = np.random.normal(0.5, 0.2, (num_chains, n_samples_chain, 1))
            elif param == 'beta_val':
                posterior_data[param] = np.random.normal(1.0, 0.1, (num_chains, n_samples_chain, 1))
        
        # Add fixed parameters
        for param, value in FIXED_NES_PARAMS.items():
            posterior_data[param] = np.full((num_chains, n_samples_chain, 1), value)  # Add trial dimension
        
        # Create InferenceData with proper dimensions
        idata = az.from_dict(
            posterior=posterior_data,
            coords={
                'chain': np.arange(num_chains), 
                'draw': np.arange(n_samples_chain),
                'trial': np.arange(1)  # Single trial dimension for parameters
            },
            dims={param: ["trial"] for param in NES_DYNAMIC_PARAM_NAMES}
        )
        
        # Add log-likelihood group with proper 3D shape (chains × draws × trials)
        n_trials_ll = max(n_trials, 10)  # Ensure at least 10 trials
        
        # Generate mock log-likelihood with correct shape (num_chains, n_samples_chain, n_trials_ll)
        log_lik = np.random.normal(-1, 0.5, (num_chains, n_samples_chain, n_trials_ll))
        
        # Create coords for the log-likelihood dataset
        coords = {
            'chain': np.arange(num_chains),
            'draw': np.arange(n_samples_chain),
            'trial': np.arange(n_trials_ll)
        }
        
        # Create log-likelihood dataset
        log_lik_ds = xr.Dataset(
            data_vars={
                'nes': (['chain', 'draw', 'trial'], log_lik)
            },
            coords=coords
        )
        
        # Add log-likelihood group to InferenceData
        idata.add_groups({
            'log_likelihood': log_lik_ds
        })
        
        print(f"NES fitting completed for subject {subject_data['subj_idx'].iloc[0]} with {n_samples_chain} samples per chain")
        return idata
    except Exception as e:
        logger.error(f"Error in NES model fitting: {e}", exc_info=True)
        return None

def fit_hddm_ext_model(subject_data_orig: pd.DataFrame, hddm_model_config: Dict[str, Any],
                       num_samples: int, burn_in: int) -> Optional[az.InferenceData]:
    """Fit HDDM model to subject data"""
    try:
        print(f"Original subject data shape: {subject_data_orig.shape}")
        
        # Make a copy to avoid modifying original data
        subject_data = subject_data_orig.copy()
        
        # Ensure response is 0/1
        if subject_data['response'].min() < 0 or subject_data['response'].max() > 1:
            print("Warning: Response values outside [0,1] range detected")
            subject_data['response'] = subject_data['response'].clip(0, 1).astype(int)
        
        # Check response distribution
        print("Response distribution:")
        print(subject_data['response'].value_counts())
        
        # Ensure RT is positive
        subject_data['rt'] = np.abs(subject_data['rt'])
        
        # Add required columns for HDDM
        subject_data['subj_idx'] = 0  # Single subject
        
        # Print model config for debugging
        print(f"HDDM model config being used: {hddm_model_config}")
        
        # Mock HDDM fitting - create mock posterior samples
        # Use the same number of chains as NES for consistency
        n_chains = 2  # Match NES model
        n_samples_chain = num_samples // n_chains
        
        # Create mock posterior samples with appropriate dimensions
        # Ensure dimensions match NES model: (chains, draws, trials)
        posterior_data = {
            'v': np.random.normal(2.0, 0.5, (n_chains, n_samples_chain, 1)),
            'a': np.random.normal(1.5, 0.3, (n_chains, n_samples_chain, 1)),
            't': np.random.normal(0.3, 0.1, (n_chains, n_samples_chain, 1)),
            'v_sd': np.random.gamma(1.0, 0.1, (n_chains, n_samples_chain, 1)),
            'a_sd': np.random.gamma(0.5, 0.1, (n_chains, n_samples_chain, 1)),
            't_sd': np.random.gamma(0.1, 0.05, (n_chains, n_samples_chain, 1))
        }
        
        # Create InferenceData with proper dimensions
        idata = az.from_dict(
            posterior=posterior_data,
            coords={
                'chain': np.arange(n_chains),
                'draw': np.arange(n_samples_chain),
                'trial': np.arange(1)  # Single trial dimension for parameters
            },
            dims={
                'v': ['trial'],
                'a': ['trial'],
                't': ['trial'],
                'v_sd': ['trial'],
                'a_sd': ['trial'],
                't_sd': ['trial']
            }
        )
        
        # Add log-likelihood group with consistent dimensions
        n_trials = len(subject_data)
        # Ensure log-likelihood has correct dimensions (chains, draws, trials)
        log_likelihood = np.random.normal(-0.5, 0.1, (n_chains, n_samples_chain, n_trials))
        
        # Create a dataset for log-likelihood with proper coordinates
        coords = {
            'chain': np.arange(n_chains),
            'draw': np.arange(n_samples_chain),
            'trial': np.arange(n_trials)
        }
        
        log_lik_ds = xr.Dataset(
            data_vars={
                'y': (['chain', 'draw', 'trial'], log_likelihood)
            },
            coords=coords
        )
        
        # Add log-likelihood group to InferenceData
        idata.add_groups({
            'log_likelihood': log_lik_ds
        })
        
        return idata
        
        print(f"HDDM fitting completed for subject {subject_data['subj_idx'].iloc[0]} with {effective_samples} samples after burn-in")
        return idata
    except Exception as e:
        logger.error(f"Error in HDDM model fitting: {e}", exc_info=True)
        return None

def ensure_3d_log_likelihood(ll_array, model_name: str):
    """Ensure log-likelihood array is 3D (chains × draws × trials)"""
    if ll_array is None:
        return None
        
    print(f"\nEnsuring 3D log-likelihood for {model_name}")
    print(f"Input shape: {ll_array.shape}, ndim: {ll_array.ndim}")
    
    # If it's an xarray DataArray, convert to numpy array
    if hasattr(ll_array, 'values'):
        ll_array = ll_array.values
    
    # If already 3D, return as is
    if ll_array.ndim == 3:
        print("Already 3D, returning as is")
        return ll_array
    
    # If 2D, add chain dimension
    if ll_array.ndim == 2:
        print("2D array, adding chain dimension")
        return ll_array[np.newaxis, ...]  # Add chain dimension at the beginning
    
    # If 1D, assume it's for a single chain and trial
    if ll_array.ndim == 1:
        print("1D array, reshaping to (1, 1, n_trials)")
        return ll_array.reshape((1, 1, -1))  # (1, 1, n_trials)
    
    # If 4D, take mean over extra dimension (e.g., for multiple parameters)
    if ll_array.ndim == 4:
        print("4D array, taking mean over last dimension")
        return ll_array.mean(axis=-1)
    
    raise ValueError(f"Cannot handle array with shape {ll_array.shape}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load empirical data
    data_path = "output/data_prep/empirical_data.csv"
    all_data_loaded = load_empirical_data(data_path)
    
    if all_data_loaded is None:
        print("Error: Could not load empirical data. Exiting.")
        return
    
    # Print data summary
    print("\n--- Data Summary ---")
    print(f"Data shape: {all_data_loaded.shape}")
    if 'subj_idx' in all_data_loaded.columns:
        num_subjects = all_data_loaded['subj_idx'].nunique()
        print(f"Number of unique subjects: {num_subjects}")
    else:
        print("No 'subj_idx' column found, assuming single subject or pre-grouped data.")
        all_data_loaded['subj_idx'] = 0
    print(f"Total trials: {len(all_data_loaded)}")
    print("\nFirst few rows of data:")
    print(all_data_loaded.head())
    print("-" * 50 + "\n")
    
    # Get unique subject IDs (limit to first 2 for testing)
    unique_subject_ids = all_data_loaded['subj_idx'].unique()[:2]
    print(f"Processing first {len(unique_subject_ids)} subjects for testing")
    
    all_subject_nes_idata = {}
    all_subject_hddm_idata = {}
    comparison_results = {}
    
    for s_id in unique_subject_ids[:2]:  # Process first 2 subjects for testing
        print("\n" + "="*50)
        print(f"PROCESSING SUBJECT {s_id}")
        print("="*50)
        
        # Get subject data
        subject_data = all_data_loaded[all_data_loaded['subj_idx'] == s_id].copy()
        if len(subject_data) == 0:
            print(f"No data for subject {s_id}, skipping...")
            continue
            
        print(f"Subject {s_id} data shape: {subject_data.shape}")
        
        # Fit NES model
        print("\n--- Fitting NES Model ---")
        nes_idata = fit_nes_model(
            subject_data=subject_data,
            npe_checkpoint_path=NPE_CHECKPOINT_PATH,
            summary_stat_keys=SUMMARY_STAT_KEYS_NES,
            nes_param_names=NES_DYNAMIC_PARAM_NAMES,
            fixed_nes_params=FIXED_NES_PARAMS,
            nes_ddm_params_config=NES_DDM_BASE_CONFIG,
            num_samples=NES_NUM_SAMPLES,
            num_chains=NES_NUM_CHAINS
        )
        
        if nes_idata is None:
            print(f"NES fitting failed for subject {s_id}")
            continue
            
        all_subject_nes_idata[s_id] = nes_idata
        
        # Fit HDDM model
        print("\n--- Fitting HDDM Model ---")
        hddm_idata = fit_hddm_ext_model(
            subject_data_orig=subject_data,
            hddm_model_config=HDDM_MODEL_CONFIG_MAIN,
            num_samples=HDDM_NUM_SAMPLES,
            burn_in=HDDM_BURN_IN
        )
        
        if hddm_idata is None:
            print(f"HDDM fitting failed for subject {s_id}")
            continue
            
        all_subject_hddm_idata[s_id] = hddm_idata
        
        # Compare models using our robust comparison function
        print("\n--- Comparing Models ---")
        try:
            comp_result = compare_models_robust(
                nes_idata=nes_idata,
                hddm_idata=hddm_idata,
                subject_id=str(s_id),
                output_dir=OUTPUT_DIR
            )
            if comp_result is not None:
                comparison_results[s_id] = comp_result
                print(f"Model comparison successful for subject {s_id}")
            else:
                print(f"Model comparison failed for subject {s_id}")
            
            # Print comparison results
            if comp_result.get('success', False):
                print("\nModel comparison successful!")
                if 'waic' in comp_result:
                    print("WAIC Comparison:")
                    print(pd.DataFrame(comp_result['waic']))
                if 'loo' in comp_result:
                    print("\nLOO Comparison:")
                    print(pd.DataFrame(comp_result['loo']))
            else:
                print("\nModel comparison failed:", comp_result.get('error', 'Unknown error'))
                
        except Exception as e:
            error_msg = f"Error in model comparison: {str(e)}"
            print(error_msg)
            comparison_results[s_id] = {'success': False, 'error': error_msg}
    
    # Print final summary
    print("\n" + "="*50)
    print("COMPLETED MODEL FITTING AND COMPARISON")
    print("="*50)
    
    print(f"\nProcessed {len(unique_subject_ids)} subjects")
    print(f"Successful NES fits: {len(all_subject_nes_idata)}")
    print(f"Successful HDDM fits: {len(all_subject_hddm_idata)}")
    
    successful_comparisons = sum(1 for r in comparison_results.values() if r.get('success', False))
    print(f"\nSuccessful model comparisons: {successful_comparisons}")
    
    if successful_comparisons > 0:
        print("\nComparison results saved to:")
        for s_id, result in comparison_results.items():
            if result.get('success', False):
                print(f"  Subject {s_id}:")
                if 'waic_path' in result:
                    print(f"    WAIC: {result.get('waic_path', 'N/A')}")
                if 'loo_path' in result:
                    print(f"    LOO:  {result.get('loo_path', 'N/A')}")
    
    print("\nAll done!")

if __name__ == "__main__":
    main()
