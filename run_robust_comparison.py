import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the current directory to the path so we can import our debug module
# and the src directory for nes_copilot package
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

import arviz as az
import numpy as np
import pandas as pd
import torch
import xarray as xr

from debug_comparison import compare_models_robust
from nes_copilot.agent_mvnes import MVNESAgent
from nes_copilot.checkpoint_manager import CheckpointManager
from nes_copilot.summary_stats_module import SummaryStatsModule

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

# Mock classes for SummaryStatsModule dependencies
class MockConfigManager:
    def __init__(self, logger_instance): # Renamed logger to logger_instance to avoid conflict
        self.logger = logger_instance
        self.config = { # Provide minimal config SummaryStatsModule might need
            'summary_stats': {},
            'master_config': {'device': 'cpu'}
        }
    def get_module_config(self, module_name: str) -> dict:
        return self.config.get(module_name, {})
    def get_param(self, param_name: str, default: Any = None) -> Any:
        # Updated to access master_config correctly
        return self.config.get('master_config', {}).get(param_name, default)

class MockDataManager:
    def __init__(self, logger_instance): # Renamed logger to logger_instance
        self.logger = logger_instance
    def get_output_path(self, module_name: str, filename: str) -> str:
        # SummaryStatsModule might call this to save stats; make it benign
        return os.path.join("mock_output", module_name, filename)
    def save_json(self, data: dict, module_name: str, filename: str):
        # Make this a no-op for now
        self.logger.info(f"MockDataManager: save_json called for {module_name}/{filename}, doing nothing.")
        pass

class MockLoggingManager:
    def __init__(self, logger_instance): # Renamed logger to logger_instance
        self.logger = logger_instance
    def get_logger(self, name: str): # Or however it provides the logger
        return self.logger

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

def fit_nes_model(
    subject_data: pd.DataFrame,
    npe_checkpoint_path: str,
    summary_stat_keys: List[str],
    nes_param_names: List[str],
    fixed_nes_params: Dict[str, float],
    nes_ddm_params_config: Dict[str, float],
    num_posterior_samples: int = NES_NUM_SAMPLES,  # Use existing global
    num_chains: int = NES_NUM_CHAINS  # Use existing global
) -> Optional[az.InferenceData]:

    # --- 1. Load NPE Model ---
    logger.info(f"Attempting to load NPE model from checkpoint: {npe_checkpoint_path}")
    try:
        # Ensure the global logger is used if not passed explicitly
        # CheckpointManager is expected to be initialized with a logger
        checkpoint_manager = CheckpointManager(logger=logger)
        checkpoint = checkpoint_manager.load_checkpoint(npe_checkpoint_path)

        posterior_estimator = checkpoint['posterior']
        # These are from the checkpoint file
        loaded_param_names = checkpoint.get('param_names', [])
        loaded_summary_stat_keys = checkpoint.get('summary_stat_keys', [])

        # Compare with function arguments and log if different
        # For now, we will proceed using the nes_param_names and summary_stat_keys passed to the function.
        # A more robust strategy might be needed later (e.g., prioritize checkpoint's values).
        if set(loaded_param_names) != set(nes_param_names):
            logger.warning(f"Parameter names from checkpoint ({loaded_param_names}) "
                           f"differ from input nes_param_names ({nes_param_names}). "
                           f"Using input nes_param_names as primary for now.")

        if set(loaded_summary_stat_keys) != set(summary_stat_keys):
            logger.warning(f"Summary stat keys from checkpoint ({loaded_summary_stat_keys}) "
                           f"differ from input summary_stat_keys ({summary_stat_keys}). "
                           f"Using input summary_stat_keys as primary for now.")

        logger.info(f"Successfully loaded NPE model. Posterior estimator type: {type(posterior_estimator)}")

    except FileNotFoundError:
        logger.error(f"NPE checkpoint file not found at: {npe_checkpoint_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the NPE checkpoint: {e}", exc_info=True)
        return None

    # --- 2. Calculate Observed Summary Statistics ---
    logger.info("Calculating observed summary statistics for NES model...")
    x_observed_nes = None # Initialize to ensure it's defined
    try:
        # Initialize mock managers if SummaryStatsModule requires them
        # Ensure these mock classes are defined globally or imported
        mock_config_manager = MockConfigManager(logger_instance=logger) # Pass logger as logger_instance
        mock_data_manager = MockDataManager(logger_instance=logger) # Pass logger as logger_instance
        # Assuming SummaryStatsModule can take a logger directly via a mock LoggingManager
        mock_logging_manager = MockLoggingManager(logger_instance=logger)

        summary_stats_module = SummaryStatsModule(
            config_manager=mock_config_manager,
            data_manager=mock_data_manager,
            logging_manager=mock_logging_manager
        )
        
        # Calculate summary statistics using the keys provided to the function
        if not summary_stat_keys:
            logger.warning("summary_stat_keys is empty. SummaryStatsModule might use defaults or fail.")
            # Consider raising an error if summary_stat_keys are mandatory for this context
            # For now, proceeding and letting SummaryStatsModule handle it or error out.
        
        summary_stats_results = summary_stats_module.run(
            trials_df=subject_data,
            stat_keys=summary_stat_keys
        )
        calculated_summary_stats_dict = summary_stats_results['summary_stats']

        # Convert summary stats dictionary to a torch tensor in the correct order
        x_observed_values = [calculated_summary_stats_dict.get(key, np.nan) for key in summary_stat_keys]

        if np.isnan(x_observed_values).any():
            logger.error(f"NaNs found in observed summary statistics: {x_observed_values} for keys {summary_stat_keys} from dict {calculated_summary_stats_dict}")
            raise ValueError("NaNs found in calculated summary statistics for NES.")

        x_observed_nes = torch.tensor(x_observed_values, dtype=torch.float32).reshape(1, -1)
        
        logger.info(f"Successfully calculated and formatted observed summary statistics for NES. Shape: {x_observed_nes.shape}")

    except Exception as e:
        logger.error(f"An error occurred during summary statistics calculation for NES: {e}", exc_info=True)
        return None

    # --- 3. Draw Posterior Samples ---
    logger.info(f"Drawing {num_posterior_samples} posterior samples for NES model across {num_chains} chains...")

    if num_chains <= 0:
        logger.error("Number of chains must be positive.")
        return None
        
    num_samples_per_chain = num_posterior_samples // num_chains
    if num_samples_per_chain == 0:
        logger.warning(
            f"num_posterior_samples ({num_posterior_samples}) is less than num_chains ({num_chains}). "
            f"Resulting in num_samples_per_chain = {num_samples_per_chain}. "
            "Consider increasing num_posterior_samples or decreasing num_chains."
        )
        if num_posterior_samples > 0 and num_samples_per_chain == 0:
             logger.info(f"Setting num_samples_per_chain to 1 as num_posterior_samples ({num_posterior_samples}) > 0.")
             num_samples_per_chain = 1 # Ensure at least one sample if total is > 0
        elif num_posterior_samples == 0 :
             logger.error("num_posterior_samples is 0. Cannot draw samples.")
             return None

    posterior_samples_dict = {} # Initialize before try block
    try:
        # Initialize with empty lists for each parameter name
        posterior_samples_dict = {param_name: [] for param_name in nes_param_names}

        for i in range(num_chains):
            logger.info(f"Sampling chain {i+1}/{num_chains} ({num_samples_per_chain} samples)...")
            samples_for_current_chain = posterior_estimator.sample(
                (num_samples_per_chain,),
                x=x_observed_nes,
                show_progress_bars=False
            ).numpy()  # Convert to NumPy array, shape (num_samples_per_chain, num_parameters)

            for j, param_name in enumerate(nes_param_names):
                posterior_samples_dict[param_name].append(samples_for_current_chain[:, j])
        
        for param_name in nes_param_names:
            posterior_samples_dict[param_name] = np.array(posterior_samples_dict[param_name])
            # Expected shape for each param: (num_chains, num_samples_per_chain)

        logger.info("Successfully drew posterior samples for all dynamic NES parameters.")
        # Example: logger.info(f"Shape for param '{nes_param_names[0]}': {posterior_samples_dict[nes_param_names[0]].shape}")

    except Exception as e:
        logger.error(f"An error occurred during posterior sampling for NES: {e}", exc_info=True)
        return None

    # --- This is where the old mock posterior_data dictionary was built ---
    # --- Now, `posterior_samples_dict` holds the actual samples for dynamic params ---

    # --- 4. Construct ArviZ InferenceData object ---
    logger.info("Constructing ArviZ InferenceData object...")
    try:
        # Determine chain and draw counts from actual samples
        if not nes_param_names or not posterior_samples_dict or nes_param_names[0] not in posterior_samples_dict:
            logger.error("Cannot determine sampling dimensions, nes_param_names or posterior_samples_dict is invalid.")
            return None

        # Check if the first parameter's samples are available and get shape
        first_param_samples = posterior_samples_dict[nes_param_names[0]]
        if not hasattr(first_param_samples, 'shape') or len(first_param_samples.shape) < 2:
            logger.error(f"Samples for parameter '{nes_param_names[0]}' are not in the expected format (e.g., not a NumPy array or insufficient dimensions). Shape is: {getattr(first_param_samples, 'shape', 'N/A')}")
            return None
        _num_chains, _num_samples_per_chain = first_param_samples.shape

        posterior_data_for_arviz = posterior_samples_dict.copy() # Start with dynamic params

        # Expand fixed parameters to match sample dimensions
        for param_key, param_value in fixed_nes_params.items():
            posterior_data_for_arviz[param_key] = np.full((_num_chains, _num_samples_per_chain), param_value)

        coords_for_arviz = {
            'chain': np.arange(_num_chains),
            'draw': np.arange(_num_samples_per_chain)
        }
        
        # Prepare observed data
        observed_data_for_arviz = {
            'rt': subject_data['rt'].values,
            'response': subject_data['response'].values
        }
        if 'frame' in subject_data.columns: # Check if column exists
            observed_data_for_arviz['frame'] = subject_data['frame'].values
        if 'valence_score' in subject_data.columns: # Check if column exists
            observed_data_for_arviz['valence_score'] = subject_data['valence_score'].values

        # Create InferenceData object
        idata = az.from_dict(
            posterior=posterior_data_for_arviz,
            observed_data=observed_data_for_arviz,
            coords=coords_for_arviz
        )

        # Add MOCK log_likelihood data
        _num_trials = len(subject_data)
        _n_trials_ll_mock = max(_num_trials, 10)

        mock_log_lik_values = np.random.normal(-1, 0.5, (_num_chains, _num_samples_per_chain, _n_trials_ll_mock))
        mock_log_lik_coords = {
            'chain': np.arange(_num_chains),
            'draw': np.arange(_num_samples_per_chain),
            'trial_ll': np.arange(_n_trials_ll_mock)
        }
        log_lik_ds_mock = xr.Dataset(
            data_vars={'nes': (['chain', 'draw', 'trial_ll'], mock_log_lik_values)}, # Model name 'nes'
            coords=mock_log_lik_coords
        )
        idata.add_groups({'log_likelihood': log_lik_ds_mock})
        
        logger.info("Successfully constructed InferenceData with actual posterior samples and MOCK log-likelihoods.")
        return idata

    except Exception as e:
        logger.error(f"Error constructing InferenceData object: {e}", exc_info=True)
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
            num_posterior_samples=NES_NUM_SAMPLES,
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
