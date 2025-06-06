import os
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
import logging
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_log_likelihood_var(idata: az.InferenceData, possible_names: List[str]) -> Optional[str]:
    """Find the log-likelihood variable name from a list of possible names."""
    if not hasattr(idata, 'log_likelihood'):
        return None
    
    for name in possible_names:
        if name in idata.log_likelihood:
            return name
    
    # If no exact match, try case-insensitive search
    ll_vars = list(idata.log_likelihood.data_vars)
    for name in possible_names:
        for var_name in ll_vars:
            if name.lower() in var_name.lower():
                return var_name
    
    # If still not found, return the first log-likelihood variable
    if ll_vars:
        return ll_vars[0]
    
    return None


def debug_log_likelihood(idata: az.InferenceData, model_name: str):
    """Debug function to inspect log-likelihood structure"""
    print(f"\n=== {model_name} Log-likelihood Debug ===")
    
    if not hasattr(idata, 'log_likelihood'):
        print("No log_likelihood group found in InferenceData")
        return
    
    print("Available log_likelihood variables:", list(idata.log_likelihood.data_vars))
    
    for var_name, var_data in idata.log_likelihood.data_vars.items():
        print(f"\nVariable: {var_name}")
        print(f"Shape: {var_data.shape}")
        print(f"Dims: {var_data.dims}")
        
        # Safely print sample values based on dimensionality
        try:
            ndim = len(var_data.dims)
            if ndim == 2:
                print(f"Sample values (first 2x2):\n{var_data.values[:2, :2]}")
            elif ndim == 3:
                print(f"Sample values (first 2x2x2):\n{var_data.values[:2, :2, :2]}")
            else:
                print(f"Sample values (first 2 elements):\n{var_data.values.flat[:2]}")
        except Exception as e:
            print(f"Could not print sample values: {e}")

def ensure_3d_log_likelihood(ll_array, model_name: str):
    """Ensure log-likelihood array is 3D (chains × draws × trials)"""
    if ll_array is None:
        return None
        
    print(f"\nEnsuring 3D log-likelihood for {model_name}")
    print(f"Input shape: {ll_array.shape}, ndim: {ll_array.ndim}")
    
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

def compare_models_robust(nes_idata: az.InferenceData, hddm_idata: az.InferenceData, 
                         subject_id: str, output_dir: str) -> Dict[str, Any]:
    """
    Compare NES and HDDM models with robust error handling and debugging.
    """
    results = {'success': False}
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Debug log-likelihood structures
        debug_log_likelihood(nes_idata, "NES")
        debug_log_likelihood(hddm_idata, "HDDM")
        
        # Get log-likelihood variables
        nes_ll_var = get_log_likelihood_var(nes_idata, ['nes', 'y', 'log_likelihood'])
        hddm_ll_var = get_log_likelihood_var(hddm_idata, ['y', 'log_likelihood', 'hddm'])
        
        if nes_ll_var is None or hddm_ll_var is None:
            raise ValueError("Could not find log-likelihood variables in one or both models")
            
        print(f"Using NES log-likelihood variable: {nes_ll_var}")
        print(f"Using HDDM log-likelihood variable: {hddm_ll_var}")
        
        # Get log-likelihood arrays
        ll_nes = nes_idata.log_likelihood[nes_ll_var].values
        ll_hddm = hddm_idata.log_likelihood[hddm_ll_var].values
        
        # Ensure both log-likelihood arrays are 3D (chains × draws × trials)
        ll_nes = ensure_3d_log_likelihood(ll_nes, "NES")
        ll_hddm = ensure_3d_log_likelihood(ll_hddm, "HDDM")
        
        # Print shapes for debugging
        print(f"\nLog-likelihood shapes before standardization:")
        print(f"NES: {ll_nes.shape}")
        print(f"HDDM: {ll_hddm.shape}")
        
        # Standardize to the same number of draws (take minimum to be safe)
        min_draws = min(ll_nes.shape[1], ll_hddm.shape[1])
        if ll_nes.shape[1] != ll_hddm.shape[1]:
            print(f"Standardizing to {min_draws} draws per chain")
            ll_nes = ll_nes[:, :min_draws, :]
            ll_hddm = ll_hddm[:, :min_draws, :]
        
        # Handle different numbers of trials by trimming to the minimum
        min_trials = min(ll_nes.shape[2], ll_hddm.shape[2])
        if ll_nes.shape[2] != ll_hddm.shape[2]:
            print(f"Standardizing to {min_trials} trials")
            ll_nes = ll_nes[:, :, :min_trials]
            ll_hddm = ll_hddm[:, :, :min_trials]
        
        print(f"\nFinal log-likelihood shapes after standardization:")
        print(f"NES: {ll_nes.shape}")
        print(f"HDDM: {ll_hddm.shape}")
        
        # Create xarray Datasets for log-likelihoods
        coords = {
            'chain': np.arange(ll_nes.shape[0]),
            'draw': np.arange(ll_nes.shape[1]),
            'trial': np.arange(ll_nes.shape[2])
        }
        
        log_lik_nes_ds = xr.Dataset(
            data_vars={
                'nes': (['chain', 'draw', 'trial'], ll_nes)
            },
            coords=coords
        )
        
        log_lik_hddm_ds = xr.Dataset(
            data_vars={
                'hddm': (['chain', 'draw', 'trial'], ll_hddm)
            },
            coords=coords
        )
        
        # Create new InferenceData objects with standardized log-likelihoods
        idata_nes_std = az.InferenceData(log_likelihood=log_lik_nes_ds)
        idata_hddm_std = az.InferenceData(log_likelihood=log_lik_hddm_ds)
        
        # Initialize results dictionary to store all outputs
        results = {
            'subject_id': subject_id,
            'waic_comparison': None,
            'waic_error': None,
            'loo_comparison': None,
            'loo_error': None,
            'success': False
        }
        
        # Compare models using WAIC
        print("\nComparing models...\n")
        print("Calculating WAIC...")
        try:
            waic_comparison = az.compare(
                {"NES": idata_nes_std, "HDDM": idata_hddm_std},
                method='stacking',
                ic="waic",
                scale="deviance"
            )
            
            # Save WAIC comparison
            waic_file = os.path.join(output_dir, f"waic_comparison_subject_{subject_id}.csv")
            waic_comparison.to_csv(waic_file)
            print(f"WAIC comparison saved to {waic_file}")
            print("\n" + str(waic_comparison) + "\n")
            
            # Store successful WAIC results
            results['waic_comparison'] = waic_comparison
            results['success'] = True
            
            # Save WAIC pointwise values for each model
            try:
                waic_nes = az.waic(idata_nes_std, pointwise=True)
                waic_hddm = az.waic(idata_hddm_std, pointwise=True)
                
                waic_nes_file = os.path.join(output_dir, f"waic_nes_subject_{subject_id}.csv")
                pd.DataFrame({
                    'elpd_waic': waic_nes.waic_i.values,
                    'p_waic': waic_nes.p_waic_i.values if hasattr(waic_nes, 'p_waic_i') else [np.nan] * len(waic_nes.waic_i)
                }).to_csv(waic_nes_file)
                print(f"Saved NES WAIC pointwise values to {waic_nes_file}")
                
                waic_hddm_file = os.path.join(output_dir, f"waic_hddm_subject_{subject_id}.csv")
                pd.DataFrame({
                    'elpd_waic': waic_hddm.waic_i.values,
                    'p_waic': waic_hddm.p_waic_i.values if hasattr(waic_hddm, 'p_waic_i') else [np.nan] * len(waic_hddm.waic_i)
                }).to_csv(waic_hddm_file)
                print(f"Saved HDDM WAIC pointwise values to {waic_hddm_file}")
                
            except Exception as e:
                print(f"Warning: Could not save WAIC pointwise values: {str(e)}")
            
        except Exception as e:
            waic_error = f"WAIC comparison failed: {str(e)}"
            print(f"Warning: {waic_error}")
            results['waic_error'] = waic_error
        
        # Try LOO comparison (likely to fail without posterior group)
        print("\nCalculating LOO...")
        try:
            # Try to calculate LOO (will likely fail without posterior group)
            loo_comparison = az.loo_compare(
                [idata_nes_std, idata_hddm_std],
                scale="deviance"
            )
            
            # Save LOO comparison if successful
            loo_file = os.path.join(output_dir, f"loo_comparison_subject_{subject_id}.csv")
            loo_comparison.to_csv(loo_file)
            print(f"LOO comparison saved to {loo_file}")
            print("\n" + str(loo_comparison) + "\n")
            
            results['loo_comparison'] = loo_comparison
            results['success'] = True
            
        except Exception as e:
            loo_error = f"LOO comparison not available: {str(e)}"
            print(f"Note: {loo_error}")
            results['loo_error'] = loo_error
        
        # If we have at least one successful comparison, log success
        if results['waic_comparison'] is not None:
            print("\nModel comparison completed successfully with WAIC results!")
        else:
            print("\nModel comparison completed with warnings. Check the logs for details.")
        
        return results

    except Exception as e:
        error_msg = f"Error in model comparison: {str(e)}"
        logger.error(error_msg, exc_info=True)
        results['error'] = error_msg
    
    return results

if __name__ == "__main__":
    # Example usage (for testing)
    print("This is a module with comparison functions. Import and use the functions in your main script.")
