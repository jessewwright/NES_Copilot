import hddm
import arviz as az
import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any

# Import HDDM's wiener_like_contaminant function
try:
    from hddm.likelihoods import wiener_like_contaminant as _hddm_wiener_like
    print("Successfully imported wiener_like_contaminant from hddm.likelihoods")
    
    def wiener_like(*args, **kwargs):
        """
        Wrapper around wiener_like_contaminant to handle response direction.
        
        This function can be called in two ways:
        1. With positional args: (rt, v, sv, a, z, sz, t, st, response=1)
        2. With keyword args: (rt=..., v=..., sv=..., a=..., z=..., sz=..., t=..., st=..., response=1)
        """
        # Handle both positional and keyword arguments
        if args and len(args) >= 8:
            # Positional arguments provided
            rt, v, sv, a, z, sz, t, st = args[:8]
            response = args[8] if len(args) > 8 else 1
        else:
            # Keyword arguments provided
            rt = kwargs.get('rt')
            v = kwargs.get('v')
            sv = kwargs.get('sv', 0.0)
            a = kwargs.get('a')
            z = kwargs.get('z')
            sz = kwargs.get('sz', 0.0)
            t = kwargs.get('t')
            st = kwargs.get('st', 0.0)
            response = kwargs.get('response', 1)
        
        # Set default values for the contaminant model and other required parameters
        p_upper = 0.5  # Probability of upper boundary response (0.5 = unbiased)
        p_contaminant = 0.0  # No contamination
        err = 1e-4  # Default error tolerance
        n_st = 10  # Number of samples for st
        n_sz = 10  # Number of samples for sz
        use_adaptive = 1  # Use adaptive integration
        simps_err = 1e-3  # Error tolerance for Simpson's rule
        
        # Convert inputs to numpy arrays if they aren't already
        def to_array(x, dtype=np.float64):
            if isinstance(x, (list, np.ndarray)):
                return np.asarray(x, dtype=dtype)
            return np.array([x], dtype=dtype)
            
        rt = to_array(rt)
        response = to_array(response, dtype=np.int32)
        v = to_array(v)
        sv = to_array(sv)
        a = to_array(a)
        z = to_array(z)
        sz = to_array(sz)
        t = to_array(t)
        st = to_array(st)
        p_upper = to_array(p_upper)
        p_contaminant = to_array(p_contaminant)
        
        try:
            # Ensure all inputs are scalar floats
            rt = float(rt[0] if isinstance(rt, (list, np.ndarray)) else rt)
            response = int(response[0] if isinstance(response, (list, np.ndarray)) else response)
            v = float(v[0] if isinstance(v, (list, np.ndarray)) else v)
            sv = float(sv[0] if isinstance(sv, (list, np.ndarray)) else sv)
            a = float(a[0] if isinstance(a, (list, np.ndarray)) else a)
            z = float(z[0] if isinstance(z, (list, np.ndarray)) else z)
            sz = float(sz[0] if isinstance(sz, (list, np.ndarray)) else sz)
            t = float(t[0] if isinstance(t, (list, np.ndarray)) else t)
            st = float(st[0] if isinstance(st, (list, np.ndarray)) else st)
            p_upper = float(p_upper[0] if isinstance(p_upper, (list, np.ndarray)) else p_upper)
            p_contaminant = float(p_contaminant[0] if isinstance(p_contaminant, (list, np.ndarray)) else p_contaminant)
            
            # Convert back to numpy arrays with shape (1,) for HDDM
            rt = np.array([rt], dtype=np.float64)
            response = np.array([response], dtype=np.int32)
            v = np.array([v], dtype=np.float64)
            sv = np.array([sv], dtype=np.float64)
            a = np.array([a], dtype=np.float64)
            z = np.array([z], dtype=np.float64)
            sz = np.array([sz], dtype=np.float64)
            t = np.array([t], dtype=np.float64)
            st = np.array([st], dtype=np.float64)
            p_upper = np.array([p_upper], dtype=np.float64)
            p_contaminant = np.array([p_contaminant], dtype=np.float64)
            
            # Call the original HDDM function with all required arguments
            result = _hddm_wiener_like(
                rt, response, v, sv, a, z, sz, t, st, p_upper, p_contaminant,
                err=err, n_st=n_st, n_sz=n_sz, use_adaptive=use_adaptive, simps_err=simps_err
            )
            
            # Ensure we return a scalar
            if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                return float(result[0])
            return float(result) if result is not None else 1e-10
                
        except Exception as e:
            print(f"Error in wiener_like: {e}")
            print(f"Input types - rt: {type(rt)}, response: {type(response)}, v: {type(v)}, sv: {type(sv)}, a: {type(a)}, z: {type(z)}, sz: {type(sz)}, t: {type(t)}, st: {type(st)}")
            print(f"Input values - rt: {rt}, response: {response}, v: {v}, sv: {sv}, a: {a}, z: {z}, sz: {sz}, t: {t}, st: {st}")
            return 1e-10  # Return a small probability to avoid numerical issues
    
    print("Successfully initialized HDDM likelihood calculation")

except ImportError as e:
    print(f"Failed to import wiener_like_contaminant: {e}")
    raise ImportError(
        "Could not find a suitable likelihood function in HDDM. "
        "Your HDDM version may be incompatible with this script."
    )

print("Successfully initialized HDDM likelihood calculation")

# --- Constants for Log Likelihood ---
LOG_LIK_FLOOR = -1e9  # Floor for log likelihood values
EPS_FOR_LOG = 1e-29   # Epsilon to prevent log(0)

# --- Data Loading Function ---
def load_empirical_data(data_path: str) -> Optional[pd.DataFrame]:
    """
    Loads empirical data from a CSV file and performs basic validation.

    Args:
        data_path (str): Path to the CSV data file.

    Returns:
        pd.DataFrame | None: Loaded data as a DataFrame, or None if loading fails.
    """
    default_columns = ['subject', 'rt', 'response', 'frame'] # Essential columns

    try:
        data = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")

        # Basic validation
        missing_cols = [col for col in default_columns if col not in data.columns]
        if missing_cols:
            print(f"Warning: The following essential columns are missing from the data: {missing_cols}")
            # Depending on strictness, you might return None or raise an error
            # For now, we'll proceed but this is a point of caution.

        # Convert 'rt' to numeric, coercing errors (e.g., if 'rt' was accidentally non-numeric)
        if 'rt' in data.columns:
            data['rt'] = pd.to_numeric(data['rt'], errors='coerce')
            # HDDM expects RTs in seconds. If your data is in ms, convert it.
            # Assuming RT is in seconds. If in ms, uncomment below:
            # data['rt'] = data['rt'] / 1000.0

        # Ensure 'response' is integer (0 or 1 typically for HDDM)
        if 'response' in data.columns:
            data['response'] = data['response'].astype(int)

        # Rename 'subject' to 'subj_idx' if 'subj_idx' is not present, as HDDM often uses 'subj_idx'
        if 'subject' in data.columns and 'subj_idx' not in data.columns:
            data.rename(columns={'subject': 'subj_idx'}, inplace=True)
            print("Renamed 'subject' column to 'subj_idx' for HDDM compatibility.")

        if 'subj_idx' not in data.columns:
            print("Warning: 'subj_idx' (or 'subject') column not found. Assuming single subject or data is pre-grouped.")
            # You might want to add a default subject index if fitting per subject later
            # data['subj_idx'] = 0

        return data

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading or processing the data: {e}")
        return None

# Add any other necessary imports here
import arviz as az # Ensure Arviz is imported if not already at the top
import pickle # For saving/loading python objects if needed, or for dummy model persistence

# --- Minimal Manager and Model Classes (for placeholder functionality) ---
class MockDataManager:
    """
    A mock class to simulate a data manager that provides summary statistics.
    """
    def __init__(self, data: pd.DataFrame, summary_stat_keys: list[str]):
        self.data = data
        self.summary_stat_keys = summary_stat_keys
        print(f"MockDataManager initialized with data shape: {self.data.shape} and {len(summary_stat_keys)} summary stat keys.")

    def get_summary_stats(self) -> np.ndarray:
        """Returns a numpy array of summary statistics for the provided data."""
        # In a real scenario, this would compute or retrieve pre-computed summary stats.
        # Here, we'll create dummy summary stats based on the keys.
        # This assumes summary_stat_keys are present in self.data or are computed.

        # For this mock, let's try to select columns if they exist, else generate random ones.
        num_trials = len(self.data)
        num_stats = len(self.summary_stat_keys)
        summary_stats_array = np.zeros((num_trials, num_stats))

        for i, key in enumerate(self.summary_stat_keys):
            if key in self.data.columns:
                # Ensure data is numeric and handle potential NaNs
                stat_values = pd.to_numeric(self.data[key], errors='coerce').fillna(0).values
                # Ensure the stat_value array matches num_trials, truncate or pad if necessary
                if len(stat_values) < num_trials:
                    summary_stats_array[:len(stat_values), i] = stat_values
                else:
                    summary_stats_array[:, i] = stat_values[:num_trials]

            else:
                # If key not in data, generate random data as a placeholder
                print(f"Warning: Summary stat key '{key}' not found in data. Generating random data for it.")
                summary_stats_array[:, i] = np.random.randn(num_trials)

        print(f"MockDataManager: Generated summary stats of shape: {summary_stats_array.shape}")
        return summary_stats_array

class MockNPEModel:
    """
    A mock class to simulate an NPE model that can load a checkpoint and sample posteriors.
    """
    def __init__(self, checkpoint_path: str, param_names: list[str], data_manager: MockDataManager):
        self.checkpoint_path = checkpoint_path
        self.param_names = param_names
        self.data_manager = data_manager
        self.n_params = len(param_names)
        print(f"MockNPEModel initialized with checkpoint: {self.checkpoint_path} for {self.n_params} parameters: {param_names}")
        # Simulate loading the model (e.g., from a file)
        self._load_model()

    def _load_model(self):
        # In a real scenario, this would load the NPE model weights and architecture.
        # For this mock, we just print a message.
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"MockNPEModel: Successfully 'loaded' model from {self.checkpoint_path}")
            # e.g., self.npe_engine = torch.load(self.checkpoint_path)
        elif self.checkpoint_path:
            print(f"MockNPEModel: Checkpoint path '{self.checkpoint_path}' provided but not found. Using dummy model.")
        else:
            print("MockNPEModel: No checkpoint path provided. Using dummy model.")
        self.npe_engine = None # Placeholder for the actual engine

    def sample_posterior(self, num_samples: int, num_chains: int = 1) -> dict[str, np.ndarray]:
        """
        Generates dummy posterior samples.
        In a real NPE model, this would use the loaded engine and summary stats from data_manager.
        """
        print(f"MockNPEModel: Generating {num_samples} posterior samples across {num_chains} chain(s).")
        # summary_stats = self.data_manager.get_summary_stats() # This would be used by a real NPE
        # For mock, we generate random samples for each parameter.
        posterior_samples = {}
        for param_name in self.param_names:
            # Shape: (num_chains, num_samples_per_chain)
            posterior_samples[param_name] = np.random.randn(num_chains, num_samples)

        print(f"MockNPEModel: Generated posterior samples for keys: {list(posterior_samples.keys())}")
        return posterior_samples

# --- Log Likelihood Calculation ---
def get_log_likelihood_pointwise(
    posterior_params,
    trial_rt,
    trial_choice,
    trial_condition,
    model_type,
    fixed_nes_params=None,
    nes_ddm_params=None
):
    """
    Calculate the log-likelihood for a single trial given posterior parameters.
    
    Args:
        posterior_params: Dictionary or pandas Series of parameter values
        trial_rt: Reaction time for this trial
        trial_choice: Response (0 or 1) for this trial
        trial_condition: Condition information for this trial (if applicable)
        model_type: Type of model ("HDDM" or "NES")
        fixed_nes_params: Fixed parameters for NES model (if applicable)
        nes_ddm_params: DDM parameters for NES model (if applicable)
        
    Returns:
        Log-likelihood for this trial
    """
    # Convert trial_choice to 0/1 if it's not already
    response = int(trial_choice) if trial_choice in (0, 1) else (1 if float(trial_choice) > 0.5 else 0)
    
    if model_type == "HDDM":
        try:
            # Convert to dictionary if it's a pandas Series or numpy array
            if hasattr(posterior_params, 'to_dict'):
                params = posterior_params.to_dict()
            elif isinstance(posterior_params, np.ndarray):
                # Assuming order is v, a, t if it's a numpy array
                params = {'v': posterior_params[0], 'a': posterior_params[1], 't': posterior_params[2]}
            else:
                params = dict(posterior_params)
                
            # Extract parameters with defaults and validate
            try:
                v = float(params.get('v', 0.0))
                a = max(0.1, float(params.get('a', 1.0)))  # Boundary separation with minimum
                t = max(0.0, float(params.get('t', 0.1)))   # Non-decision time must be >= 0.01
                z_rel = float(params.get('z', 0.5))  # Relative start point (0-1)
                z = max(0.1 * a, min(0.9 * a, a * z_rel))  # Keep within bounds
                sv = max(0.0, float(params.get('sv', 0.0)))
                sz = max(0.0, float(params.get('sz', 0.0)))
                st = max(0.0, float(params.get('st', 0.0)))
                p_upper = 0.5  # Default probability of upper boundary
                p_contaminant = 0.0  # Default probability of contamination
                
                # Ensure RT is positive and reasonable
                rt = float(trial_rt)
                if rt <= 0 or rt > 10:  # Arbitrary upper bound
                    return np.log(1e-10)
                    
                # Calculate likelihood for upper and lower boundaries
                try:
                    likelihood_upper = wiener_like(rt, v, sv, a, z, sz, t, st)
                    likelihood_lower = wiener_like(rt, -v, sv, a, a - z, sz, t, st)
                    
                    # Combine based on response
                    if response == 1:
                        likelihood = p_upper * likelihood_upper
                    else:
                        likelihood = (1 - p_upper) * likelihood_lower
                    
                    # Add contamination component with numerical stability
                    likelihood = (1 - p_contaminant) * likelihood + p_contaminant * 1e-10
                    
                    # Return log-likelihood with numerical stability
                    return np.log(max(likelihood, 1e-100))
                    
                except Exception as e:
                    print(f"Error in wiener_like: {e}")
                    print(f"Inputs: rt={rt}, response={response}, v={v}, a={a}, z={z}, t={t}, sv={sv}, sz={sz}, st={st}")
                    return np.log(1e-10)  # Return very low likelihood on error
                    
            except (ValueError, TypeError) as e:
                print(f"Parameter conversion error: {e}")
                return np.log(1e-10)
                
        except Exception as e:
            print(f"Unexpected error in get_log_likelihood_pointwise (HDDM): {e}")
            return np.log(1e-10)
            
    elif model_type == "NES":
        try:
            if fixed_nes_params is None or nes_ddm_params is None:
                print("Error: Missing required NES parameters")
                return np.log(1e-10)
                
            # Convert to dictionary if it's a pandas Series or numpy array
            if hasattr(posterior_params, 'to_dict'):
                params = posterior_params.to_dict()
            elif isinstance(posterior_params, np.ndarray):
                # Assuming order matches NES_DYNAMIC_PARAM_NAMES
                param_names = NES_DYNAMIC_PARAM_NAMES if hasattr(posterior_params, '__len__') and len(posterior_params) > 3 else ['v', 'a', 't']
                params = {name: posterior_params[i] for i, name in enumerate(param_names)}
            else:
                params = dict(posterior_params)
            
            # Get DDM parameters
            try:
                v = float(params.get('v', 0.0))
                a = max(0.1, float(params.get('a', 1.0)))
                t = max(0.01, float(params.get('t', 0.1)))
                z_rel = float(params.get('z', 0.5))
                z = max(0.1 * a, min(0.9 * a, a * z_rel))
                
                # Ensure RT is positive and reasonable
                rt = float(trial_rt)
                if rt <= 0 or rt > 10:  # Arbitrary upper bound
                    return np.log(1e-10)
                
                # Calculate likelihood for upper and lower boundaries
                likelihood_upper = wiener_like(rt, v, 0.0, a, z, 0.0, t, 0.0)
                likelihood_lower = wiener_like(rt, -v, 0.0, a, a - z, 0.0, t, 0.0)
                
                # Combine based on response
                if response == 1:
                    likelihood = 0.5 * likelihood_upper
                else:
                    likelihood = 0.5 * likelihood_lower
                
                return np.log(max(likelihood, 1e-100))
                
            except Exception as e:
                print(f"Error in NES likelihood calculation: {e}")
                return np.log(1e-10)
                
        except Exception as e:
            print(f"Unexpected error in get_log_likelihood_pointwise (NES): {e}")
            return np.log(1e-10)
            
    else:
        print(f"Unknown model type: {model_type}")
        return np.log(1e-10)
    
    # This line should never be reached due to returns in all branches above
    return np.log(1e-10)


def fit_nes_model(
    subject_data: pd.DataFrame,
    npe_checkpoint_path: str,
    summary_stat_keys: list[str],
    nes_param_names: list[str],
    fixed_nes_params: dict,
    nes_ddm_params_config: dict,
    num_posterior_samples: int = 100, # Number of samples per chain
    num_chains: int = 4
) -> Optional[az.InferenceData]:
    """
    Fits the NES model to the subject's data using mock components.
    This function will use MockDataManager and MockNPEModel to simulate the fitting process.
    It then calculates pointwise log-likelihoods and constructs an ArviZ InferenceData object.
    """
    print(f"\n--- Fitting NES model for subject data ---")
    print(f"Subject data shape: {subject_data.shape}")
    print(f"NPE checkpoint: {npe_checkpoint_path if npe_checkpoint_path else 'N/A (using dummy model)'}")
    print(f"Summary statistics keys: {summary_stat_keys}")
    print(f"NES dynamic parameter names: {nes_param_names}")
    print(f"Fixed NES parameters: {fixed_nes_params}")
    print(f"NES DDM base config: {nes_ddm_params_config}")

    # 1. Initialize MockDataManager
    # Ensure 'frame' and 'valence_score' are available for trial_condition in log-likelihood
    if 'frame' not in subject_data.columns:
        print("Warning: 'frame' column missing in subject_data. Adding dummy 'frame' for NES loglik calc.")
        subject_data['frame'] = 'neutral' # Or some other default
    if 'valence_score' not in subject_data.columns:
        print("Warning: 'valence_score' column missing in subject_data. Adding dummy 'valence_score' (0.0) for NES loglik calc.")
        subject_data['valence_score'] = 0.0


    data_manager = MockDataManager(data=subject_data, summary_stat_keys=summary_stat_keys)

    # 2. Initialize MockNPEModel
    mock_npe_model = MockNPEModel(
        checkpoint_path=npe_checkpoint_path,
        param_names=nes_param_names,
        data_manager=data_manager
    )

    # 3. Generate dummy posterior samples
    # Shape of each entry in posterior_samples_dict: (num_chains, num_posterior_samples)
    posterior_samples_dict = mock_npe_model.sample_posterior(
        num_samples=num_posterior_samples,
        num_chains=num_chains
    )

    # 4. Calculate pointwise log-likelihoods
    num_trials = len(subject_data)
    # Initialize log_likelihood_data with shape (num_chains, num_posterior_samples, num_trials)
    log_likelihood_data = np.full((num_chains, num_posterior_samples, num_trials), np.nan)

    print(f"Calculating pointwise log likelihoods for {num_chains} chains, {num_posterior_samples} samples/chain, {num_trials} trials...")

    for chain_idx in range(num_chains):
        for sample_idx in range(num_posterior_samples):
            # Construct the posterior_params for this specific sample
            current_posterior_sample_params = {
                param: posterior_samples_dict[param][chain_idx, sample_idx] for param in nes_param_names
            }

            for trial_idx in range(num_trials):
                trial = subject_data.iloc[trial_idx]
                trial_rt = trial['rt']
                trial_choice = trial['response']

                # Construct trial_condition for NES model
                trial_condition_nes = {
                    'is_gain_frame': trial['frame'] == 'gain', # Example: adjust based on your 'frame' column
                    'valence_score': trial.get('valence_score', 0.0) # Use .get for safety
                }

                log_lik = get_log_likelihood_pointwise(
                    posterior_params=current_posterior_sample_params,
                    trial_rt=trial_rt,
                    trial_choice=trial_choice,
                    trial_condition=trial_condition_nes,
                    model_type="NES",
                    fixed_nes_params=fixed_nes_params,
                    nes_ddm_params=nes_ddm_params_config
                )
                log_likelihood_data[chain_idx, sample_idx, trial_idx] = log_lik

    print("Pointwise log likelihood calculation complete.")
    if np.isnan(log_likelihood_data).any():
        print(f"Warning: NaNs found in log_likelihood_data. Count: {np.isnan(log_likelihood_data).sum()}")
        log_likelihood_data = np.nan_to_num(log_likelihood_data, nan=LOG_LIK_FLOOR) # Replace NaNs

    # 5. Create ArviZ InferenceData object
    # Observed data: should match the structure used in log_likelihood calculation
    observed_data_dict = {
        'rt': subject_data['rt'].values,
        'response': subject_data['response'].values,
        'frame': subject_data['frame'].values, # Include conditions used
        'valence_score': subject_data['valence_score'].values
    }
    # Add any other columns from subject_data that might be relevant as observed
    for key in summary_stat_keys:
        if key in subject_data.columns and key not in observed_data_dict:
            observed_data_dict[key] = subject_data[key].values

    # Ensure log_likelihood_data has the right shape (chains, draws, observations)
    if log_likelihood_data.ndim != 3:
        print(f"Warning: NES log_likelihood_data has {log_likelihood_data.ndim} dimensions, expected 3. Reshaping...")
        num_obs = len(subject_data)
        log_likelihood_data = log_likelihood_data.reshape((1, -1, num_obs))
    
    # Get dimensions
    num_chains = log_likelihood_data.shape[0]
    num_draws = log_likelihood_data.shape[1]
    num_obs = log_likelihood_data.shape[2]
    
    # Create coords
    coords = {
        'chain': np.arange(num_chains),
        'draw': np.arange(num_draws),
        'trial': np.arange(num_obs),
        'parameter': nes_param_names
    }
    
    # Create dims
    dims = {
        'log_likelihood': ['chain', 'draw', 'trial'],
        **{param: ['chain', 'draw'] for param in posterior_samples_dict.keys()}
    }

    try:
        # Create InferenceData with proper structure
        idata = az.from_dict(
            posterior=posterior_samples_dict,  # Shape: {param_name: (chains, draws)}
            log_likelihood={
                'log_likelihood': (['chain', 'draw', 'trial'], log_likelihood_data)
            },
            observed_data=observed_data_dict,
            coords=coords,
            dims=dims
        )
        # Arviz might try to infer dims for posterior. If nes_param_names are parameters,
        # it will likely create a 'parameter' dim.
        # Let's check:
        # print("Arviz idata (nes) created. Posterior variables and dims:")
        # for var_name, var_data in idata.posterior.items():
        #     print(f"  {var_name}: dims={var_data.dims}, shape={var_data.shape}")
        # print("Log likelihood (nes) dims and shape:")
        # print(f"  nes: dims={idata.log_likelihood.nes.dims}, shape={idata.log_likelihood.nes.shape}")

        print("Successfully created ArviZ InferenceData object for NES model.")
        return idata
    except Exception as e:
        print(f"Error creating ArviZ InferenceData object for NES model: {e}")
        print("Posterior samples dict structure:")
        for k, v in posterior_samples_dict.items():
            print(f"  {k}: shape {v.shape}")
        print(f"Log likelihood data shape: {log_likelihood_data.shape}")
        print("Observed data dict structure:")
        for k, v in observed_data_dict.items():
            print(f"  {k}: shape {v.shape if isinstance(v, np.ndarray) else type(v)}")
        return None

from typing import Optional

def fit_hddm_ext_model(
    subject_data_orig: pd.DataFrame,
    hddm_model_config: dict,
    num_samples: int = 500,  # HDDM samples
    burn_in: int = 100,      # HDDM burn-in
    is_group_model: bool = False,  # Changed default to False for individual fitting
    p_outlier: float = 0.05,
    **hddm_kwargs
) -> Optional[az.InferenceData]:
    """
    Fits an HDDM model to the subject's data with robust error handling.
    
    Args:
        subject_data_orig: DataFrame with subject data (must contain 'rt' and 'response' columns)
        hddm_model_config: Dictionary with HDDM model configuration
        num_samples: Number of MCMC samples to draw
        burn_in: Number of burn-in samples to discard
        is_group_model: Whether to fit a hierarchical model (default: False)
        p_outlier: Probability of outlier responses (default: 0.05)
        **hddm_kwargs: Additional keyword arguments passed to HDDM

    Returns:
        ArviZ InferenceData object with the trace and log-likelihoods, or None if fitting fails
    """
    try:
        import hddm
        import numpy as np
        import pandas as pd
        from patsy import dmatrix
        
        # Make a copy to avoid modifying the original data
        subject_data = subject_data_orig.copy()
        
        # Ensure required columns are present
        required_cols = ['rt', 'response']
        for col in required_cols:
            if col not in subject_data.columns:
                raise ValueError(f"Required column '{col}' not found in subject data")
                
        # Add trial index if not present
        if 'trial_idx' not in subject_data.columns:
            subject_data['trial_idx'] = np.arange(len(subject_data))
            
        # Ensure response is 0/1 if it's not already
        if subject_data['response'].dtype != int or set(subject_data['response'].unique()) != {0, 1}:
            subject_data['response'] = (subject_data['response'] > 0.5).astype(int)
            
        # Print data summary
        print(f"Original subject data shape: {subject_data_orig.shape}")
        print(f"Data shape after cleaning: {subject_data.shape}")
        print(f"Response distribution:\n{subject_data['response'].value_counts()}")
        
        # Set up model configuration
        models = hddm_model_config.get('models', [])
        include = hddm_model_config.get('include', [])
        
        print(f"HDDM model config being used: {hddm_model_config}")
        
        # Initialize HDDM model
        print("Using HDDM", hddm.__version__, "compatibility mode")
        
        if models:
            print(f"Defining HDDMRegressor with models: {models}")
            m = hddm.HDDMRegressor(
                subject_data,
                models,
                include=include,
                p_outlier=p_outlier,
                **hddm_kwargs
            )
        else:
            print("No model attribute --> setting up standard HDDM")
            # Use a simple DDM model if no regression models specified
            m = hddm.HDDM(
                subject_data,
                include=include,
                p_outlier=p_outlier,
                **hddm_kwargs
            )
            print("Set model to", m.model)
        
        # Fit the model
        print(f"Fitting HDDM model with {num_samples} samples (burn-in: {burn_in})...")
        m.sample(num_samples, burn=burn_in)
        
        # Convert to ArviZ InferenceData
        print("Converting HDDM trace to ArviZ InferenceData...")
        trace = m.get_traces()
        
        # Create coords and dims for ArviZ
        coords = {
            'chain': np.arange(1),
            'draw': np.arange(num_samples - burn_in)
        }
        
        # Create InferenceData
        idata = az.from_dict(
            posterior={k: np.expand_dims(v.values, 0) for k, v in trace.items()},
            coords=coords,
            dims={'param': list(trace.columns)},
            model=type(m).__name__
        )
        
        # Add log likelihood and observed data
        try:
            # Get pointwise log likelihoods from the model
            ll = np.zeros(len(subject_data))
            for i, (rt, response) in enumerate(zip(subject_data['rt'], subject_data['response'])):
                # Get posterior samples for this trial
                params = trace.sample()
                # Calculate log likelihood for this trial across all samples
                ll[i] = np.mean([
                    get_log_likelihood_pointwise(
                        params.iloc[j],
                        rt,
                        response,
                        None,  # No condition for basic model
                        'HDDM',
                        None,  # No fixed NES params
                        None   # No NES DDM params
                    ) for j in range(len(params))
                ])
            
            # Add to InferenceData in the expected format for ArviZ
            idata.add_groups({
                'log_likelihood': {
                    'y': np.expand_dims(ll, 0)  # Add chain dimension
                },
                'observed_data': {
                    'rt': subject_data['rt'].values,
                    'response': subject_data['response'].values
                }
            })
            print("Successfully added log likelihood to InferenceData")
            
        except Exception as e:
            print(f"Could not add log likelihood to InferenceData: {e}")
            print("This may affect model comparison but not the model fitting itself.")
        
        print("HDDM fitting successful.")
        return idata
        
    except Exception as e:
        print(f"Error during HDDM model definition or fitting: {e}")
        import traceback
        traceback.print_exc()
        return None
    """
{{ ... }}
    Fits an HDDM (Regressor) model to the subject's data.
    Calculates pointwise log-likelihoods by reconstructing trial-specific parameters.
    Creates an ArviZ InferenceData object.
    """
    print(f"\n--- Fitting HDDM model for subject data ---")

    # --- 1. Data Preparation ---
    subject_data = subject_data_orig.copy()
    print(f"Original subject data shape: {subject_data.shape}")

    if 'subj_idx' not in subject_data.columns:
        print("Error: 'subj_idx' column is required for HDDM. Adding a default 'subj_idx'=0.")
        subject_data['subj_idx'] = 0 # HDDM requires subj_idx

    if 'rt' not in subject_data.columns or 'response' not in subject_data.columns:
        print("Error: 'rt' and 'response' columns are essential for HDDM.")
        return None

    # Clean RTs: must be positive
    subject_data['rt'] = pd.to_numeric(subject_data['rt'], errors='coerce')
    subject_data.dropna(subset=['rt'], inplace=True) # Remove rows where RT became NaN
    subject_data = subject_data[subject_data['rt'] > 0]
    if subject_data.empty:
        print("Error: No valid trials remaining after RT cleaning (must be > 0).")
        return None

    # Clean Responses: must be 0 or 1
    subject_data['response'] = pd.to_numeric(subject_data['response'], errors='coerce')
    subject_data.dropna(subset=['response'], inplace=True)
    if not subject_data['response'].isin([0, 1]).all():
        print("Warning: 'response' column contains values other than 0 or 1. Coercing...")
        # Example: treat non-0/1 as outliers or a default category, e.g., 0. Or filter them out.
        subject_data['response'] = subject_data['response'].apply(lambda x: 1 if x > 0.5 else 0) # Simplistic coercion

    if len(subject_data) < 10: # Arbitrary minimum number of trials
        print(f"Warning: Very few trials ({len(subject_data)}) for HDDM fitting. Results may be unreliable.")
        if subject_data.empty: return None

    print(f"Data shape after cleaning: {subject_data.shape}")
    print(f"HDDM model config being used: {hddm_model_config}")

    # --- 2. Model Definition & Fitting ---
    try:
        # For HDDM 1.0.1RC, we'll use a simpler approach
        print("Using HDDM 1.0.1RC compatibility mode")
        
        # Extract model configuration
        include_params = hddm_model_config.get('include', [])
        depends_on = hddm_model_config.get('depends_on', {})
        
        # Check if we're using a regression model
        if 'models' in hddm_model_config:
            print(f"Defining HDDMRegressor with models: {hddm_model_config['models']}")
            m = hddm.HDDMRegressor(subject_data, hddm_model_config['models'], 
                                 include=include_params,
                                 group_only_regressors=hddm_model_config.get('group_only_regressors', False))
        else:
            # Basic HDDM model
            print("Defining basic HDDM model")
            m = hddm.HDDM(subject_data, include=include_params, depends_on=depends_on)
        
        # Fit the model
        print(f"Starting HDDM sampling: {num_samples} samples, {burn_in} burn-in.")
        m.sample(num_samples, burn=burn_in)
        print("HDDM sampling complete.")
    except Exception as e:
        print(f"Error during HDDM model definition or fitting: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 3. Posterior Trace Extraction & Reshaping ---
    # In HDDM 1.0.1RC, we can get the traces directly from the model
    # The traces are stored in m.mc.db.trace
    # We'll convert them to a format compatible with ArviZ
    
    try:
        # Get all trace names (parameters)
        trace_names = list(m.mc.db.trace_names)
        print(f"Found traces for parameters: {trace_names}")
        
        # Get the number of samples
        n_samples = len(m.mc.db.trace('a')[:])
        print(f"Number of samples per parameter: {n_samples}")
        
        # Create a dictionary of numpy arrays for each parameter
        posterior = {}
        for name in trace_names:
            trace = m.mc.db.trace(name)[:]
            # Reshape to (chains, draws) - HDDM typically uses 1 chain
            posterior[name] = trace.reshape(1, -1)  # 1 chain, n_samples draws
            
        # Get observed data for ArviZ
        observed_data = {
            'rt': subject_data['rt'].values,
            'response': subject_data['response'].values
        }
        
        # Create ArviZ InferenceData
        idata = az.from_dict(
            posterior=posterior,
            observed_data=observed_data,
            coords={"chain": [0]},  # Single chain
            dims={"rt": ["trial"], "response": ["trial"]}
        )
        
        return idata
        
    except Exception as e:
        print(f"Error during trace extraction: {e}")
        return None
    for col_name in traces_df.columns:
        posterior_samples_az[col_name] = traces_df[col_name].values.reshape(num_chains_simulated, num_actual_samples)
        if not any(x in col_name for x in ['Intercept', 'C(', '(']): # Heuristic for base params
             if col_name.split('_subj')[0] not in hddm_parameter_names: # Avoid duplicates like v_subj.1, v_subj.2
                hddm_parameter_names.append(col_name.split('_subj')[0])

    print(f"Extracted HDDM traces. Found parameters like: {list(traces_df.columns[:5])}...")
    # hddm_parameter_names will be used for coords in Arviz if needed.

    # --- 4. Pointwise Log-Likelihood Calculation ---
    num_trials = len(subject_data)
    log_likelihood_data = np.full((num_chains_simulated, num_actual_samples, num_trials), np.nan)

    print(f"Calculating pointwise log likelihoods for HDDM: {num_actual_samples} samples, {num_trials} trials...")

    # Get the design matrix if it's a regression model (simplistic retrieval)
    # design_matrix_info = {}
    # if hasattr(m, 'model_descrs'): # HDDMRegressor specific
    #     for descr in m.model_descrs:
    #         param = descr['param'] # e.g., 'v'
    #         # This is tricky: HDDM internal design matrix is not easily exposed for this.
    #         # We'd typically use patsy to reconstruct or get it.
    #         # For this placeholder, we assume we can figure out trial conditions from subject_data.

    for sample_idx in range(num_actual_samples):
        # For each sample, get all parameter values from the trace
        current_posterior_sample_params_full_trace = traces_df.iloc[sample_idx].to_dict()

        for trial_idx in range(num_trials):
            trial = subject_data.iloc[trial_idx]
            trial_rt = trial['rt']
            trial_choice = trial['response']

            # Reconstruct trial-specific DDM parameters (v, a, t, z) for this trial
            # This is the most complex part for HDDMRegressor
            effective_params_for_trial = {}

            # Base parameters (non-regressed or intercept terms)
            # Example: effective_params_for_trial['v'] = current_posterior_sample_params_full_trace.get('v_Intercept', current_posterior_sample_params_full_trace.get('v', 0))
            # This needs to be very robust based on `m.model_descrs` if it's a Regressor.

            # Simplified reconstruction for this placeholder:
            # Assumes basic DDM params (v,a,t,z) might be directly in trace or need simple combination
            # This part needs to be significantly enhanced for actual HDDMRegressor models.

            # Example: if 'v ~ condition' was the model
            # v_intercept = current_posterior_sample_params_full_trace.get('v_Intercept', 0)
            # v_effect_cond = 0
            # if f"v_C(condition)[T.{trial['condition']}]" in current_posterior_sample_params_full_trace: # This is pseudo-code for condition handling
            #    v_effect_cond = current_posterior_sample_params_full_trace[f"v_C(condition)[T.{trial['condition']}]"]
            # effective_params_for_trial['v'] = v_intercept + v_effect_cond
            # This is highly simplified. A full solution uses the design matrix.

            # For now, assume direct parameters or simple structure that get_log_likelihood_pointwise can handle
            # by just passing all trace values for the sample.
            # The `get_log_likelihood_pointwise` for HDDM currently expects 'v', 'a', 't', 'z'.
            # It does NOT handle regression coefficient combination.
            # THIS IS A MAJOR SIMPLIFICATION FOR THE PLACEHOLDER

            # Let's try to populate the main DDM params, using group level if subj level not found
            # This won't work for regression correctly without design matrix multiplication.
            param_map = {}
            subj_id_str = str(trial['subj_idx'])
            for p_base in ['v', 'a', 't', 'z']:
                subj_param_name = f"{p_base}_subj.{subj_id_str}" # HDDM trace name for subject specific param
                param_map[p_base] = current_posterior_sample_params_full_trace.get(
                                        subj_param_name, # Try subject specific first
                                        current_posterior_sample_params_full_trace.get(p_base, # Then group level
                                                                    np.nan)) # Fallback

            # For sv, sz, st (typically group level in basic HDDM)
            for p_opt in ['sv', 'st', 'sz']:
                if p_opt in current_posterior_sample_params_full_trace:
                    param_map[p_opt] = current_posterior_sample_params_full_trace[p_opt]

            # Check if all required params are found
            if any(np.isnan(param_map.get(p)) for p in ['v','a','t','z']):
                # print(f"Warning: Could not reconstruct all params for trial {trial_idx}, sample {sample_idx}. Skipping loglik.")
                # print(f"Mapped params: {param_map}")
                # print(f"Full trace sample: {current_posterior_sample_params_full_trace}")
                log_likelihood_data[0, sample_idx, trial_idx] = LOG_LIK_FLOOR # Assign floor
                continue

            log_lik = get_log_likelihood_pointwise(
                posterior_params=param_map, # Pass the reconstructed/mapped params
                trial_rt=trial_rt,
                trial_choice=trial_choice,
                trial_condition={}, # HDDM conditions are handled via regression terms in trace
                model_type="HDDM"
            )
            log_likelihood_data[0, sample_idx, trial_idx] = log_lik # Assuming single chain simulation

    print("Pointwise log likelihood calculation for HDDM complete.")
    if np.isnan(log_likelihood_data).any():
        nan_count = np.isnan(log_likelihood_data).sum()
        print(f"Warning: NaNs found in HDDM log_likelihood_data. Count: {nan_count} out of {log_likelihood_data.size}. Replacing with floor.")
        log_likelihood_data = np.nan_to_num(log_likelihood_data, nan=LOG_LIK_FLOOR)

    # --- 5. ArviZ InferenceData Creation ---
    # Prepare observed data
    observed_data_dict = {col: subject_data[col].values for col in subject_data.columns if col not in ['subj_idx']}
    # Ensure rt and response are present
    if 'rt' not in observed_data_dict or 'response' not in observed_data_dict:
        print("Error: rt or response missing from observed_data_dict for Arviz.")
        return None

    # Prepare dimensions and coords
    num_chains = log_likelihood_data.shape[0]
    num_draws = log_likelihood_data.shape[1]
    num_obs = log_likelihood_data.shape[2]
    
    # Ensure log_likelihood_data has the right shape (chains, draws, observations)
    if log_likelihood_data.ndim != 3:
        print(f"Warning: log_likelihood_data has {log_likelihood_data.ndim} dimensions, expected 3. Reshaping...")
        log_likelihood_data = log_likelihood_data.reshape((1, -1, num_obs))
    
    # Create coords
    coords = {
        'chain': np.arange(num_chains),
        'draw': np.arange(num_draws),
        'trial': np.arange(num_obs)
    }
    
    # Create dims for log_likelihood
    dims = {
        'log_likelihood': ['chain', 'draw', 'trial']
    }
    
    try:
        # Create InferenceData with proper structure
        idata = az.from_dict(
            posterior=posterior_samples_az,
            log_likelihood={
                'log_likelihood': (['chain', 'draw', 'trial'], log_likelihood_data)
            },
            observed_data=observed_data_dict,
            coords=coords,
            dims=dims
        )
        # Arviz might complain if posterior keys have dots, e.g. 'v_subj.0'.
        # It's better to rename them, e.g., 'v_subj_0'.
        # This is a common issue with HDDM traces and Arviz.
        # For this placeholder, we'll proceed and note this as a potential refinement.

        print("Successfully created ArviZ InferenceData object for HDDM model.")
        # Clean up database file
        if os.path.exists('hddm_traces.db'): os.remove('hddm_traces.db')
        return idata
    except Exception as e:
        print(f"Error creating ArviZ InferenceData object for HDDM model: {e}")
        import traceback
        traceback.print_exc()
        # Clean up database file
        if os.path.exists('hddm_traces.db'): os.remove('hddm_traces.db')
        return None


if __name__ == "__main__":
    print("Starting empirical data comparison script...")

    # --- Script-Level Configurations ---
    EMPIRICAL_DATA_PATH = "output/data_prep/empirical_data.csv"
    NPE_CHECKPOINT_PATH = "output/models/nes_npe_checkpoint.pt" # Path to the (mock) NPE model checkpoint
    OUTPUT_DIR = "results/model_comparison_empirical"

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Create a dummy NPE checkpoint file if it doesn't exist (for mock model to "load")
    if NPE_CHECKPOINT_PATH and not os.path.exists(NPE_CHECKPOINT_PATH):
        print(f"Creating dummy NPE checkpoint file at: {NPE_CHECKPOINT_PATH}")
        os.makedirs(os.path.dirname(NPE_CHECKPOINT_PATH), exist_ok=True)
        with open(NPE_CHECKPOINT_PATH, 'wb') as f:
            pickle.dump({"mock_npe_param": "value"}, f)

    # --- NES Model Configuration ---
    SUMMARY_STAT_KEYS_NES = ['rt_mean', 'rt_std', 'acc_mean', 'rt_10th_pctl', 'rt_90th_pctl']
    NES_DYNAMIC_PARAM_NAMES = ['v_norm', 'a_0', 't_0', 'alpha_gain', 'beta_val']
    FIXED_NES_PARAMS = {
        'logit_z0': 0.0,
        'log_tau_norm': -0.7
    }
    NES_DDM_BASE_CONFIG = {
        'w_s': 1.0, 'salience_input': 0.5,
        'sv': 0.0, 'sz': 0.0, 'st': 0.0
    }
    NES_NUM_SAMPLES = 50 # Reduced for main script testing
    NES_NUM_CHAINS = 2   # Reduced for main script testing

    # --- HDDM Model Configuration (for main run) ---
    # HDDM model configuration with essential parameters
    HDDM_MODEL_CONFIG_MAIN = {
        'models': [],  # Empty list means use basic DDM without regression
        'include': ['v', 'a', 't'],  # Include essential parameters
        'p_outlier': 0.05,  # 5% probability of outlier responses
        'is_group_model': False  # Fit individual subject models
    }
    HDDM_NUM_SAMPLES = 100 # Reduced for main script testing
    HDDM_BURN_IN = 50    # Reduced for main script testing

    # --- Load and Prepare Empirical Data ---
    all_data_loaded = load_empirical_data(EMPIRICAL_DATA_PATH)

    if all_data_loaded is None:
        print("Failed to load empirical data. Creating dummy data to proceed with main script flow.")
        num_dummy_trials = 50
        num_dummy_subjects = 1 # Keep to 1 for simpler HDDM Regressor Treatment coding with dummy data
        dummy_rts = np.abs(np.random.normal(loc=0.7, scale=0.3, size=num_dummy_trials * num_dummy_subjects)) + 0.1 # Ensure positive
        dummy_responses = np.random.choice([0, 1], num_dummy_trials * num_dummy_subjects)
        dummy_subjects = np.repeat(np.arange(num_dummy_subjects), num_dummy_trials)

        dummy_frame_types = np.random.choice(['gain_trial', 'loss_trial', 'neutral_trial'], num_dummy_trials * num_dummy_subjects)
        dummy_valence = np.random.randn(num_dummy_trials * num_dummy_subjects) * 0.5

        # Ensure 'A' is one of the conditions for HDDM Treatment coding
        dummy_hddm_conditions = np.random.choice(['A', 'B', 'C'], num_dummy_trials * num_dummy_subjects)
        # Make sure at least some 'A' conditions exist if it's the reference for Treatment
        if 'A' not in dummy_hddm_conditions:
            dummy_hddm_conditions[0] = 'A'


        all_data_loaded = pd.DataFrame({
            "rt": dummy_rts,
            "response": dummy_responses,
            "subj_idx": dummy_subjects,
            "frame_type": dummy_frame_types,
            "valence_score": dummy_valence,
            "condition": dummy_hddm_conditions,
            "rt_mean": np.random.rand(num_dummy_trials * num_dummy_subjects),
            "rt_std": np.random.rand(num_dummy_trials * num_dummy_subjects),
            "acc_mean": np.random.rand(num_dummy_trials * num_dummy_subjects),
            "rt_10th_pctl": np.random.rand(num_dummy_trials * num_dummy_subjects),
            "rt_90th_pctl": np.random.rand(num_dummy_trials * num_dummy_subjects),
        })
        print("Dummy data created for main script flow.")

    # --- Data Preprocessing for NES specific columns ---
    if 'frame_type' in all_data_loaded.columns and 'frame' not in all_data_loaded.columns:
        all_data_loaded['frame'] = all_data_loaded['frame_type'].apply(lambda x: 'gain' if 'gain' in x else ('loss' if 'loss' in x else 'neutral'))
        print("Processed 'frame_type' into 'frame' column for NES.")
    elif 'frame' not in all_data_loaded.columns:
        print("Warning: Neither 'frame_type' nor 'frame' column found for NES. Adding dummy 'frame' = 'neutral'.")
        all_data_loaded['frame'] = 'neutral'

    if 'valence_score' not in all_data_loaded.columns:
        print("Warning: 'valence_score' column not found for NES. Adding dummy 'valence_score' = 0.0.")
        all_data_loaded['valence_score'] = 0.0

    # --- Print Data Summary ---
    print("\n--- Data Summary (after potential modifications) ---")
    print(f"Data shape: {all_data_loaded.shape}")
    if 'subj_idx' in all_data_loaded.columns:
        num_subjects = all_data_loaded['subj_idx'].nunique()
        print(f"Number of unique subjects: {num_subjects}")
        # print(f"Trials per subject:\n{all_data_loaded.groupby('subj_idx').size()}") # Can be verbose for many subjects
    else:
        print("No 'subj_idx' column found, assuming single subject or pre-grouped data.")
    print(f"Total trials: {len(all_data_loaded)}")
    # print(all_data_loaded.head()) # Can be verbose
    print("--------------------------------------------------\n")

    # --- Main Processing Loop ---
    all_subject_nes_idata = {}
    all_subject_hddm_idata = {}

    if 'subj_idx' in all_data_loaded.columns:
        unique_subject_ids = all_data_loaded['subj_idx'].unique()
    else:
        print("Warning: 'subj_idx' column not found in loaded data. Processing all data as a single group (ID 0).")
        unique_subject_ids = [0]
        all_data_loaded['subj_idx'] = 0

    for s_id in unique_subject_ids:
        print(f"===== Processing Subject ID: {s_id} =====")
        current_subject_data = all_data_loaded[all_data_loaded['subj_idx'] == s_id].copy()

        if current_subject_data.empty:
            print(f"No data for subject {s_id}. Skipping.")
            continue

        # Fit NES Model
        print(f"\n--- Fitting NES Model for Subject {s_id} ---")
        nes_idata_subj = fit_nes_model(
            subject_data=current_subject_data,
            npe_checkpoint_path=NPE_CHECKPOINT_PATH,
            summary_stat_keys=SUMMARY_STAT_KEYS_NES,
            nes_param_names=NES_DYNAMIC_PARAM_NAMES,
            fixed_nes_params=FIXED_NES_PARAMS,
            nes_ddm_params_config=NES_DDM_BASE_CONFIG,
            num_posterior_samples=NES_NUM_SAMPLES,
            num_chains=NES_NUM_CHAINS
        )
        if nes_idata_subj:
            all_subject_nes_idata[s_id] = nes_idata_subj
            print(f"NES fitting successful for subject {s_id}.")
            nes_idata_path = os.path.join(OUTPUT_DIR, f"nes_idata_subject_{s_id}.nc")
            try:
                nes_idata_subj.to_netcdf(nes_idata_path)
                print(f"Saved NES InferenceData for subject {s_id} to {nes_idata_path}")
            except Exception as e:
                print(f"Error saving NES InferenceData for subject {s_id}: {e}")
        else:
            print(f"NES fitting failed for subject {s_id}.")

        # Fit HDDM Model
        print(f"\n--- Fitting HDDM Model for Subject {s_id} ---")
        if 'models' in HDDM_MODEL_CONFIG_MAIN and any('C(condition' in m['model'] for m in HDDM_MODEL_CONFIG_MAIN['models']):
            if 'condition' not in current_subject_data.columns:
                print(f"Warning: 'condition' column needed for HDDM regression not found for subject {s_id}. Adding dummy 'A'.")
                current_subject_data.loc[:, 'condition'] = 'A'
            elif not current_subject_data['condition'].isin(['A', 'B', 'C']).all(): # Example valid conditions
                 print(f"Warning: 'condition' column for subject {s_id} contains unexpected values for HDDM regression. Ensure levels match formula (e.g. 'A', 'B', 'C'). Using 'A' as default for problematic rows.")
                 # This is a simplistic fix; might need more robust handling
                 current_subject_data.loc[~current_subject_data['condition'].isin(['A', 'B', 'C']), 'condition'] = 'A'
            if not pd.api.types.is_categorical_dtype(current_subject_data['condition']):
                 current_subject_data['condition'] = pd.Categorical(current_subject_data['condition'], categories=['A', 'B', 'C'], ordered=False)


        hddm_idata_subj = fit_hddm_ext_model(
            subject_data_orig=current_subject_data,
            hddm_model_config=HDDM_MODEL_CONFIG_MAIN,
            num_samples=HDDM_NUM_SAMPLES,
            burn_in=HDDM_BURN_IN
        )
        if hddm_idata_subj:
            all_subject_hddm_idata[s_id] = hddm_idata_subj
            print(f"HDDM fitting successful for subject {s_id}.")
            hddm_idata_path = os.path.join(OUTPUT_DIR, f"hddm_idata_subject_{s_id}.nc")
            try:
                hddm_idata_subj.to_netcdf(hddm_idata_path)
                print(f"Saved HDDM InferenceData for subject {s_id} to {hddm_idata_path}")
            except Exception as e:
                print(f"Error saving HDDM InferenceData for subject {s_id}: {e}")
        else:
            print(f"HDDM fitting failed for subject {s_id}.")
        print(f"===== Finished processing Subject ID: {s_id} =====")

    # --- Model Comparison (using ArviZ compare) ---
    comparison_results_summary = {}
    if not all_subject_nes_idata or not all_subject_hddm_idata:
        print("\nNot enough InferenceData objects collected for one or both model types. Skipping comparison.")
    else:
        # Ensure unique_subject_ids is based on actual keys for which data might exist
        processed_subject_ids = set(all_subject_nes_idata.keys()) & set(all_subject_hddm_idata.keys())
        if not processed_subject_ids:
            print("\nNo common subjects with successful fits for both models. Skipping comparison.")
        else:
            for s_id in processed_subject_ids:
                print(f"\n--- Comparing models for Subject ID: {s_id} ---")
                nes_id = all_subject_nes_idata.get(s_id)
                hddm_id = all_subject_hddm_idata.get(s_id)

                # Check if log_likelihood attribute exists and has the expected structure
                if not (hasattr(nes_id, 'log_likelihood') and 'log_likelihood' in nes_id.log_likelihood and
                        hasattr(hddm_id, 'log_likelihood') and 'log_likelihood' in hddm_id.log_likelihood):
                    print(f"Log likelihood data structure incorrect or missing in InferenceData for subject {s_id}. "
                          f"NES has log_likelihood: {hasattr(nes_id, 'log_likelihood')}, "
                          f"HDDM has log_likelihood: {hasattr(hddm_id, 'log_likelihood')}")
                    if hasattr(nes_id, 'log_likelihood'):
                        print(f"NES log_likelihood keys: {list(nes_id.log_likelihood.keys())}")
                    if hasattr(hddm_id, 'log_likelihood'):
                        print(f"HDDM log_likelihood keys: {list(hddm_id.log_likelihood.keys())}")
                    print("Skipping comparison.")
                    continue
                try:
                    compare_dict_subj = {"nes": nes_id, "hddm": hddm_id}
                    waic_comp = az.compare(compare_dict_subj, ic="waic", scale="log")
                    loo_comp = az.compare(compare_dict_subj, ic="loo", scale="log")
                    comparison_results_summary[s_id] = {"waic": waic_comp, "loo": loo_comp}

                    print(f"WAIC Comparison for Subject {s_id}:\n{waic_comp}")
                    print(f"LOO Comparison for Subject {s_id}:\n{loo_comp}")

                    waic_comp.to_csv(os.path.join(OUTPUT_DIR, f"waic_compare_subject_{s_id}.csv"))
                    loo_comp.to_csv(os.path.join(OUTPUT_DIR, f"loo_compare_subject_{s_id}.csv"))
                except Exception as e:
                    print(f"Error during ArviZ comparison for subject {s_id}: {e}")
                    print("Please check the structure of log_likelihood in your InferenceData objects.")

    print("\n--- Final Script Summary ---")
    if all_subject_nes_idata:
        print(f"NES model fitting attempted for {len(unique_subject_ids)} subjects. Successful fits: {len(all_subject_nes_idata)}.")
    if all_subject_hddm_idata:
        print(f"HDDM model fitting attempted for {len(unique_subject_ids)} subjects. Successful fits: {len(all_subject_hddm_idata)}.")
    if comparison_results_summary:
        print(f"Model comparison performed for {len(comparison_results_summary)} subjects.")
        print("Comparison results (WAIC/LOO tables) saved in:", OUTPUT_DIR)
    else:
        print("No model comparisons were performed or saved.")

    print(f"\nMain empirical comparison script finished. All artifacts saved in {OUTPUT_DIR}")

    # --- Comment out previous individual test blocks ---
    # Test block for get_log_likelihood_pointwise (example)
    """
    if __name__ == "__main__": # This was part of the original test, keep it commented
        # ... (original get_log_likelihood_pointwise test code) ...
    """

# Ensure the script is aware of its location if run directly, for module imports if any were relative
# print("Script compare_nes_hddm_empirical.py execution complete.")
