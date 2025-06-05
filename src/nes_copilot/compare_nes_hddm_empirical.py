import pandas as pd
import numpy as np
import arviz as az
import hddm
import os
import torch
from hddm.wfpt import wiener_like_rldf # For log likelihood calculation
# Add any other necessary imports here

# --- Constants for Log Likelihood ---
LOG_LIK_FLOOR = -1e9  # Floor for log likelihood values
EPS_FOR_LOG = 1e-29   # Epsilon to prevent log(0)

# --- Data Loading Function ---
def load_empirical_data(data_path: str) -> pd.DataFrame | None:
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
    Calculates the pointwise log likelihood for a single trial given posterior parameters.

    Args:
        posterior_params (dict or pd.Series): Dictionary or Series of posterior parameters for the model.
        trial_rt (float): Reaction time for the trial.
        trial_choice (int or float): Choice for the trial (e.g., 0 or 1).
        trial_condition (dict): Dictionary containing condition-specific information for the trial.
                                For NES, this includes 'is_gain_frame' (bool) and 'valence_score' (float).
        model_type (str): Type of model ("HDDM" or "NES").
        fixed_nes_params (dict, optional): Fixed parameters for the NES model (e.g., 'logit_z0', 'log_tau_norm').
                                           Required if model_type is "NES".
        nes_ddm_params (dict, optional): DDM-related parameters for the NES model (e.g., 'w_s', 'salience_input').
                                         Required if model_type is "NES".

    Returns:
        float: The calculated log likelihood for the trial.
    """
    if isinstance(posterior_params, pd.Series):
        posterior_params = posterior_params.to_dict()

    if model_type == "HDDM":
        v = posterior_params['v']
        a = posterior_params['a']
        t = posterior_params['t']
        z_rel = posterior_params['z']  # Relative start point

        sv = posterior_params.get('sv', 0.0)
        sz = posterior_params.get('sz', 0.0)
        st = posterior_params.get('st', 0.0)

        if trial_rt <= t or a <= 1e-6:
            return LOG_LIK_FLOOR

        z_abs = z_rel * a  # Absolute start point

        if not (1e-6 < z_abs < a - 1e-6): # Check if z_abs is within (0, a) excluding boundaries
            return LOG_LIK_FLOOR

        likelihood = wiener_like_rldf(
            trial_rt, v, sv, a, z_abs, sz, t, st,
            resp_upper_boundary=int(trial_choice),
            eps=1e-15 # Numerical precision for wfpt
        )
        return np.log(likelihood + EPS_FOR_LOG)

    elif model_type == "NES":
        if fixed_nes_params is None or nes_ddm_params is None:
            raise ValueError("fixed_nes_params and nes_ddm_params are required for NES model type.")

        # Extract NES dynamic parameters
        v_norm = posterior_params['v_norm']
        a_0 = posterior_params['a_0']
        t_0 = posterior_params['t_0']
        alpha_gain = posterior_params['alpha_gain']
        beta_val = posterior_params['beta_val']

        t_nes = t_0
        decision_time = trial_rt - t_nes

        if decision_time <= 1e-6: # Decision time must be positive
            return LOG_LIK_FLOOR

        is_gain = trial_condition.get('is_gain_frame', False) # Default to False if not provided
        a_nes = a_0 * (1.0 + alpha_gain) if is_gain else a_0

        if a_nes <= 1e-6: # Boundary must be positive
            return LOG_LIK_FLOOR

        logit_z_trial = fixed_nes_params['logit_z0'] + beta_val * trial_condition.get('valence_score', 0.0)
        z_nes_rel = 1.0 / (1.0 + np.exp(-logit_z_trial)) # Sigmoid transformation
        z_nes_abs = z_nes_rel * a_nes

        if not (1e-6 < z_nes_abs < a_nes - 1e-6): # Check if z_nes_abs is within (0, a_nes)
            return LOG_LIK_FLOOR

        norm_input = 1.0 if is_gain else -1.0
        tau_norm = np.exp(fixed_nes_params['log_tau_norm'])

        decay_factor = np.exp(-decision_time / tau_norm) if tau_norm > 1e-6 else 1.0 # Avoid division by zero

        v_drift_norm_component = v_norm * decay_factor * norm_input

        # Assuming w_s and salience_input are part of nes_ddm_params
        v_nes = nes_ddm_params['w_s'] * nes_ddm_params.get('salience_input', 0.0) - v_drift_norm_component

        sv_nes = nes_ddm_params.get('sv', 0.0)
        sz_nes = nes_ddm_params.get('sz', 0.0)
        st_nes = nes_ddm_params.get('st', 0.0)

        likelihood = wiener_like_rldf(
            trial_rt, v_nes, sv_nes, a_nes, z_nes_abs, sz_nes, t_nes, st_nes,
            resp_upper_boundary=int(trial_choice),
            eps=1e-15 # Numerical precision for wfpt
        )
        return np.log(likelihood + EPS_FOR_LOG)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def fit_nes_model(
    subject_data: pd.DataFrame,
    npe_checkpoint_path: str,
    summary_stat_keys: list[str],
    nes_param_names: list[str],
    fixed_nes_params: dict,
    nes_ddm_params_config: dict,
    num_posterior_samples: int = 100, # Number of samples per chain
    num_chains: int = 4
) -> az.InferenceData | None:
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
    # Observed data: should match the structure used in log_likelihood calculation (rt, response, conditions)
    observed_data_dict = {
        'rt': subject_data['rt'].values,
        'response': subject_data['response'].values,
        'frame': subject_data['frame'].values, # Include conditions used
        'valence_score': subject_data['valence_score'].values
    }
    # Add any other columns from subject_data that might be relevant as observed.
    # For example, if summary_stat_keys are direct columns from subject_data:
    for key in summary_stat_keys:
        if key in subject_data.columns and key not in observed_data_dict:
            observed_data_dict[key] = subject_data[key].values

    # Coordinate values for parameters
    coords = {'parameter': nes_param_names}

    try:
        idata = az.from_dict(
            posterior=posterior_samples_dict,  # Shape: {param_name: (chains, draws)}
            log_likelihood={'nes': log_likelihood_data},  # Shape: (chains, draws, trials)
            observed_data=observed_data_dict, # Should be a dictionary of 1D arrays (trials,)
            coords=coords,
            dims={'nes_log_likelihood': ['trial']} # Example dimension name for log_likelihood
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

def fit_hddm_ext_model(subject_data, hddm_model_config):
    """
    Fits the HDDM extension model to the subject's data.
    !!! REPLACE WITH ACTUAL HDDM FITTING !!!
    """
    print(f"Fitting HDDM extension model for subject_data shape: {subject_data.shape}")
    print(f"HDDM model config: {hddm_model_config}")

    # This is a dummy implementation for HDDM
    # In a real scenario, you would use HDDM to fit the model and extract samples
    # For example:
    # model = hddm.HDDM(subject_data, **hddm_model_config)
    # model.sample(2000, burn=200)
    # samples = model.get_traces()

    # Dummy InferenceData object
    num_draws = 100
    num_chains = 4

    # Assuming HDDM parameters like v, a, t, etc.
    # These names should match those expected by your HDDM model configuration
    hddm_param_names = ['v', 'a', 't', 'z'] # Example parameters
    if 'sv' in hddm_model_config.get('include', []): # Example of conditional parameter
        hddm_param_names.append('sv')

    posterior_samples = {param: np.random.randn(num_chains, num_draws) for param in hddm_param_names}

    # Dummy pointwise log likelihood
    log_likelihood_data = np.random.rand(num_chains, num_draws, len(subject_data))

    # Create observed_data dictionary for ArviZ
    observed_data_dict = {key: subject_data[key].values for key in subject_data.columns}

    idata = az.from_dict(
        posterior=posterior_samples,
        log_likelihood={'hddm': log_likelihood_data}, # Ensure this key matches model_name in az.compare
        observed_data=observed_data_dict,
        dims={'hddm': ['parameter']}, # Optional: add parameter dimension name
        coords={'parameter': hddm_param_names} # Optional: add parameter names
    )
    return idata

if __name__ == "__main__":
    print("Starting empirical data comparison script...")

    # --- Configuration ---
    # Path to the empirical data. Consider making this a command-line argument.
    # For now, using the specified default path.
    empirical_data_path = "output/data_prep/empirical_data.csv"
    npe_checkpoint_path = "output/models/nes_model_checkpoint.pt" # Example, update if needed
    output_dir = "results/comparison_empirical"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- Configuration for NES Model ---
    # These would typically come from a config file or a more structured setup
    summary_stat_keys_nes = [
        'rt_mean', 'rt_std', 'acc_mean', # Example summary stats
        'rt_10th_pctl', 'rt_90th_pctl'
    ]
    nes_dynamic_param_names = ['v_norm', 'a_0', 't_0', 'alpha_gain', 'beta_val'] # Dynamic params from NPE

    # Fixed parameters for the NES model (part of its definition, not from NPE)
    fixed_nes_params_config = {
        'logit_z0': 0.0,       # Base logit for starting point z
        'log_tau_norm': -0.7   # Log of decay constant for v_norm component
    }

    # DDM-like parameters for the NES model (base DDM structure, not from NPE)
    nes_ddm_base_config = {
        'w_s': 1.0,            # Weight for salience input to drift
        'salience_input': 0.5, # A constant or condition-driven salience input
        'sv': 0.0,             # Inter-trial variability of drift (set to 0 if not used)
        'sz': 0.0,             # Inter-trial variability of start point (set to 0 if not used)
        'st': 0.0              # Inter-trial variability of non-decision time (set to 0 if not used)
    }

    # HDDM model configuration (remains for the HDDM part of the script)
    hddm_model_config = {
        "include": ["sv", "st", "sz"],
        "p_outlier": 0.05
    }

    # --- Load Empirical Data ---
    all_data = load_empirical_data(empirical_data_path)

    if all_data is None:
        print("Failed to load empirical data. Exiting script.")
        # Optionally, create dummy data to allow the script to proceed for testing structure
        # For now, we will exit if data loading fails.
        # To enable dummy data for testing, uncomment the block below:
        # print("Creating dummy data to proceed with script structure testing...")
        # num_dummy_trials = 100
        # num_dummy_subjects = 2
        # dummy_rts = np.random.uniform(0.2, 2.0, num_dummy_trials * num_dummy_subjects)
        # dummy_responses = np.random.choice([0, 1], num_dummy_trials * num_dummy_subjects)
        # dummy_subjects = np.repeat(np.arange(num_dummy_subjects), num_dummy_trials)
        # dummy_stimuli = np.random.choice(['stimA', 'stimB'], num_dummy_trials * num_dummy_subjects)
        # all_data = pd.DataFrame({
        #     "rt": dummy_rts,
        #     "response": dummy_responses,
        #     "subj_idx": dummy_subjects,
        #     "stimulus": dummy_stimuli,
        #     "frame": np.tile(np.arange(num_dummy_trials), num_dummy_subjects), # Dummy frame column
        #     **{f"rt_{i+1}": np.random.rand(num_dummy_trials * num_dummy_subjects) for i in range(5)},
        #     **{f"response_{i+1}": np.random.randint(0,2, num_dummy_trials * num_dummy_subjects) for i in range(5)}
        # })
        # print("Dummy data created.")
        # Create a dummy NPE checkpoint file if it doesn't exist, for mock model to "load"
        if npe_checkpoint_path and not os.path.exists(npe_checkpoint_path):
            print(f"Creating dummy NPE checkpoint file at: {npe_checkpoint_path}")
            os.makedirs(os.path.dirname(npe_checkpoint_path), exist_ok=True)
            with open(npe_checkpoint_path, 'wb') as f:
                pickle.dump({"dummy_model_param": "value"}, f) # Save some dummy content

        # exit() # Exit if data loading failed and no dummy data is generated.
        # If all_data is None, create dummy data to allow script to run for testing structure
        if all_data is None:
            print("Creating dummy data as loading failed, to proceed with script structure testing...")
            num_dummy_trials = 50 # Reduced for faster testing
            num_dummy_subjects = 1 # Reduced for faster testing
            dummy_rts = np.random.uniform(0.2, 1.5, num_dummy_trials * num_dummy_subjects)
            dummy_responses = np.random.choice([0, 1], num_dummy_trials * num_dummy_subjects)
            dummy_subjects = np.repeat(np.arange(num_dummy_subjects), num_dummy_trials)
            dummy_frames = np.random.choice(['gain', 'loss', 'neutral'], num_dummy_trials * num_dummy_subjects)
            dummy_valence = np.random.randn(num_dummy_trials * num_dummy_subjects) * 0.5

            all_data = pd.DataFrame({
                "rt": dummy_rts,
                "response": dummy_responses,
                "subj_idx": dummy_subjects,
                "frame": dummy_frames,
                "valence_score": dummy_valence,
                # Dummy summary stats (can be generated by MockDataManager too)
                "rt_mean": np.random.rand(num_dummy_trials * num_dummy_subjects),
                "rt_std": np.random.rand(num_dummy_trials * num_dummy_subjects),
                "acc_mean": np.random.rand(num_dummy_trials * num_dummy_subjects),
                "rt_10th_pctl": np.random.rand(num_dummy_trials * num_dummy_subjects),
                "rt_90th_pctl": np.random.rand(num_dummy_trials * num_dummy_subjects),
            })
            print("Dummy data created.")


    # Print data summary
    print("\n--- Data Summary ---")
    if all_data is not None:
        print(f"Loaded data shape: {all_data.shape}")
        if 'subj_idx' in all_data.columns:
            num_subjects = all_data['subj_idx'].nunique()
            print(f"Number of unique subjects: {num_subjects}")
            print(f"Trials per subject:\n{all_data.groupby('subj_idx').size()}")
        else:
            print("No 'subj_idx' column found, assuming single subject or pre-grouped data.")
            num_subjects = 1 # Default to 1 if no subject identifier
        print(f"Total trials: {len(all_data)}")
        print("First 5 rows of the data:")
        print(all_data.head())
    else:
        print("all_data is None, cannot print summary.")
    print("--------------------\n")

    # Store InferenceData objects for each model and subject
    all_nes_idata = {}
    all_hddm_idata = {}

    # Loop through subjects (or handle data hierarchically)
    # HDDM typically handles multiple subjects internally if 'subj_idx' is in the data
    # For NES, you might fit per subject or use a hierarchical NES approach.
    # This example assumes fitting per subject for NES for simplicity.

    # Determine unique subjects for looping
    if 'subj_idx' in all_data.columns:
        unique_subjects = all_data['subj_idx'].unique()
    else:
        # If no subj_idx, assume all data belongs to a single group/subject
        # or that HDDM will handle it (e.g. if it's already single subject data)
        print("Warning: 'subj_idx' column not found. Processing data as a single group.")
        unique_subjects = [0] # Placeholder for single group processing
        all_data['subj_idx'] = 0 # Add a default subject index if not present

    for subject_id in unique_subjects:
        print(f"\nProcessing subject/group: {subject_id}")
        # If processing as a single group (subject_id == 0 and no 'subj_idx' originally),
        # use all_data. Otherwise, filter by actual subject_id.
        if 'subj_idx' in all_data.columns and subject_id in all_data['subj_idx'].unique():
             subject_data = all_data[all_data['subj_idx'] == subject_id].copy()
        else: # Handles the case where we added subj_idx = 0 to ungrouped data
             subject_data = all_data.copy()

        # Ensure 'rt' and 'response' columns are present for HDDM for this subject's data
        if 'rt' not in subject_data.columns or 'response' not in subject_data.columns:
            print(f"Error: 'rt' or 'response' column not found in data for subject/group {subject_id}.")
            print("HDDM requires these columns. Skipping this subject/group.")
            continue

        # Ensure subject_data is not empty
        if subject_data.empty:
            print(f"Skipping subject/group {subject_id} due to empty data after filtering.")
            continue

        # Fit NES model
        print(f"Fitting NES model for subject/group {subject_id}...")
        nes_idata = fit_nes_model(
            subject_data=subject_data,
            npe_checkpoint_path=npe_checkpoint_path,
            summary_stat_keys=summary_stat_keys_nes, # Use NES specific summary stats
            nes_param_names=nes_dynamic_param_names, # Use NES specific dyn param names
            fixed_nes_params=fixed_nes_params_config,
            nes_ddm_params_config=nes_ddm_base_config,
            num_posterior_samples=50, # Reduced for faster testing
            num_chains=2              # Reduced for faster testing
        )
        if nes_idata:
            all_nes_idata[subject_id] = nes_idata
            print(f"NES model fitting complete for subject/group {subject_id}.")
            # print(all_nes_idata[subject_id]) # Print summary of idata
        else:
            print(f"NES model fitting failed for subject/group {subject_id}.")


        # Fit HDDM extension model
        print(f"Fitting HDDM model for subject {subject_id}...")
        # HDDM expects data for all subjects at once if fitting hierarchically,
        # or data for a single subject if fitting individually.
        # For this example, we pass the whole dataset to HDDM if multiple subjects are handled by HDDM itself,
        # or subject_data if fitting one by one (though HDDM usually prefers grouped data).
        # The current HDDM placeholder `fit_hddm_ext_model` takes `subject_data`.
        # If your actual HDDM fitting function handles all subjects, you might call it once outside the loop.

        # For HDDM, ensure the data passed contains the subject identifier if the model is hierarchical
        # and the configuration expects it.
        # The `fit_hddm_ext_model` placeholder currently takes `subject_data`.
        # If your actual HDDM fitting is hierarchical and expects all data, you might call it once, outside this loop.
        # For this structure, we assume `fit_hddm_ext_model` can handle data per subject or that `subject_data` is appropriate.

        # Check if 'stimulus' column is present for HDDM if depends_on uses it
        # This check is illustrative; your actual HDDM config will determine requirements.
        if "depends_on" in hddm_model_config:
            for param, condition in hddm_model_config["depends_on"].items():
                if condition not in subject_data.columns: # e.g., if condition is 'stimulus'
                    print(f"Warning: Column '{condition}' for parameter '{param}' in HDDM 'depends_on' config "
                          f"not found in data for subject/group {subject_id}. HDDM may fail or use default.")
                    # Optionally, add a dummy column if critical for placeholder execution:
                    # subject_data[condition] = 'default_condition_value'

        hddm_idata = fit_hddm_ext_model(subject_data, hddm_model_config)
        all_hddm_idata[subject_id] = hddm_idata
        print(f"HDDM model fitting complete for subject {subject_id}.")

    # Perform model comparison using ArviZ
    # This comparison will be per subject if models were fit per subject
    # Or a single comparison if models were fit hierarchically over all subjects.

    comparison_results = {}

    # Check if there are any subjects to process before attempting comparison
    if not unique_subjects or not any(s_id in all_nes_idata and s_id in all_hddm_idata for s_id in unique_subjects):
        print("\nNo subjects processed successfully or no InferenceData available for comparison. Skipping model comparison.")
    else:
        for subject_id in unique_subjects:
            print(f"\nComparing models for subject/group: {subject_id}")
            if subject_id not in all_nes_idata or subject_id not in all_hddm_idata:
                print(f"Skipping comparison for subject/group {subject_id} due to missing InferenceData for one or both models.")
                continue

            nes_idata_subj = all_nes_idata[subject_id]
            hddm_idata_subj = all_hddm_idata[subject_id]

            # Ensure log_likelihood is present
            if 'log_likelihood' not in nes_idata_subj or 'nes' not in nes_idata_subj.log_likelihood:
                print(f"Error: Log likelihood 'nes' not found in NES InferenceData for subject/group {subject_id}.")
                continue
            if 'log_likelihood' not in hddm_idata_subj or 'hddm' not in hddm_idata_subj.log_likelihood:
                print(f"Error: Log likelihood 'hddm' not found in HDDM InferenceData for subject/group {subject_id}.")
                continue

            try:
                compare_models_dict = {"nes": nes_idata_subj, "hddm": hddm_idata_subj}

                # Check if log_likelihood data is empty or problematic before calling compare
                if nes_idata_subj.log_likelihood['nes'].size == 0 or hddm_idata_subj.log_likelihood['hddm'].size == 0:
                    print(f"Error: Log likelihood data is empty for one or both models for subject/group {subject_id}. Skipping comparison.")
                    continue

                waic_compare = az.compare(compare_models_dict, ic="waic", scale="log")
                loo_compare = az.compare(compare_models_dict, ic="loo", scale="log")

                comparison_results[subject_id] = {
                    "waic": waic_compare,
                    "loo": loo_compare
                }
                print(f"WAIC comparison for subject/group {subject_id}:\n{waic_compare}")
                print(f"LOO comparison for subject/group {subject_id}:\n{loo_compare}")

            except Exception as e:
                print(f"Error during ArviZ comparison for subject/group {subject_id}: {e}")
                print("Ensure that InferenceData objects are correctly formatted with non-empty log likelihood data.")

    # Save results
    if comparison_results:
        print(f"\nSaving comparison results to {output_dir}...")
        for subject_id, results in comparison_results.items():
            try:
                results["waic"].to_csv(os.path.join(output_dir, f"waic_compare_subject_{subject_id}.csv"))
                results["loo"].to_csv(os.path.join(output_dir, f"loo_compare_subject_{subject_id}.csv"))
                # az.to_netcdf(all_nes_idata[subject_id], os.path.join(output_dir, f"nes_idata_subject_{subject_id}.nc"))
                # az.to_netcdf(all_hddm_idata[subject_id], os.path.join(output_dir, f"hddm_idata_subject_{subject_id}.nc"))
            except Exception as e:
                print(f"Error saving results for subject/group {subject_id}: {e}")
        print("Results saved.")
    else:
        print("\nNo comparison results to save.")

    print(f"\nComparison script finished.")

    if comparison_results:
        print("\n--- Final Comparison Summaries ---")
        for subject_id, result in comparison_results.items():
            print(f"\nSummary for Subject/Group {subject_id}:")
            print("WAIC Comparison:")
            print(result['waic'])
            print("\nLOO Comparison:")
            print(result['loo'])
    else:
        print("\nNo final comparison summaries to display.")

    # --- Test block for get_log_likelihood_pointwise ---
    # In a real run, all_data would come from load_empirical_data
    all_trials_df = all_data # Use loaded data if available

    if 'all_trials_df' not in locals() or all_trials_df is None or all_trials_df.empty:
        print("Creating dummy all_trials_df for testing get_log_likelihood_pointwise.")
        dummy_test_data = {
            'subj_idx': [1, 1, 1],
            'rt': [0.5, 0.7, 0.6],
            'response': [1, 0, 1], # Assuming 1 for upper, 0 for lower
            'frame': ['gain', 'loss', 'gain'], # Example: 'gain' or 'loss'
            'valence_score': [0.5, -0.3, 0.8] # Example valence scores
        }
        all_trials_df = pd.DataFrame(dummy_test_data)

    if not all_trials_df.empty:
        sample_trial = all_trials_df.iloc[0]

        # Test HDDM
        dummy_hddm_params = {'v': 1.5, 'a': 2.0, 't': 0.3, 'z': 0.5, 'sv': 0.1, 'sz': 0.1, 'st': 0.05}
        trial_cond_hddm = {} # HDDM conditions can be passed here if model depends on them (e.g. v ~ condition)

        if sample_trial['rt'] > dummy_hddm_params['t']:
            log_lik_hddm = get_log_likelihood_pointwise(
                posterior_params=dummy_hddm_params,
                trial_rt=sample_trial['rt'],
                trial_choice=sample_trial['response'],
                trial_condition=trial_cond_hddm,
                model_type="HDDM"
            )
            print(f"Test HDDM Log Lik: {log_lik_hddm}")
        else:
            print(f"Sample trial RT {sample_trial['rt']} <= t {dummy_hddm_params['t']} for HDDM test, skipping.")

        # Test NES
        dummy_nes_dyn_params = {'v_norm': 1.0, 'a_0': 1.5, 't_0': 0.2, 'alpha_gain': 0.1, 'beta_val': 0.5}
        fixed_nes_params_test = {'logit_z0': 0.0, 'log_tau_norm': -0.7} # log_tau_norm for exp decay
        nes_ddm_params_test = {'w_s': 1.0, 'salience_input': 1.0, 'sv': 0.0, 'sz': 0.0, 'st': 0.0} # DDM base params for NES

        trial_cond_nes = {
            'is_gain_frame': sample_trial['frame'] == 'gain',
            'valence_score': sample_trial['valence_score']
        }

        if sample_trial['rt'] > dummy_nes_dyn_params['t_0']:
            log_lik_nes = get_log_likelihood_pointwise(
                posterior_params=dummy_nes_dyn_params,
                trial_rt=sample_trial['rt'],
                trial_choice=sample_trial['response'],
                trial_condition=trial_cond_nes,
                model_type="NES",
                fixed_nes_params=fixed_nes_params_test,
                nes_ddm_params=nes_ddm_params_test
            )
            print(f"Test NES Log Lik: {log_lik_nes}")
        else:
            print(f"Sample trial RT {sample_trial['rt']} <= t_0 {dummy_nes_dyn_params['t_0']} for NES test, skipping.")


# Ensure the script is aware of its location if run directly, for module imports if any were relative
# print("Script compare_nes_hddm_empirical.py execution complete.")
