**Project Plan: "NES Co-Pilot" - An Agentic System for Iterative Cognitive Model Development & Validation**

**1. Vision & Overall Goal:**

To develop a modular, extensible software system ("NES Co-Pilot") that automates and streamlines the iterative process of training, validating (via SBC), fitting (to empirical data), and evaluating (via PPC) computational cognitive models, specifically focusing on the Normative Executive System (NES) and similar DDM-based architectures. The system should be designed to eventually incorporate agentic capabilities for self-correction, experimental design suggestion, and hypothesis testing based on evolving results.

**2. Core Principles:**

*   **Modularity:** Each major stage (NPE training, SBC, Empirical Fitting, PPC, Data Preprocessing, Simulation, Analysis) should be a distinct, configurable module.
*   **Configuration-Driven:** The system should be primarily driven by configuration files (e.g., YAML, JSON) that specify parameters, data paths, model variants, and analysis choices, rather than requiring extensive code changes for each new experiment.
*   **Extensibility:** Easy to add new model parameters, summary statistics, simulator variants, or analysis routines.
*   **Reproducibility:** Rigorous tracking of seeds, software versions, configurations, and outputs for every run. Version control of configurations and outputs is key.
*   **Automation:** Minimize manual intervention for standard pipeline runs.
*   **Diagnostics & Logging:** Comprehensive logging and generation of diagnostic plots/reports at each stage.
*   **(Future) Agentic Capabilities:** Design with hooks for future AI-driven decision-making (e.g., "If SBC fails for parameter X, try adding summary stat Y and rerun with N_test sims").

**3. System Architecture & Key Modules:**

**(A) Core Pipeline Engine:**
*   **Workflow Manager:** Orchestrates the execution of different modules based on a master configuration file. Handles dependencies between stages (e.g., NPE must be trained before SBC can use it).
    *   *Consider using existing workflow tools like Snakemake, Nextflow, or even simpler Python-based orchestration with `luigi` or `prefect` for robust dependency management and execution.*
*   **Configuration Manager:** Loads, validates, and provides access to run-specific configurations.
*   **Data Manager / Versioner:** Handles paths to input data, manages output directories, and versions results (e.g., using DVC - Data Version Control, or a clear naming convention and metadata files).
*   **Logging & Reporting Module:** Centralized logging, generation of HTML/PDF reports summarizing runs.

**(B) Data Preparation Module:**
*   **Empirical Data Loader & Preprocessor:**
    *   Input: Path to raw empirical data (e.g., Roberts et al. CSV), configuration for filtering (e.g., `trialType='target'`), subject exclusion criteria.
    *   Output: Cleaned, preprocessed empirical dataframes (per subject, or a combined one) ready for summary stat calculation or providing trial structures.
*   **Stimulus Valence Processor (NEW):**
    *   Input: Stimulus text (from empirical data or defined for conditions).
    *   Action: Uses a specified sentiment analysis model (e.g., RoBERTa via Hugging Face Transformers) and rescaling rules (from config) to generate `valence_score` per stimulus.
    *   Output: Dataset of stimuli mapped to valence scores.
*   **Trial Template Generator:**
    *   Input: Preprocessed empirical data, number of template trials, seed.
    *   Action: Creates the `SUBJECT_TRIAL_STRUCTURE_TEMPLATE` (as in your current scripts), now also incorporating `valence_score` and `norm_category_for_trial` from the Valence Processor and configuration.
    *   Output: Saved trial template (e.g., CSV or Feather file).

**(C) Simulation Module (Wraps `MVNESAgent`):**
*   **Simulator Interface:** A standardized Python interface that takes:
    *   A parameter set (`theta`).
    *   A trial structure (e.g., the global template or a specific subject's trial sequence).
    *   Fixed model parameters (e.g., `logit_z0`, `log_tau_norm`).
    *   Configuration for `MVNESAgent`.
*   **Action:** Initializes `MVNESAgent` with the current configuration, runs all trials using the provided `theta` and trial structure (including passing `valence_score_trial` and `norm_category_for_trial` to `MVNESAgent.run_mvnes_trial`).
*   **Output:** A DataFrame of simulated choices and RTs for one full dataset.
*   *This module should be easily adaptable if `MVNESAgent`'s parameters or internal logic change.*

**(D) Summary Statistics Module:**
*   **Schema Definition (`stats_schema.py`):** Central definition of `ROBERTS_SUMMARY_STAT_KEYS` and `EXPECTED_NUM_STATS`. (You've already created this!)
*   **Calculator Function:** A robust `calculate_summary_stats(df_trials, stat_keys)` function that takes a trial DataFrame and returns a dictionary or ordered vector of stats. (Your current `calculate_summary_stats_roberts` is a good base).
*   **Configuration:** Allow selection of different summary stat sets (e.g., "lean_25", "full_60") via configuration.

**(E) NPE Training Module:**
*   **Input:** Prior definitions, number of training simulations, path to trial template, simulator configuration, summary statistics configuration, NPE architecture details (e.g., MAF, hidden features, transforms), device, seed.
*   **Action:**
    1.  Generates `N_train` parameter sets (`theta`) from the prior.
    2.  For each `theta`, calls the Simulation Module to generate a dataset, then the Summary Statistics Module to get `x`.
    3.  Trains the `sbi.inference.SNPE` model.
*   **Output:** Saved NPE checkpoint (`.pt` file containing `density_estimator_state_dict` and comprehensive metadata: priors, `num_summary_stats`, `summary_stat_keys`, training args, `sbi_version`, etc.).

**(F) Simulation-Based Calibration (SBC) Module:**
*   **Input:** Path to a trained NPE checkpoint, number of SBC datasets, number of posterior samples per SBC dataset, device, seed.
*   **Action:**
    1.  Loads the NPE.
    2.  Generates `N_sbc` ground truth parameter sets (`theta_gt`).
    3.  For each `theta_gt`, calls Simulation Module then Summary Stats Module to get `x_obs_sbc`.
    4.  For each `x_obs_sbc`, samples from the loaded NPE posterior to get `posterior_samples`.
    5.  Calculates ranks of `theta_gt` within `posterior_samples`.
*   **Output:**
    *   SBC ranks CSV.
    *   KS test results JSON.
    *   SBC diagnostic plots (ECDFs, histograms).
    *   SBC run metadata.

**(G) Empirical Fitting Module:**
*   **Input:** Path to a trained & SBC-validated NPE checkpoint, path to preprocessed empirical data, number of posterior samples per subject, device, seed.
*   **Action:**
    1.  Loads the NPE.
    2.  For each empirical subject:
        a.  Calculates their observed summary statistics (`x_empirical_subj`).
        b.  Samples from the loaded NPE posterior conditioned on `x_empirical_subj`.
*   **Output:** CSV file with posterior summaries (mean, median, std) for each parameter for each subject. (Optionally save full posterior traces per subject).

**(H) Posterior Predictive Check (PPC) Module:**
*   **Input:** Path to an NPE checkpoint, path to empirical fitting results (posterior summaries/samples per subject), path to preprocessed empirical data, number of PPC simulations per subject, number of posterior draws to use for PPC, device, seed.
*   **Action:**
    1.  Loads NPE and empirical fitting results.
    2.  For each subject (or selected subset):
        a.  Draws parameter sets from their fitted posterior.
        b.  For each parameter set, calls Simulation Module (using subject's *actual trial structure*) then Summary Stats Module.
        c.  Compares distributions of simulated summary stats to observed summary stats.
*   **Output:**
    *   PPC coverage summary table (CSV).
    *   Visual PPC plots for representative subjects/statistics.

**(I) Analysis & Visualization Module:**
*   **Input:** Outputs from Empirical Fitting (fitted params) and PPC (coverage table), empirical data.
*   **Action:**
    *   Calculates correlations (e.g., `v_norm` vs. framing index, `alpha_gain` vs. RT metrics).
    *   Generates publication-quality plots (scatter plots with correlations, parameter distributions, PPC coverage heatmaps/bar charts).
*   **Output:** Saved plots and correlation tables.

**4. Configuration System:**

*   **Master Configuration File (e.g., `run_config.yaml`):**
    *   Specifies which modules to run (e.g., `run_npe_training: true`, `run_sbc: true`, `run_empirical_fit: false`).
    *   Global settings (seed, device).
    *   Paths to common data (Roberts raw data).
    *   Pointers to module-specific configuration files.
*   **Module-Specific Configuration Files (e.g., `npe_config.yaml`, `sbc_config.yaml`):**
    *   `npe_config.yaml`: `num_training_sims`, `template_trials`, prior definitions (or path to prior spec), NPE architecture.
    *   `sbc_config.yaml`: `npe_checkpoint_to_use`, `num_sbc_datasets`, `num_posterior_samples`.
    *   `empirical_fit_config.yaml`: `npe_checkpoint_to_use`, `num_posterior_samples`.
    *   `ppc_config.yaml`: `npe_checkpoint_to_use`, `empirical_fit_results_file`, `num_ppc_simulations`, etc.
    *   `summary_stats_config.yaml`: Pointer to `stats_schema.py` or allows selection of different pre-defined schemas.
    *   `simulator_config.yaml`: Which version of `MVNESAgent` to use, fixed parameters for `MVNESAgent`.

**5. Technology Stack Considerations:**

*   **Primary Language:** Python.
*   **Core Libraries:** `sbi`, `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `tqdm`.
*   **Sentiment Analysis:** `transformers` (Hugging Face).
*   **Configuration:** `PyYAML` or `json`.
*   **Workflow Management (Optional but Recommended for complex pipelines):**
    *   `Snakemake`: Python-based, good for bioinformatics but general-purpose, defines workflows using rules and dependencies. Manages parallelism.
    *   `Nextflow`: Groovy-based (can call Python scripts), excellent for parallelism and HPC.
    *   `Luigi` / `Prefect` / `Airflow`: Python-based task scheduling and dependency management.
*   **Data Versioning (Optional but Recommended):** `DVC`.
*   **Parallelism:** `joblib`, Python's `multiprocessing`, or features within the chosen workflow manager. `pebble` for timeouts if fine-grained process control is needed.

**6. Development Trajectory (Iterative):**

1.  **Phase 1: Core Modules & Configuration (No Agentic AI yet)**
    *   Develop robust, standalone Python scripts for each module (NPE train, SBC, Empirical Fit, PPC). This is largely what you've been doing.
    *   Refine `stats_schema.py` and the `MVNESAgent`.
    *   Implement the centralized configuration system (e.g., YAML files).
    *   Create a simple master script that reads the config and calls the module scripts in sequence.
    *   Focus on rigorous output saving and metadata tracking.

2.  **Phase 2: Workflow Automation & Robustness**
    *   Integrate a workflow manager (e.g., Snakemake) to handle dependencies, execution, and re-computation automatically.
    *   Implement more sophisticated logging and reporting.
    *   Add comprehensive unit and integration tests for each module.

3.  **Phase 3: Introducing "Agentic" Heuristics (Rule-Based)**
    *   Develop a "Supervisor" module that analyzes outputs from SBC/PPC.
    *   Implement rule-based decisions:
        *   "If SBC KS p-value for param `X` < 0.05 AND ECDF shows U-shape, THEN try increasing `NPE_TRAINING_SIMS` by 50% and rerun NPE_train & SBC."
        *   "If PPC coverage for `mean_rt_Gain_TC` < 20%, THEN trigger an experiment: modify simulator with `alpha_gain_test_value = 0.7` and rerun PPC for 3 subjects."
    *   This would involve the Supervisor being able to generate new configuration files and re-trigger parts of the workflow.

4.  **Phase 4: Advanced Agentic Capabilities (Machine Learning for Control - More Ambitious)**
    *   Train a meta-learner (e.g., reinforcement learning agent, Bayesian optimization) to make decisions about:
        *   Which summary statistics to add/remove.
        *   Which model variants (e.g., different fixed params, different `MVNESAgent` structures) to try.
        *   Optimal hyperparameters for NPE training.
    *   This agent would learn from the history of runs and their outcomes (SBC scores, PPC coverage).

