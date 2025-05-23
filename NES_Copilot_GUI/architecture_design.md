# NES Copilot System Architecture Design

## Overview

The NES Copilot is a modular, extensible software system designed to automate and streamline the iterative process of training, validating, fitting, and evaluating computational cognitive models, with a specific focus on the Normative Executive System (NES) and similar DDM-based architectures. This document outlines the system architecture, module interfaces, data flow, and configuration schema for Phase 1 implementation.

## System Architecture

The architecture follows a modular design pattern, with each major component encapsulated as a distinct module with well-defined interfaces. The system is orchestrated through a central pipeline engine that coordinates the execution of modules based on configuration files.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     NES Copilot System                          │
│                                                                 │
│  ┌───────────────┐                                              │
│  │Configuration  │                                              │
│  │   System      │                                              │
│  └───────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐ │
│  │ Core Pipeline │     │ Data Manager/ │     │ Logging &     │ │
│  │   Engine      │◄───►│   Versioner   │◄───►│ Reporting     │ │
│  └───────┬───────┘     └───────────────┘     └───────────────┘ │
│          │                                                      │
│          ▼                                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │ │
│  │  │  Data   │  │Simulation│  │Summary  │  │  NPE    │      │ │
│  │  │  Prep   │  │ Module   │  │Statistics│  │Training │      │ │
│  │  │ Module  │  │          │  │ Module  │  │ Module  │      │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │ │
│  │                                                           │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │ │
│  │  │  SBC    │  │Empirical │  │  PPC    │  │Analysis &│      │ │
│  │  │ Module  │  │ Fitting  │  │ Module  │  │Visualiza-│      │ │
│  │  │         │  │ Module   │  │         │  │tion     │      │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System

The configuration system is the foundation of the NES Copilot, enabling flexible, reproducible execution without code changes. It consists of:

**Master Configuration File (`config/master_config.yaml`):**
- Specifies which modules to run (e.g., `run_npe_training: true`)
- Global settings (seed, output directory, device)
- Paths to common data (Roberts raw data)
- Pointers to module-specific configuration files

**Module-Specific Configuration Files:**
- `config/data_prep_config.yaml`: Data preprocessing parameters, valence processor settings
- `config/simulator_config.yaml`: MVNESAgent parameters, fixed model parameters
- `config/summary_stats_config.yaml`: Summary statistics selection
- `config/npe_config.yaml`: Training parameters, prior definitions, NPE architecture
- `config/sbc_config.yaml`: SBC parameters, number of datasets, posterior samples
- `config/empirical_fit_config.yaml`: Fitting parameters, posterior samples
- `config/ppc_config.yaml`: PPC parameters, number of simulations
- `config/analysis_config.yaml`: Analysis and visualization settings

**Configuration Manager (`nes_copilot/config_manager.py`):**
- Loads and validates configuration files
- Provides a unified interface for accessing configuration parameters
- Ensures configuration consistency across modules
- Handles default values and parameter validation

### 2. Core Pipeline Engine

The core pipeline engine orchestrates the execution of modules based on the configuration.

**Workflow Manager (`nes_copilot/workflow_manager.py`):**
- Parses the master configuration to determine which modules to run
- Manages dependencies between modules (e.g., NPE training must complete before SBC)
- Handles execution flow and error handling
- Implements a simple Python-based orchestration system

**Data Manager/Versioner (`nes_copilot/data_manager.py`):**
- Manages input and output paths
- Creates timestamped output directories
- Tracks metadata for each run (configuration, seeds, software versions)
- Implements basic versioning through directory structure and metadata files

**Logging & Reporting Module (`nes_copilot/logging_manager.py`):**
- Centralizes logging across all modules
- Generates HTML/PDF reports summarizing runs
- Provides progress tracking and error reporting

### 3. Functional Modules

#### Data Preparation Module (`nes_copilot/data_prep/`)

**Empirical Data Loader & Preprocessor (`data_loader.py`):**
- Loads and preprocesses raw empirical data (Roberts et al. CSV)
- Applies filtering criteria (e.g., `trialType='target'`)
- Handles subject exclusion

**Stimulus Valence Processor (`valence_processor.py`):**
- Implements sentiment analysis using RoBERTa
- Generates valence scores for stimuli
- Applies rescaling rules from configuration

**Trial Template Generator (`trial_generator.py`):**
- Creates trial structure templates
- Incorporates valence scores and norm categories
- Saves templates to disk

#### Simulation Module (`nes_copilot/simulation/`)

**Simulator Interface (`simulator.py`):**
- Wraps the MVNESAgent class
- Provides a standardized interface for simulations
- Handles parameter sets, trial structures, and fixed parameters

**MVNESAgent Integration (`agent_mvnes.py`):**
- Integrates the existing MVNESAgent class
- Ensures compatibility with the simulator interface
- Handles valence scores and norm categories

#### Summary Statistics Module (`nes_copilot/summary_stats/`)

**Summary Statistics Calculator (`stats_calculator.py`):**
- Implements the calculation of summary statistics
- Integrates the existing stats_schema.py
- Supports different summary statistic sets

**Stats Schema Integration (`stats_schema.py`):**
- Integrates the existing stats schema
- Defines canonical summary statistics
- Provides validation logic

#### NPE Training Module (`nes_copilot/npe/`)

**NPE Trainer (`npe_trainer.py`):**
- Implements the training of Neural Posterior Estimation models
- Generates parameter sets from prior
- Calls simulation and summary statistics modules
- Trains sbi.inference.SNPE models

**NPE Checkpoint Manager (`checkpoint_manager.py`):**
- Saves and loads NPE checkpoints
- Manages metadata for checkpoints
- Ensures reproducibility

#### SBC Module (`nes_copilot/sbc/`)

**SBC Runner (`sbc_runner.py`):**
- Implements Simulation-Based Calibration
- Generates ground truth parameter sets
- Calculates ranks and performs KS tests
- Generates diagnostic plots

#### Empirical Fitting Module (`nes_copilot/empirical_fit/`)

**Empirical Fitter (`empirical_fitter.py`):**
- Fits NPE models to empirical data
- Calculates observed summary statistics
- Samples from posterior distributions
- Saves posterior summaries

#### PPC Module (`nes_copilot/ppc/`)

**PPC Runner (`ppc_runner.py`):**
- Implements Posterior Predictive Checks
- Draws parameter sets from fitted posteriors
- Compares simulated and observed statistics
- Generates coverage tables and plots

#### Analysis & Visualization Module (`nes_copilot/analysis/`)

**Analysis Runner (`analysis_runner.py`):**
- Calculates correlations and statistics
- Generates publication-quality plots
- Creates summary tables

**Visualization Tools (`visualization.py`):**
- Implements common visualization functions
- Ensures consistent styling across plots
- Supports various plot types (scatter, histogram, etc.)

## Data Flow

The data flow through the NES Copilot system follows a logical progression from data preparation to analysis:

1. **Data Preparation Flow:**
   - Raw empirical data → Preprocessed data → Trial templates
   - Stimulus text → Valence scores → Augmented trial templates

2. **NPE Training Flow:**
   - Prior definitions → Parameter sets
   - Parameter sets + Trial templates → Simulated data
   - Simulated data → Summary statistics
   - Parameter sets + Summary statistics → Trained NPE model

3. **SBC Flow:**
   - Trained NPE → Ground truth parameter sets
   - Ground truth parameters + Trial templates → Simulated data
   - Simulated data → Summary statistics
   - Summary statistics + Trained NPE → Posterior samples
   - Ground truth parameters + Posterior samples → SBC ranks
   - SBC ranks → Diagnostic plots and KS tests

4. **Empirical Fitting Flow:**
   - Trained NPE + Preprocessed empirical data → Observed summary statistics
   - Observed summary statistics + Trained NPE → Posterior samples
   - Posterior samples → Posterior summaries

5. **PPC Flow:**
   - Posterior summaries + Trial templates → Simulated data
   - Simulated data → Simulated summary statistics
   - Simulated statistics + Observed statistics → Coverage metrics
   - Coverage metrics → PPC plots and tables

6. **Analysis Flow:**
   - Posterior summaries + Coverage metrics → Correlation analyses
   - Analyses → Publication-quality plots and tables

## Module Interfaces

Each module in the NES Copilot system exposes a consistent interface to ensure interoperability:

### Common Interface Pattern

```python
class ModuleBase:
    def __init__(self, config_manager, data_manager, logger):
        self.config = config_manager
        self.data_manager = data_manager
        self.logger = logger
        
    def run(self, **kwargs):
        """
        Execute the module's main functionality.
        Returns a dictionary of results and output paths.
        """
        pass
        
    def validate_inputs(self, **kwargs):
        """
        Validate that all required inputs are available and correctly formatted.
        """
        pass
        
    def save_outputs(self, results):
        """
        Save module outputs to disk using the data manager.
        """
        pass
```

### Specific Module Interfaces

**Data Preparation Module:**
```python
class DataPrepModule(ModuleBase):
    def run(self, raw_data_path=None):
        """
        Preprocess data, generate trial templates, and calculate valence scores.
        
        Args:
            raw_data_path: Optional override for the raw data path in config.
            
        Returns:
            dict: Paths to preprocessed data, trial templates, and metadata.
        """
        pass
```

**Simulation Module:**
```python
class SimulationModule(ModuleBase):
    def run(self, parameters, trial_template, fixed_params=None):
        """
        Run simulations with the given parameters and trial template.
        
        Args:
            parameters: Parameter set (theta) for simulation.
            trial_template: Trial structure template.
            fixed_params: Optional fixed parameters for the model.
            
        Returns:
            DataFrame: Simulated choices and RTs.
        """
        pass
```

**Summary Statistics Module:**
```python
class SummaryStatsModule(ModuleBase):
    def run(self, trials_df, stat_keys=None):
        """
        Calculate summary statistics for the given trials.
        
        Args:
            trials_df: DataFrame of trials (simulated or empirical).
            stat_keys: Optional list of specific statistics to calculate.
            
        Returns:
            dict: Calculated summary statistics.
        """
        pass
```

**NPE Training Module:**
```python
class NPETrainingModule(ModuleBase):
    def run(self, trial_template=None):
        """
        Train an NPE model using the specified configuration.
        
        Args:
            trial_template: Optional override for the trial template.
            
        Returns:
            dict: Path to saved NPE checkpoint and training metadata.
        """
        pass
```

**SBC Module:**
```python
class SBCModule(ModuleBase):
    def run(self, npe_checkpoint=None, trial_template=None):
        """
        Perform Simulation-Based Calibration on a trained NPE.
        
        Args:
            npe_checkpoint: Optional override for the NPE checkpoint path.
            trial_template: Optional override for the trial template.
            
        Returns:
            dict: Paths to SBC results, plots, and metadata.
        """
        pass
```

**Empirical Fitting Module:**
```python
class EmpiricalFittingModule(ModuleBase):
    def run(self, npe_checkpoint=None, empirical_data=None):
        """
        Fit a trained NPE to empirical data.
        
        Args:
            npe_checkpoint: Optional override for the NPE checkpoint path.
            empirical_data: Optional override for the empirical data path.
            
        Returns:
            dict: Paths to fitting results and metadata.
        """
        pass
```

**PPC Module:**
```python
class PPCModule(ModuleBase):
    def run(self, npe_checkpoint=None, fitting_results=None, empirical_data=None):
        """
        Perform Posterior Predictive Checks.
        
        Args:
            npe_checkpoint: Optional override for the NPE checkpoint path.
            fitting_results: Optional override for the fitting results path.
            empirical_data: Optional override for the empirical data path.
            
        Returns:
            dict: Paths to PPC results, plots, and metadata.
        """
        pass
```

**Analysis & Visualization Module:**
```python
class AnalysisModule(ModuleBase):
    def run(self, fitting_results=None, ppc_results=None, empirical_data=None):
        """
        Perform analysis and generate visualizations.
        
        Args:
            fitting_results: Optional override for the fitting results path.
            ppc_results: Optional override for the PPC results path.
            empirical_data: Optional override for the empirical data path.
            
        Returns:
            dict: Paths to analysis results, plots, and metadata.
        """
        pass
```

## Configuration Schema

The configuration schema defines the structure and content of the configuration files:

### Master Configuration Schema

```yaml
# Master configuration file
output_dir: "/path/to/output"  # Base output directory
run_id: "run_20250522"  # Unique run identifier (default: timestamp)
seed: 12345  # Global random seed
device: "cpu"  # Computation device (cpu or cuda)

# Data paths
data:
  roberts_data: "/path/to/roberts_data.csv"
  
# Module execution flags
run_modules:
  data_prep: true
  npe_training: true
  sbc: true
  empirical_fit: true
  ppc: true
  analysis: true
  
# Module configuration paths
module_configs:
  data_prep: "config/data_prep_config.yaml"
  simulator: "config/simulator_config.yaml"
  summary_stats: "config/summary_stats_config.yaml"
  npe: "config/npe_config.yaml"
  sbc: "config/sbc_config.yaml"
  empirical_fit: "config/empirical_fit_config.yaml"
  ppc: "config/ppc_config.yaml"
  analysis: "config/analysis_config.yaml"
```

### Module-Specific Configuration Schemas

**Data Preparation Configuration:**
```yaml
# Data preparation configuration
filtering:
  trial_type: "target"
  exclude_subjects: []
  
valence_processor:
  model_name: "cardiffnlp/twitter-roberta-base-sentiment"
  rescaling:
    min_value: -1.0
    max_value: 1.0
    
trial_template:
  num_template_trials: 1000
  save_path: "trial_template.csv"
```

**Simulator Configuration:**
```yaml
# Simulator configuration
mvnes_agent:
  fixed_params:
    logit_z0: 0.0
    log_tau_norm: 0.0
  
  # Other MVNESAgent configuration parameters
  use_metacognitive_monitoring: false
```

**Summary Statistics Configuration:**
```yaml
# Summary statistics configuration
stat_set: "full_60"  # Options: "lean_25", "full_60"
custom_stats: []  # Additional custom statistics
```

**NPE Training Configuration:**
```yaml
# NPE training configuration
num_training_sims: 10000
batch_size: 100

prior:
  v_norm:
    distribution: "uniform"
    low: 0.0
    high: 5.0
  alpha_gain:
    distribution: "uniform"
    low: 0.0
    high: 1.0
  beta_val:
    distribution: "uniform"
    low: -1.0
    high: 1.0
  # Additional parameters...
  
npe_architecture:
  density_estimator: "maf"
  hidden_features: 50
  num_transforms: 5
```

**SBC Configuration:**
```yaml
# SBC configuration
num_sbc_datasets: 100
num_posterior_samples: 1000
```

**Empirical Fitting Configuration:**
```yaml
# Empirical fitting configuration
num_posterior_samples: 10000
save_full_posterior: false
```

**PPC Configuration:**
```yaml
# PPC configuration
num_ppc_simulations: 100
num_posterior_draws: 100
timeout_seconds: 300
```

**Analysis Configuration:**
```yaml
# Analysis configuration
correlations:
  compute:
    - ["v_norm", "framing_index"]
    - ["alpha_gain", "rt_metrics"]
  
plots:
  scatter:
    - ["v_norm", "framing_index"]
    - ["alpha_gain", "rt_metrics"]
  parameter_distributions: true
  ppc_coverage: true
```

## Directory Structure

The NES Copilot system follows a well-organized directory structure:

```
nes_copilot/
├── config/                      # Configuration files
│   ├── master_config.yaml
│   ├── data_prep_config.yaml
│   ├── simulator_config.yaml
│   ├── summary_stats_config.yaml
│   ├── npe_config.yaml
│   ├── sbc_config.yaml
│   ├── empirical_fit_config.yaml
│   ├── ppc_config.yaml
│   └── analysis_config.yaml
│
├── nes_copilot/                 # Core package
│   ├── __init__.py
│   ├── config_manager.py        # Configuration management
│   ├── workflow_manager.py      # Workflow orchestration
│   ├── data_manager.py          # Data and versioning management
│   ├── logging_manager.py       # Logging and reporting
│   │
│   ├── data_prep/               # Data preparation module
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── valence_processor.py
│   │   └── trial_generator.py
│   │
│   ├── simulation/              # Simulation module
│   │   ├── __init__.py
│   │   ├── simulator.py
│   │   └── agent_mvnes.py       # Existing MVNESAgent integration
│   │
│   ├── summary_stats/           # Summary statistics module
│   │   ├── __init__.py
│   │   ├── stats_calculator.py
│   │   └── stats_schema.py      # Existing stats schema integration
│   │
│   ├── npe/                     # NPE training module
│   │   ├── __init__.py
│   │   ├── npe_trainer.py
│   │   └── checkpoint_manager.py
│   │
│   ├── sbc/                     # SBC module
│   │   ├── __init__.py
│   │   └── sbc_runner.py
│   │
│   ├── empirical_fit/           # Empirical fitting module
│   │   ├── __init__.py
│   │   └── empirical_fitter.py
│   │
│   ├── ppc/                     # PPC module
│   │   ├── __init__.py
│   │   └── ppc_runner.py
│   │
│   └── analysis/                # Analysis and visualization module
│       ├── __init__.py
│       ├── analysis_runner.py
│       └── visualization.py
│
├── scripts/                     # Command-line scripts
│   ├── run_pipeline.py          # Main entry point
│   ├── run_data_prep.py         # Individual module runners
│   ├── run_npe_training.py
│   ├── run_sbc.py
│   ├── run_empirical_fit.py
│   ├── run_ppc.py
│   └── run_analysis.py
│
├── tests/                       # Unit and integration tests
│   ├── test_data_prep.py
│   ├── test_simulation.py
│   ├── test_summary_stats.py
│   ├── test_npe.py
│   ├── test_sbc.py
│   ├── test_empirical_fit.py
│   ├── test_ppc.py
│   └── test_analysis.py
│
├── examples/                    # Example configurations and usage
│   ├── example_configs/
│   │   ├── minimal_config.yaml
│   │   └── full_pipeline_config.yaml
│   └── notebooks/
│       ├── data_exploration.ipynb
│       └── results_analysis.ipynb
│
├── docs/                        # Documentation
│   ├── architecture.md
│   ├── configuration.md
│   ├── modules.md
│   └── examples.md
│
├── requirements.txt             # Package dependencies
├── setup.py                     # Package installation
└── README.md                    # Project overview
```

## Integration with Existing Code

The NES Copilot system is designed to integrate seamlessly with the existing codebase:

1. **agent_mvnes.py Integration:**
   - The existing MVNESAgent class will be incorporated directly into the simulation module
   - The simulator.py file will provide a wrapper around MVNESAgent that conforms to the module interface
   - All existing functionality will be preserved while adding the necessary integration points

2. **stats_schema.py Integration:**
   - The existing stats schema will be incorporated directly into the summary statistics module
   - The canonical list of summary statistics (ROBERTS_SUMMARY_STAT_KEYS) will be used as the default
   - The validation logic will be preserved and extended as needed

3. **fit_nes_to_roberts_data_sbc_focused.py Integration:**
   - The data loading and preparation logic will inform the data preparation module
   - The nes_sbi_simulator function will be adapted for the simulation module
   - The NPE training and SBC logic will be refactored into their respective modules

4. **run_ppc_nes_roberts.py Integration:**
   - The PPC logic will be refactored into the PPC module
   - The coverage calculation and visualization code will be adapted for the analysis module

5. **Sentiment Analysis Integration:**
   - The conceptual plan for get_trial_valence_scores will be implemented in the valence processor

## Conclusion

The NES Copilot system architecture provides a robust, modular, and extensible framework for automating cognitive model development and validation. By leveraging existing code and focusing on configuration-driven execution, the system will streamline the iterative process of training, validating, fitting, and evaluating computational cognitive models.

The design prioritizes reproducibility, modularity, and ease of use, while providing a solid foundation for future enhancements, including the agentic capabilities outlined in the project vision. The system will be implemented in phases, with Phase 1 focusing on the core modules and configuration system.
