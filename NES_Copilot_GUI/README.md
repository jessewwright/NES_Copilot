# NES Copilot - User Guide

## Overview

NES Copilot is an agentic system for automating cognitive model development and validation, specifically designed for the Normative Executive System (NES). It provides a modular, extensible framework for data preparation, simulation, summary statistics calculation, neural posterior estimation (NPE) training, simulation-based calibration (SBC), empirical fitting, posterior predictive checks (PPC), and analysis.

## Installation

### Requirements

- Python 3.8+
- Required packages (see `requirements.txt`)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nes_copilot.git
cd nes_copilot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
nes_copilot/
├── nes_copilot/            # Main package
│   ├── __init__.py
│   ├── config_manager.py   # Configuration management
│   ├── data_manager.py     # Data management
│   ├── logging_manager.py  # Logging management
│   ├── workflow_manager.py # Workflow orchestration
│   ├── module_base.py      # Base class for modules
│   ├── data_prep/          # Data preparation module
│   ├── simulation/         # Simulation module
│   ├── summary_stats/      # Summary statistics module
│   ├── npe/                # NPE training module
│   ├── sbc/                # SBC module
│   ├── empirical_fit/      # Empirical fitting module
│   ├── ppc/                # PPC module
│   └── analysis/           # Analysis module
├── scripts/                # Utility scripts
│   └── run_pipeline.py     # Main entry point
├── tests/                  # Unit and integration tests
├── configs/                # Configuration files
└── examples/               # Example usage
```

## Configuration

NES Copilot uses a hierarchical configuration system with a master configuration file and module-specific configuration files. The master configuration file specifies global settings and paths to module-specific configuration files.

### Master Configuration

Example `master_config.yaml`:

```yaml
output_dir: /path/to/output
run_id: my_run
seed: 12345
device: cuda  # or cpu
data:
  roberts_data: /path/to/roberts_data.csv
run_modules:
  data_prep: true
  simulation: true
  summary_stats: true
  npe_training: true
  sbc: true
  empirical_fit: true
  ppc: true
  analysis: true
module_configs:
  data_prep: /path/to/data_prep_config.yaml
  simulation: /path/to/simulation_config.yaml
  summary_stats: /path/to/summary_stats_config.yaml
  npe: /path/to/npe_config.yaml
  sbc: /path/to/sbc_config.yaml
  empirical_fit: /path/to/empirical_fit_config.yaml
  ppc: /path/to/ppc_config.yaml
  analysis: /path/to/analysis_config.yaml
```

### Module Configuration

Example module configuration files can be found in the `configs/` directory.

## Usage

### Running the Pipeline

To run the complete pipeline:

```bash
python scripts/run_pipeline.py --config /path/to/master_config.yaml
```

### Using Individual Modules

You can also use individual modules in your own code:

```python
from nes_copilot.config_manager import ConfigManager
from nes_copilot.data_manager import DataManager
from nes_copilot.logging_manager import LoggingManager
from nes_copilot.data_prep import DataPrepModule

# Initialize managers
config_manager = ConfigManager('/path/to/master_config.yaml')
data_manager = DataManager(config_manager)
logging_manager = LoggingManager(data_manager)

# Initialize and run a module
data_prep_module = DataPrepModule(config_manager, data_manager, logging_manager)
results = data_prep_module.run()
```

## Modules

### Data Preparation

The data preparation module handles loading, filtering, and processing empirical data, as well as generating trial templates for simulations.

### Simulation

The simulation module runs the MVNESAgent to simulate trial data based on parameter sets and trial templates.

### Summary Statistics

The summary statistics module calculates summary statistics from trial data, such as probability of gambling, mean reaction time, and framing index.

### NPE Training

The NPE training module trains Neural Posterior Estimation models using simulation-based inference.

### SBC

The Simulation-Based Calibration module validates trained NPE models.

### Empirical Fitting

The empirical fitting module fits trained NPE models to empirical data.

### PPC

The Posterior Predictive Checks module validates fitted models by comparing simulated data to empirical data.

### Analysis

The analysis module provides tools for analyzing and visualizing results.

## Testing

To run the tests:

```bash
python -m unittest discover tests
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
