"""
README for NES Co-Pilot Mission Control GUI

This document provides instructions for installing, running, and using the NES Co-Pilot Mission Control GUI.
"""

# NES Co-Pilot Mission Control GUI

## Overview

The NES Co-Pilot Mission Control GUI is a Streamlit-based web application that provides an intuitive interface for configuring, launching, monitoring, and analyzing NES Co-Pilot experiments. It allows users to:

- View and select configuration files
- Launch new runs with selected configurations
- Monitor active runs and view logs in real-time
- View and analyze results from completed runs (SBC, Empirical Fit, PPC)

## Installation

### Requirements

- Python 3.8+
- Streamlit
- Matplotlib
- Pandas
- PyYAML
- Pillow

### Setup

1. The GUI is part of the NES Co-Pilot repository. Make sure you have the repository cloned and set up.

2. Install the required dependencies:

```bash
pip install streamlit matplotlib pandas pyyaml pillow
```

## Running the GUI

From the root directory of the NES Co-Pilot repository, run:

```bash
streamlit run gui/app.py
```

This will start the Streamlit server and open the GUI in your default web browser.

## Using the GUI

### Home Page

The home page provides an overview of the system, recent activity, and quick access to common actions.

### Configuration Viewer

The Configuration Viewer allows you to:
- Browse and select configuration files from the `configs/` directory
- View the contents of selected configuration files in a structured format

### Run Launcher

The Run Launcher allows you to:
- Select a configuration file to use for a new run
- Specify an output directory for the run results
- Launch a new run with the selected configuration

### Run Monitor

The Run Monitor allows you to:
- View the status of the current run
- Monitor the log output in real-time
- Terminate a running process if needed
- Navigate to the Results Viewer when the run completes

### Results Viewer

The Results Viewer allows you to:
- Browse and select run directories
- View different types of results (SBC, Empirical Fit, PPC)
- Analyze results with interactive visualizations
- View subject-specific results for empirical fits and PPCs

## Directory Structure

```
gui/
├── app.py                 # Main application entry point
├── state.py               # Session state management
├── components/            # Reusable UI components
├── pages/                 # Individual page modules
├── utils/                 # Utility functions
└── architecture_design.md # Detailed architecture documentation
```

## Configuration

The GUI expects configuration files to be located in the `configs/` directory of the NES Co-Pilot repository. Results are expected to follow the standard output directory structure with `models/`, `data/`, `plots/`, and `logs/` subdirectories.

## Troubleshooting

- If the GUI fails to start, ensure that Streamlit is installed and that you're running the command from the correct directory.
- If configuration files are not showing up, check that they exist in the `configs/` directory and have the correct file extensions (.yaml, .yml, or .json).
- If run results are not showing up, ensure that the output directory follows the expected structure and contains the necessary files.

## Support

For issues or questions, please contact the NES Co-Pilot team.
