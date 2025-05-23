# NES Co-Pilot Mission Control - GUI Architecture Design

## Overview

The NES Co-Pilot Mission Control GUI is a Streamlit-based web application that provides an intuitive interface for configuring, launching, monitoring, and analyzing NES Co-Pilot experiments. This document outlines the architecture of the GUI, including its structure, components, data flow, and integration with the NES Co-Pilot backend.

## System Architecture

The GUI is designed as a separate module within the NES Co-Pilot repository, following a modular architecture that mirrors the backend's structure. It interfaces with the backend primarily through subprocess calls to `run_pipeline.py` and by reading/parsing output files.

### High-Level Architecture

```
NES Co-Pilot Repository
├── nes_copilot/           # Main backend package
├── scripts/               # Backend scripts including run_pipeline.py
├── configs/               # Configuration files
├── gui/                   # GUI module (new)
│   ├── app.py             # Main Streamlit application entry point
│   ├── components/        # Reusable UI components
│   ├── pages/             # Individual page modules
│   ├── utils/             # Utility functions for the GUI
│   └── state.py           # Session state management
└── tests/                 # Tests including GUI tests
```

### GUI Module Structure

The GUI module follows a modular design with clear separation of concerns:

1. **Main Application (`app.py`)**: Entry point that sets up the Streamlit app, navigation, and global state.

2. **Pages**: Individual functional areas of the application:
   - `pages/home.py`: Dashboard with overview and quick access
   - `pages/config_viewer.py`: Configuration file viewing and selection
   - `pages/run_launcher.py`: Interface for launching new runs
   - `pages/run_monitor.py`: Monitoring active runs and viewing logs
   - `pages/results_viewer.py`: Viewing and analyzing results from completed runs

3. **Components**: Reusable UI elements used across pages:
   - `components/config_display.py`: YAML/JSON configuration display
   - `components/log_viewer.py`: Log file viewer with auto-refresh
   - `components/plot_display.py`: Plot rendering utilities
   - `components/result_tables.py`: Tabular data display utilities
   - `components/navigation.py`: Navigation elements

4. **Utilities**: Helper functions for backend interaction:
   - `utils/file_utils.py`: File operations (reading configs, logs, results)
   - `utils/subprocess_utils.py`: Managing subprocess calls to run_pipeline.py
   - `utils/parsing_utils.py`: Parsing output files (JSON, CSV)
   - `utils/plotting_utils.py`: Creating and customizing plots

5. **State Management (`state.py`)**: Manages session state for the application

## Data Flow

The GUI interacts with the NES Co-Pilot backend in the following ways:

1. **Reading Configuration Files**:
   - Reads YAML configuration files from the `configs/` directory
   - Parses and displays them in a structured format

2. **Launching Runs**:
   - Constructs subprocess calls to `run_pipeline.py` with selected configuration
   - Captures and displays stdout/stderr from the subprocess

3. **Monitoring Runs**:
   - Reads and displays log files from the run's output directory
   - Implements auto-refresh to show real-time updates

4. **Viewing Results**:
   - Reads output files (CSV, JSON) from completed runs
   - Loads and displays saved plots (PNG)
   - Generates additional visualizations as needed

## Component Interactions

### Configuration Viewer
- **Inputs**: Path to configuration files in `configs/` directory
- **Processing**: Parses YAML files, structures for display
- **Outputs**: Formatted display of configuration parameters

### Run Launcher
- **Inputs**: Selected configuration file, optional run parameters
- **Processing**: Constructs subprocess command, launches run
- **Outputs**: Subprocess status, initial log output

### Run Monitor
- **Inputs**: Path to active run's log file
- **Processing**: Reads log file with periodic refresh
- **Outputs**: Formatted display of log contents, status indicators

### Results Viewer
- **Inputs**: Path to completed run's output directory
- **Processing**: Reads and parses output files, loads plots
- **Outputs**: Tabular data, plots, summary statistics

## Implementation Phases

The implementation will follow the phased approach outlined in the requirements:

### Phase 1.0: Basic Structure & Read-Only Views
- Setup Streamlit application structure
- Implement configuration file viewing
- Implement basic results viewing for SBC outputs

### Phase 1.1: Launching Pre-configured Runs
- Implement run launcher with configuration selection
- Implement basic run monitoring with log display

### Phase 1.2: Comprehensive Results Viewing
- Extend results viewer to handle empirical fit and PPC results
- Implement subject-specific result viewing

### Future Phases
- Interactive configuration editing
- Advanced visualization and comparison tools
- Integration with additional backend features

## Technical Considerations

### State Management
Streamlit's session state will be used to maintain application state across interactions, including:
- Currently selected configuration file
- Currently viewed run output directory
- Active subprocess handles
- UI state (selected tabs, expanded sections)

### Performance Optimization
- Caching of file reads and parsing operations
- Selective refreshing of UI components
- Efficient handling of large log files and result sets

### Error Handling
- Graceful handling of missing or malformed files
- Clear error messages for subprocess failures
- Fallback displays when expected outputs are not available

## User Interface Design

The UI will follow a clean, modern design with a focus on clarity and usability:

1. **Navigation**: Sidebar with main section navigation
2. **Dashboard**: Card-based layout with key information and quick actions
3. **Configuration**: Hierarchical display of configuration parameters
4. **Results**: Tabbed interface for different result types with interactive elements
5. **Monitoring**: Auto-refreshing log display with status indicators

## Conclusion

This architecture provides a solid foundation for the NES Co-Pilot Mission Control GUI, ensuring modularity, extensibility, and alignment with the backend system. The phased implementation approach allows for incremental delivery of value while maintaining a clear path to the full feature set.
