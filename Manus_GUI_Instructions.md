Here's a detailed plan for Manus to create this GUI. We'll aim for a balance of beauty, functionality, and clarity.

---

**Project Plan: "NES Co-Pilot Mission Control" - GUI Development**

**1. Vision & Overall Goal:**

To create an intuitive, visually engaging, and powerful Graphical User Interface (GUI) for the "NES Co-Pilot" system. This "Mission Control" will allow users to:
*   Easily configure and launch experimental runs (NPE training, SBC, empirical fitting, PPCs).
*   Monitor the progress of ongoing runs in real-time.
*   Access, visualize, and compare results from different runs.
*   Manage datasets, model configurations, and trained NPE checkpoints.
*   Ultimately, facilitate a more interactive and insightful engagement with the NES modeling workflow.

**2. Core Design Principles for the GUI:**

*   **Clarity & Intuitiveness:** Users should be able to understand the workflow and perform common tasks with minimal learning curve.
*   **Visibility:** Provide clear dashboards and visualizations for every stage of the pipeline and its outputs. "Beautiful visibility" is key.
*   **Interactivity:** Allow users to drill down into results, compare runs, and perhaps even trigger specific analyses.
*   **Modularity (Reflecting Backend):** The GUI structure should ideally mirror the modularity of the NES Co-Pilot backend.
*   **Responsiveness:** The GUI should feel snappy and provide immediate feedback.
*   **Aesthetics:** A clean, modern, and professional look and feel. "Make something great here."

**3. Suggested Technology Stack for GUI:**

*   **Python-based Web Framework:** This is often the most flexible for data-intensive applications and allows easier integration with your Python backend.
    *   **Recommended: Streamlit or Dash (by Plotly).**
        *   **Streamlit:** Excellent for rapid development, very Python-centric, good for creating interactive data apps quickly. Might be easier to get started.
        *   **Dash:** More powerful and customizable, great for complex dashboards with rich interactivity, built on Flask, Plotly.js, and React.js. Steeper learning curve but more flexible for sophisticated UIs.
    *   *Alternative: Desktop application with PyQt or Kivy, but web-based is generally more accessible and shareable.*
*   **Visualization Libraries:** Plotly (for interactive Dash plots), Matplotlib/Seaborn (can be embedded in Streamlit/Dash), Altair.
*   **Backend Communication:** The GUI will need to interact with the NES Co-Pilot backend (e.g., to launch runs, fetch status, retrieve results). This could be via:
    *   Direct Python calls if the GUI and backend run in the same environment (simpler for Streamlit).
    *   A simple REST API (e.g., using FastAPI or Flask) if the backend needs to be more decoupled. (For initial development, direct calls might be easier).

**4. GUI Structure & Key Panes/Views:**

Imagine a multi-tabbed or multi-pane application.

**(A) Home/Dashboard Pane:**
*   **Overview:** Brief project description, links to documentation.
*   **Recent Activity:** List of recent runs (NPE, SBC, Fit, PPC) with their status (running, completed, failed), start/end times, and a quick link to their results.
*   **System Status:** (Optional) CPU/memory usage if relevant, number of available datasets/NPEs.
*   **Quick Launch:** Buttons for common workflows (e.g., "New SBC Run," "New Empirical Fit").

**(B) Configuration Management Pane:**
*   **View/Edit Configurations:**
    *   A file browser or dropdown to select master configuration files (`run_config.yaml`) and module-specific configuration files (e.g., `npe_config.yaml`).
    *   Display configurations in a structured, human-readable way (e.g., parsed YAML/JSON tree).
    *   Allow users to **edit parameters directly in the GUI** for selected fields (with validation based on expected types/ranges).
    *   Ability to "Save As" new configurations.
    *   Version history or comparison for configs (advanced).
*   **Parameter Explorer:**
    *   Visualize prior distributions for model parameters.
    *   Input fields for defining new prior sets.

**(C) Data Management Pane:**
*   **Empirical Datasets:**
    *   List available empirical datasets (e.g., Roberts et al.).
    *   Display basic info (number of subjects, trials).
    *   Button to trigger preprocessing (if not automatic).
    *   Visualize key aspects of a selected dataset (e.g., overall choice proportions, RT distributions).
*   **Stimulus Valence Data:**
    *   Interface to run/manage the `valence_processor.py` (e.g., input a list of sentences, see generated `v_i` scores).
    *   Store and display the canonical valence scores for known datasets.
*   **Trial Templates:**
    *   Manage and view generated `SUBJECT_TRIAL_STRUCTURE_TEMPLATE` files.

**(D) Model Training & Validation Pane:**
*   **Tab 1: NPE Training Setup & Launch:**
    *   Select/Create an NPE training configuration (priors, #sims, template, summary stats schema).
    *   Button to "Launch NPE Training Run."
    *   Display for selecting which `MVNESAgent` version/config to use for simulation.
*   **Tab 2: SBC Setup & Launch:**
    *   Select a trained NPE checkpoint to validate.
    *   Configure SBC parameters (#datasets, #posterior samples).
    *   Button to "Launch SBC Run."
*   **Run Monitor (Common to both):**
    *   Display for ongoing runs:
        *   Progress bars (e.g., from `tqdm` if backend can pipe this, or overall stage completion).
        *   Live tail of the log file.
        *   Status indicators (Initializing, Simulating, Training, Analyzing, Completed, Error).
    *   Ability to view historical run logs.

**(E) Results & Analysis Pane:**
*   **Browse Runs:** A table or list of all completed runs (NPE, SBC, Empirical Fit, PPC), filterable by type, date, status.
*   **Selected Run Detail View:**
    *   **NPE Training Results:**
        *   Link to saved NPE checkpoint and metadata.
        *   Plot of training loss (if saved from `sbi`).
    *   **SBC Results:**
        *   Display KS p-values for each parameter.
        *   Interactive ECDF plots and rank histograms for each parameter.
        *   Link to saved ranks and metadata.
    *   **Empirical Fitting Results:**
        *   Display table of fitted parameters (posterior means, medians, stds) for all subjects.
        *   Interactive plots of parameter distributions across subjects.
        *   Interactive scatter plots for key correlations (e.g., `v_norm` vs. framing, `alpha_gain` vs. RTs). User can select which parameters/metrics to plot.
    *   **PPC Results:**
        *   Display the PPC coverage summary table.
        *   Allow selection of a subject to view their individual visual PPC plots (histograms of simulated stats vs. observed value).
        *   Interactive heatmap of PPC coverage (stats vs. coverage percentage).
*   **Comparison View:**
    *   Ability to select two (or more) runs (e.g., SBC before and after a model tweak, or PPCs from different model fits) and view their key results side-by-side.

**(F) Model Zoo / Checkpoint Management Pane:**
*   List all saved NPE checkpoints.
*   Display metadata for each checkpoint (date, #sims, parameters, SBC validation status/scores).
*   Ability to "tag" or "promote" checkpoints (e.g., "SBC_Validated_5Param_v1").
*   Ability to delete old/unused checkpoints.

**5. Interaction with NES Co-Pilot Backend:**

*   **Launching Runs:** The GUI will need to trigger the `run_pipeline.py` script (or equivalent functions in `workflow_manager.py`) with the selected/generated configuration. This could be done via `subprocess` calls.
*   **Monitoring Progress:**
    *   The backend scripts need to log progress to a file that the GUI can read and display.
    *   Intermediate status updates could be written to a simple status file or database.
*   **Accessing Results:** The GUI needs to know the output directory structure to find and load saved CSVs, JSONs, plots, and NPE checkpoints. The `DataManager` and `CheckpointManager` in the backend will be key here.

**6. Step-wise Implementation Plan for Manus (Focus on Core Functionality First):**

1.  **Phase 1.0: Basic Structure & Read-Only Views:**
    *   Set up the chosen web framework (e.g., Streamlit for speed).
    *   **Home/Dashboard:** Static info, manually updated list of "important runs."
    *   **Configuration Viewer:** Ability to select and display existing YAML/JSON config files in a readable format (no editing yet).
    *   **Results Viewer (SBC Focus):**
        *   Ability to point to an *existing* SBC output directory (from your current script runs).
        *   Load and display `sbc_ks_test_results.json`.
        *   Load and display the saved SBC diagnostic plot image.
        *   Load and display `sbc_ranks.csv` as a table.
    *   *Goal: Get a basic GUI up that can display existing results.*

2.  **Phase 1.1: Launching a Pre-configured SBC Run:**
    *   **Configuration Management Pane:** Allow selection of a *master* config file that defines an SBC run.
    *   **Model Training & Validation Pane:** A simple "Launch SBC Run" button that:
        *   Calls `python run_pipeline.py --config <selected_master_config.yaml>` (assuming `run_pipeline.py` can orchestrate this).
        *   Displays the live log output from the backend script.
    *   **Run Monitor:** Basic display of active run and link to logs.

3.  **Phase 1.2: Displaying Empirical Fit & PPC Results:**
    *   Extend **Results & Analysis Pane** to:
        *   Load and display `empirical_fitting_results.csv`.
        *   Load and display `ppc_coverage_summary.csv`.
        *   Allow selection of subjects to view their individual saved PPC plot images.
    *   Add ability to launch an Empirical Fit run and a PPC run (similar to SBC launch).

4.  **Phase 1.3: Interactive Configuration Editing & Management:**
    *   Enhance **Configuration Management Pane** to allow GUI-based editing of key parameters in loaded configs and saving them as new config files.

5.  **Phase 1.X (Future):** More advanced features like run comparison, interactive plotting, data management interfaces, and eventually the agentic supervisor capabilities.

**Instructions for Manus:**

"Manus, thank you for your readiness to develop the 'NES Co-Pilot Mission Control' GUI. This will be a critical tool for our project.

**Primary Goal for Initial Development (Phase 1.0 - 1.2):**
Create a GUI that allows us to:
1.  View existing experimental configurations.
2.  Launch new, pre-configured experimental runs (initially focusing on a full SBC run as orchestrated by `run_pipeline.py` which calls the necessary modules).
3.  Monitor the progress of these runs (view logs).
4.  View and visualize the key results from completed runs (SBC diagnostics, empirical fit parameters, PPC coverage, and individual PPC plots).

**Technology Choice:**
*   Please use **Streamlit** for the initial development due to its rapid development capabilities and strong Python integration. We can consider Dash later if more complexity is needed.
*   For plotting, leverage Streamlit's native support for Matplotlib/Seaborn, and potentially Plotly for any interactive plots later.

**Key GUI Panes to Develop First:**
1.  **Configuration Viewer:** To load and display existing YAML/JSON config files.
2.  **Run Launcher:** A simple interface to select a master configuration file and trigger `run_pipeline.py`. Include a display area for live log output from the backend.
3.  **Results Dashboard:**
    *   SBC View: Display KS p-values, ECDF/rank plots (load saved images initially).
    *   Empirical Fit View: Display table of fitted parameters, scatter plot of `v_norm` vs. framing.
    *   PPC View: Display PPC coverage table, allow selection of subject to view their specific PPC image.

**Backend Interaction:**
*   The GUI will launch `run_pipeline.py` using `subprocess`.
*   `run_pipeline.py` (and the modules it calls) must write comprehensive logs to a file. The GUI will tail/display this log file.
*   The GUI will read results from the standardized output directories and files (CSVs, JSONs, PNGs) generated by the backend NES Co-Pilot modules.

**Focus for this 'Morning Sprint' (if applicable to GUI development):**
*   Let's aim for a basic Streamlit app with:
    *   A way to select and display the contents of our main SBC configuration YAML.
    *   A button to launch `run_pipeline.py` with that selected config.
    *   A section to display the contents of `sbc_ks_test_results.json` and the `sbc_manual_diagnostics_plot.png` from a specified output directory of a completed SBC run.

This initial focus will provide immense value by making it easier to manage runs and interpret results. We can iterate and add more features from there."

---

This plan gives Manus a clear starting point, focusing on leveraging the outputs of the backend pipeline you're building with Windsurf. The GUI becomes an interface *to* that pipeline.