"""
Run launcher page for the NES Co-Pilot Mission Control GUI.

This module provides the run launcher page for launching new runs.
"""

import streamlit as st
import os
import time
from typing import Dict, Any, Optional

from gui.components import page_header, config_selector, log_tail
from gui.utils import launch_run, check_process_status
from gui.state import SessionState

def render():
    """
    Render the run launcher page.
    """
    page_header(
        "Run Launcher",
        "Launch new NES Co-Pilot experiments using selected configuration files."
    )
    
    # Create a two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Get the config directory from session state
        config_dir = SessionState.config_dir
        
        # Create a config selector
        selected_file = config_selector(config_dir, on_select=on_config_selected)
        
        # Output directory input
        st.subheader("Output Directory")
        output_dir = st.text_input(
            "Output Directory",
            value=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"),
            help="Directory where run outputs will be saved."
        )
        
        # Launch button
        if st.button("Launch Run", disabled=not selected_file):
            if selected_file:
                # Launch the run
                try:
                    process, log_file = launch_run(selected_file, output_dir)
                    
                    # Store the process and log file in session state
                    SessionState.set_run_process(process)
                    SessionState.set_run_log_file(log_file)
                    SessionState.set_run_status("running")
                    
                    # Set the results directory
                    SessionState.set_results_dir(output_dir)
                    
                    # Switch to the run monitor page
                    SessionState.set_page("Run Monitor")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error launching run: {str(e)}")
    
    with col2:
        # Get the selected config file from session state
        selected_config_file = SessionState.get_config_file()
        
        if selected_config_file:
            # Display the selected config file
            st.subheader("Selected Configuration")
            st.info(f"Selected: {os.path.basename(selected_config_file)}")
            
            # Display the config content
            config_content = SessionState.get_config_content()
            if config_content:
                # Display a summary of the config
                st.json(config_content)
        else:
            st.info("Select a configuration file to launch a run.")
            
        # Display the current run status
        run_status = SessionState.get_run_status()
        if run_status != "idle":
            st.subheader("Current Run Status")
            st.warning(f"A run is currently {run_status}. Please wait for it to complete or check the Run Monitor.")
            
            # Display the log tail if available
            log_file = SessionState.get_run_log_file()
            if log_file and os.path.exists(log_file):
                st.subheader("Recent Log Output")
                log_tail(log_file, max_lines=10)

def on_config_selected(file_path: str):
    """
    Callback function for when a configuration file is selected.
    
    Args:
        file_path: Path to the selected configuration file.
    """
    # Set the selected config file in session state
    SessionState.set_config_file(file_path)
