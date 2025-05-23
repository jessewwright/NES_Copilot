"""
Run monitor page for the NES Co-Pilot Mission Control GUI.

This module provides the run monitor page for monitoring active runs.
"""

import streamlit as st
import os
import time
from typing import Dict, Any, Optional

from gui.components import page_header, display_log
from gui.utils import check_process_status, terminate_process
from gui.state import SessionState

def render():
    """
    Render the run monitor page.
    """
    page_header(
        "Run Monitor",
        "Monitor active NES Co-Pilot experiments and view logs."
    )
    
    # Get the current run process and log file from session state
    process = SessionState.get_run_process()
    log_file = SessionState.get_run_log_file()
    
    # Check if there's an active run
    if process is None:
        st.info("No active run to monitor. Use the Run Launcher to start a new run.")
        return
    
    # Check the process status
    status = check_process_status(process)
    SessionState.set_run_status(status)
    
    # Display the status
    status_color = {
        "running": "orange",
        "completed": "green",
        "failed": "red"
    }.get(status, "gray")
    
    st.markdown(f"### Run Status: <span style='color:{status_color}'>{status.upper()}</span>", unsafe_allow_html=True)
    
    # Create a two-column layout for controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Add a button to terminate the process if it's running
        if status == "running":
            if st.button("Terminate Run"):
                if terminate_process(process):
                    st.success("Run terminated successfully.")
                    SessionState.set_run_status("failed")
                    st.experimental_rerun()
                else:
                    st.error("Failed to terminate run.")
    
    with col2:
        # Add a button to view results if the run is completed
        if status in ["completed", "failed"]:
            if st.button("View Results"):
                SessionState.set_page("Results Viewer")
                st.experimental_rerun()
    
    # Display the log file if available
    if log_file and os.path.exists(log_file):
        # Display the log with auto-refresh
        display_log(log_file, max_lines=100, auto_refresh=(status == "running"), refresh_interval=5)
    else:
        st.warning("Log file not found or not specified.")
