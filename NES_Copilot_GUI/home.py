"""
Home page for the NES Co-Pilot Mission Control GUI.

This module provides the home page with dashboard and overview.
"""

import streamlit as st
import os
import time
from typing import Dict, Any, List

from gui.components import page_header
from gui.utils import list_run_directories, get_run_metadata
from gui.state import SessionState

def render():
    """
    Render the home page.
    """
    page_header(
        "NES Co-Pilot Mission Control",
        "Welcome to the NES Co-Pilot Mission Control interface. This tool allows you to configure, launch, monitor, and analyze NES Co-Pilot experiments."
    )
    
    # Create a dashboard layout with two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_recent_activity()
        
    with col2:
        render_quick_actions()
        render_system_info()

def render_recent_activity():
    """
    Render the recent activity section.
    """
    st.subheader("Recent Activity")
    
    # Get the results directory
    results_dir = SessionState.get_results_dir()
    
    if not results_dir or not os.path.exists(results_dir):
        st.warning("No results directory specified. Please set a results directory in the Results Viewer.")
        return
    
    # List run directories
    run_dirs = list_run_directories(results_dir)
    
    if not run_dirs:
        st.info("No runs found in the results directory.")
        return
    
    # Get metadata for each run
    runs_metadata = [get_run_metadata(run_dir) for run_dir in run_dirs]
    
    # Sort by timestamp (newest first)
    runs_metadata.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Display recent runs (limit to 5)
    recent_runs = runs_metadata[:5]
    
    for run in recent_runs:
        with st.expander(f"**{run['name']}** - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run['timestamp']))}"):
            # Display run details
            st.markdown(f"**Path:** {run['path']}")
            
            # Display available result types
            result_types = []
            if run['has_sbc_results']:
                result_types.append("SBC")
            if run['has_empirical_fit_results']:
                result_types.append("Empirical Fit")
            if run['has_ppc_results']:
                result_types.append("PPC")
                
            if result_types:
                st.markdown(f"**Available Results:** {', '.join(result_types)}")
            else:
                st.markdown("**Available Results:** None")
                
            # Add a button to view results
            if st.button("View Results", key=f"view_{run['name']}"):
                SessionState.set_selected_run_dir(run['path'])
                SessionState.set_page("Results Viewer")
                st.experimental_rerun()

def render_quick_actions():
    """
    Render the quick actions section.
    """
    st.subheader("Quick Actions")
    
    # Add buttons for common actions
    if st.button("Configure New Run"):
        SessionState.set_page("Configuration")
        st.experimental_rerun()
        
    if st.button("Launch Run"):
        SessionState.set_page("Run Launcher")
        st.experimental_rerun()
        
    if st.button("View Results"):
        SessionState.set_page("Results Viewer")
        st.experimental_rerun()

def render_system_info():
    """
    Render the system information section.
    """
    st.subheader("System Information")
    
    # Get the config directory
    config_dir = SessionState.config_dir
    
    # Count configuration files
    config_files = []
    if os.path.exists(config_dir):
        for root, dirs, files in os.walk(config_dir):
            for file in files:
                if file.endswith(('.yaml', '.yml', '.json')):
                    config_files.append(os.path.join(root, file))
    
    # Get the results directory
    results_dir = SessionState.get_results_dir()
    
    # Count run directories
    run_dirs = []
    if results_dir and os.path.exists(results_dir):
        run_dirs = list_run_directories(results_dir)
    
    # Display counts
    st.markdown(f"**Configuration Files:** {len(config_files)}")
    st.markdown(f"**Run Directories:** {len(run_dirs)}")
    
    # Display current run status
    run_status = SessionState.get_run_status()
    status_color = {
        "idle": "blue",
        "running": "orange",
        "completed": "green",
        "failed": "red"
    }.get(run_status, "gray")
    
    st.markdown(f"**Current Run Status:** <span style='color:{status_color}'>{run_status.upper()}</span>", unsafe_allow_html=True)
