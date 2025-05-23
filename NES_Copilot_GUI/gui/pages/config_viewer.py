"""
Configuration viewer page for the NES Co-Pilot Mission Control GUI.

This module provides the configuration viewer page for viewing and selecting configuration files.
"""

import streamlit as st
import os
import yaml
from typing import Dict, Any, Optional

from gui.components import page_header, config_selector, display_config
from gui.utils import read_yaml_config, read_json_file
from gui.state import SessionState

def render():
    """
    Render the configuration viewer page.
    """
    page_header(
        "Configuration Viewer",
        "View and select configuration files for NES Co-Pilot experiments."
    )
    
    # Create a two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Get the config directory from session state
        config_dir = SessionState.get_config_dir()
        
        # Create a config selector
        selected_file = config_selector(config_dir, on_select=on_config_selected)
        
        # Add a button to refresh the config directory
        if st.button("Refresh Config Directory"):
            st.rerun()
    
    with col2:
        # Get the selected config file from session state
        config_content = SessionState.get_config_content()
        
        if config_content:
            # Display the config
            display_config(config_content, title="Configuration Details")
        else:
            st.info("Select a configuration file to view its details.")

def on_config_selected(file_path: str):
    """
    Callback function for when a configuration file is selected.
    
    Args:
        file_path: Path to the selected configuration file.
    """
    # Set the selected config file in session state
    SessionState.set_config_file(file_path)
