"""
Configuration display component for the NES Co-Pilot Mission Control GUI.

This component provides a reusable UI element for displaying configuration files.
"""

import streamlit as st
import yaml
import json
import os
from typing import Dict, Any, Optional

def display_config(config_content: Dict[str, Any], title: str = "Configuration"):
    """
    Display a configuration in a formatted way.
    
    Args:
        config_content: Dictionary containing the configuration.
        title: Title to display above the configuration.
    """
    st.subheader(title)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Formatted", "Raw YAML"])
    
    with tab1:
        # Display the configuration in a formatted way
        display_config_recursive(config_content)
        
    with tab2:
        # Display the raw YAML
        st.code(yaml.dump(config_content, default_flow_style=False), language="yaml")

def display_config_recursive(config: Dict[str, Any], prefix: str = ""):
    """
    Recursively display a configuration with expandable sections.
    
    Args:
        config: Dictionary containing the configuration.
        prefix: Prefix for the current level of the configuration.
    """
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict) and value:
            # For dictionaries, create an expander
            with st.expander(f"**{key}**"):
                display_config_recursive(value, full_key)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # For lists of dictionaries, create an expander with numbered items
            with st.expander(f"**{key}** (List)"):
                for i, item in enumerate(value):
                    st.markdown(f"**Item {i+1}**")
                    display_config_recursive(item, f"{full_key}[{i}]")
        elif isinstance(value, list):
            # For simple lists, display as is
            st.markdown(f"**{key}**: {value}")
        else:
            # For simple values, display as is
            st.markdown(f"**{key}**: {value}")

def config_selector(config_dir: str, on_select=None):
    """
    Create a configuration file selector.
    
    Args:
        config_dir: Directory containing configuration files.
        on_select: Callback function to call when a configuration is selected.
    """
    st.subheader("Select Configuration")
    
    # Check if the config directory exists
    if not os.path.exists(config_dir):
        st.error(f"Configuration directory not found: {config_dir}")
        return
    
    # List all YAML and JSON files in the directory
    config_files = []
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith(('.yaml', '.yml', '.json')):
                config_files.append(os.path.join(root, file))
    
    # Sort the files
    config_files.sort()
    
    # Create a selectbox with relative paths for better readability
    relative_paths = [os.path.relpath(f, config_dir) for f in config_files]
    selected_index = st.selectbox(
        "Configuration File",
        range(len(relative_paths)),
        format_func=lambda i: relative_paths[i]
    )
    
    if selected_index is not None:
        selected_file = config_files[selected_index]
        
        # Display file info
        st.info(f"Selected: {relative_paths[selected_index]}")
        
        # Call the callback if provided
        if on_select:
            on_select(selected_file)
            
        return selected_file
    
    return None
