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

def display_config_recursive(config: Dict[str, Any], prefix: str = "", level: int = 0):
    """
    Recursively display a configuration with proper indentation.
    
    Args:
        config: Dictionary containing the configuration.
        prefix: Prefix for the current level of the configuration.
        level: Current nesting level (used for indentation).
    """
    indent = "    " * level  # 4 spaces per level
    
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict) and value:
            # For dictionaries, display the key and recurse
            st.markdown(f"{indent}**{key}:**")
            display_config_recursive(value, full_key, level + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # For lists of dictionaries, display each item
            st.markdown(f"{indent}**{key}:**")
            for i, item in enumerate(value):
                st.markdown(f"{indent}  - Item {i+1}:")
                display_config_recursive(item, f"{full_key}[{i}]", level + 2)
        elif isinstance(value, list):
            # For simple lists, display as comma-separated values
            list_str = ", ".join(str(v) for v in value)
            st.markdown(f"{indent}**{key}:** [{list_str}]")
        else:
            # For simple values, display as is
            st.markdown(f"{indent}**{key}:** {value}")

def config_selector(config_dir: str, on_select=None):
    """
    Create a configuration file selector.
    
    Args:
        config_dir: Directory containing configuration files.
        on_select: Callback function to call when a configuration is selected.
    """
    st.subheader("Select Configuration")
    
    # Normalize the path to handle any inconsistencies
    config_dir = os.path.normpath(config_dir)
    
    # Check if the config directory exists
    if not os.path.exists(config_dir):
        st.error(f"Configuration directory not found: {config_dir}")
        return
    
    # List all YAML and JSON files in the directory
    config_files = []
    try:
        for file in os.listdir(config_dir):
            if file.endswith(('.yaml', '.yml', '.json')):
                file_path = os.path.join(config_dir, file)
                config_files.append(file_path)
        if not config_files:
            st.warning("No YAML or JSON configuration files found in the directory.")
            return
            
        # Create a selectbox with relative paths for better readability
        relative_paths = [os.path.relpath(f, config_dir) for f in config_files]
        selected_index = st.selectbox(
            "Configuration File",
            range(len(relative_paths)),
            format_func=lambda i: relative_paths[i]
        )
    except Exception as e:
        st.error(f"Error loading configuration files: {e}")
        return
    
    if selected_index is not None:
        selected_file = config_files[selected_index]
        
        # Display file info
        st.info(f"Selected: {relative_paths[selected_index]}")
        
        # Call the callback if provided
        if on_select:
            on_select(selected_file)
            
        return selected_file
    
    return None
