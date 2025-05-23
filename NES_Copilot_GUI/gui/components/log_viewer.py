"""
Log viewer component for the NES Co-Pilot Mission Control GUI.

This component provides a reusable UI element for viewing log files.
"""

import streamlit as st
import os
import time
from typing import List, Optional

def display_log(log_file: str, max_lines: int = 100, auto_refresh: bool = True, refresh_interval: int = 5):
    """
    Display a log file with optional auto-refresh.
    
    Args:
        log_file: Path to the log file.
        max_lines: Maximum number of lines to display.
        auto_refresh: Whether to automatically refresh the log.
        refresh_interval: Refresh interval in seconds.
    """
    st.subheader("Log Viewer")
    
    # Check if the log file exists
    if not os.path.exists(log_file):
        st.error(f"Log file not found: {log_file}")
        return
    
    # Create a container for the log
    log_container = st.container()
    
    # Add auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh", value=auto_refresh)
    
    # Function to read and display the log
    def read_and_display_log():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Get the last N lines
            last_lines = lines[-max_lines:] if len(lines) > max_lines else lines
            
            # Display the log
            log_container.code(''.join(last_lines), language="bash")
            
            # Return the number of lines
            return len(lines)
        except Exception as e:
            log_container.error(f"Error reading log file: {str(e)}")
            return 0
    
    # Initial display
    num_lines = read_and_display_log()
    
    # Auto-refresh if enabled
    if auto_refresh:
        placeholder = st.empty()
        placeholder.info(f"Auto-refreshing every {refresh_interval} seconds...")
        
        # Use a session state variable to track the last update time
        if 'last_log_update' not in st.session_state:
            st.session_state.last_log_update = time.time()
            
        # Check if it's time to refresh
        current_time = time.time()
        if current_time - st.session_state.last_log_update >= refresh_interval:
            st.session_state.last_log_update = current_time
            num_lines = read_and_display_log()
            placeholder.info(f"Auto-refreshing every {refresh_interval} seconds... Last update: {time.strftime('%H:%M:%S')}")
    
    return num_lines

def log_tail(log_file: str, max_lines: int = 20):
    """
    Display the tail of a log file without auto-refresh.
    
    Args:
        log_file: Path to the log file.
        max_lines: Maximum number of lines to display.
    """
    if not os.path.exists(log_file):
        st.warning(f"Log file not found: {log_file}")
        return []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Get the last N lines
        last_lines = lines[-max_lines:] if len(lines) > max_lines else lines
        
        # Display the log
        st.code(''.join(last_lines), language="bash")
        
        return last_lines
    except Exception as e:
        st.error(f"Error reading log file: {str(e)}")
        return []
