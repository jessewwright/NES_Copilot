"""
Main application entry point for the NES Co-Pilot Mission Control GUI.

This module sets up the Streamlit application, navigation, and global state.
"""

import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components and utilities
from gui.components import sidebar_navigation, page_header
from gui.state import SessionState
from gui.pages import home, config_viewer, run_launcher, run_monitor, results_viewer

# Set page config
st.set_page_config(
    page_title="NES Co-Pilot Mission Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """
    Main application entry point.
    """
    # Initialize session state
    SessionState.initialize()
    
    # Define pages
    pages = {
        "Home": home.render,
        "Configuration": config_viewer.render,
        "Run Launcher": run_launcher.render,
        "Run Monitor": run_monitor.render,
        "Results Viewer": results_viewer.render
    }
    
    # Create sidebar navigation
    selected_page = sidebar_navigation(pages)
    
    # Render the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
