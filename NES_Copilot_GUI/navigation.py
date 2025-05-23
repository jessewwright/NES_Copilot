"""
Navigation component for the NES Co-Pilot Mission Control GUI.

This component provides reusable UI elements for navigation.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable

def sidebar_navigation(pages: Dict[str, Callable], title: str = "NES Co-Pilot Mission Control"):
    """
    Create a sidebar navigation menu.
    
    Args:
        pages: Dictionary mapping page names to page functions.
        title: Title for the sidebar.
    
    Returns:
        The selected page name.
    """
    st.sidebar.title(title)
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # Create the navigation menu
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # Add additional sidebar content
    with st.sidebar.expander("About"):
        st.markdown("""
        **NES Co-Pilot Mission Control** is a GUI for managing and monitoring NES Co-Pilot experiments.
        
        This tool allows you to:
        - Configure and launch experiments
        - Monitor ongoing runs
        - View and analyze results
        """)
    
    return selected_page

def create_tabs(tabs: Dict[str, Callable], default_tab: Optional[str] = None):
    """
    Create a tabbed interface.
    
    Args:
        tabs: Dictionary mapping tab names to tab functions.
        default_tab: Optional default tab to select.
    
    Returns:
        The selected tab name.
    """
    tab_names = list(tabs.keys())
    
    # Set the default tab index
    default_index = tab_names.index(default_tab) if default_tab in tab_names else 0
    
    # Create the tabs
    selected_tab = st.radio("", tab_names, index=default_index, horizontal=True)
    
    # Add a separator
    st.markdown("---")
    
    return selected_tab

def page_header(title: str, description: Optional[str] = None):
    """
    Create a page header with title and optional description.
    
    Args:
        title: Page title.
        description: Optional page description.
    """
    st.title(title)
    
    if description:
        st.markdown(description)
    
    # Add a separator
    st.markdown("---")
