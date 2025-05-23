"""
State management for the NES Co-Pilot Mission Control GUI.

This module provides utilities for managing session state across the Streamlit application.
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Union
import os
import yaml
import json

class SessionState:
    """
    Manages session state for the NES Co-Pilot Mission Control GUI.
    
    This class provides a centralized way to manage and access session state
    variables across different pages and components of the application.
    """
    
    @staticmethod
    def initialize():
        """
        Initialize session state variables if they don't exist.
        """
        if 'initialized' not in st.session_state:
            # General app state
            st.session_state.initialized = True
            st.session_state.current_page = "home"
            
            # Configuration state
            st.session_state.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
            st.session_state.selected_config_file = None
            st.session_state.config_content = None
            
            # Run state
            st.session_state.run_process = None
            st.session_state.run_log_file = None
            st.session_state.run_status = "idle"  # idle, running, completed, failed
            
            # Results state
            st.session_state.results_dir = None
            st.session_state.selected_run_dir = None
            st.session_state.selected_result_type = None  # sbc, empirical_fit, ppc
            st.session_state.selected_subject = None
    
    @staticmethod
    def set_page(page_name: str):
        """
        Set the current page.
        
        Args:
            page_name: Name of the page to set as current.
        """
        st.session_state.current_page = page_name
    
    @staticmethod
    def get_page() -> str:
        """
        Get the current page.
        
        Returns:
            Name of the current page.
        """
        return st.session_state.current_page
    
    @staticmethod
    def set_config_file(config_file: str):
        """
        Set the selected configuration file.
        
        Args:
            config_file: Path to the configuration file.
        """
        st.session_state.selected_config_file = config_file
        
        # Load the config content
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    st.session_state.config_content = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    st.session_state.config_content = json.load(f)
                else:
                    st.session_state.config_content = f.read()
        except Exception as e:
            st.error(f"Error loading configuration file: {str(e)}")
            st.session_state.config_content = None
    
    @staticmethod
    def get_config_file() -> Optional[str]:
        """
        Get the selected configuration file.
        
        Returns:
            Path to the selected configuration file, or None if not set.
        """
        return st.session_state.selected_config_file
    
    @staticmethod
    def get_config_content() -> Optional[Dict[str, Any]]:
        """
        Get the content of the selected configuration file.
        
        Returns:
            Content of the selected configuration file, or None if not loaded.
        """
        return st.session_state.config_content
    
    @staticmethod
    def set_run_process(process):
        """
        Set the current run process.
        
        Args:
            process: Subprocess handle for the current run.
        """
        st.session_state.run_process = process
        st.session_state.run_status = "running"
    
    @staticmethod
    def get_run_process():
        """
        Get the current run process.
        
        Returns:
            Subprocess handle for the current run, or None if not running.
        """
        return st.session_state.run_process
    
    @staticmethod
    def set_run_log_file(log_file: str):
        """
        Set the log file for the current run.
        
        Args:
            log_file: Path to the log file.
        """
        st.session_state.run_log_file = log_file
    
    @staticmethod
    def get_run_log_file() -> Optional[str]:
        """
        Get the log file for the current run.
        
        Returns:
            Path to the log file, or None if not set.
        """
        return st.session_state.run_log_file
    
    @staticmethod
    def set_run_status(status: str):
        """
        Set the status of the current run.
        
        Args:
            status: Status of the run (idle, running, completed, failed).
        """
        st.session_state.run_status = status
    
    @staticmethod
    def get_run_status() -> str:
        """
        Get the status of the current run.
        
        Returns:
            Status of the run.
        """
        return st.session_state.run_status
    
    @staticmethod
    def set_results_dir(results_dir: str):
        """
        Set the results directory.
        
        Args:
            results_dir: Path to the results directory.
        """
        st.session_state.results_dir = results_dir
    
    @staticmethod
    def get_results_dir() -> Optional[str]:
        """
        Get the results directory.
        
        Returns:
            Path to the results directory, or None if not set.
        """
        return st.session_state.results_dir
    
    @staticmethod
    def set_selected_run_dir(run_dir: str):
        """
        Set the selected run directory.
        
        Args:
            run_dir: Path to the run directory.
        """
        st.session_state.selected_run_dir = run_dir
    
    @staticmethod
    def get_selected_run_dir() -> Optional[str]:
        """
        Get the selected run directory.
        
        Returns:
            Path to the selected run directory, or None if not set.
        """
        return st.session_state.selected_run_dir
    
    @staticmethod
    def set_selected_result_type(result_type: str):
        """
        Set the selected result type.
        
        Args:
            result_type: Type of result (sbc, empirical_fit, ppc).
        """
        st.session_state.selected_result_type = result_type
    
    @staticmethod
    def get_selected_result_type() -> Optional[str]:
        """
        Get the selected result type.
        
        Returns:
            Type of result, or None if not set.
        """
        return st.session_state.selected_result_type
    
    @staticmethod
    def set_selected_subject(subject: str):
        """
        Set the selected subject.
        
        Args:
            subject: Subject ID.
        """
        st.session_state.selected_subject = subject
    
    @staticmethod
    def get_selected_subject() -> Optional[str]:
        """
        Get the selected subject.
        
        Returns:
            Subject ID, or None if not set.
        """
        return st.session_state.selected_subject
