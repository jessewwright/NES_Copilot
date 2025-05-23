"""
Results viewer page for the NES Co-Pilot Mission Control GUI.

This module provides the results viewer page for viewing and analyzing results from completed runs.
"""

import streamlit as st
import os
import time
from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt

from gui.components import (
    page_header, create_tabs, display_image, display_matplotlib_figure,
    display_multiple_images, display_ks_test_results, display_empirical_fit_results,
    display_ppc_coverage_summary, display_subject_selector
)
from gui.utils import (
    list_run_directories, get_run_metadata, find_sbc_results,
    find_empirical_fit_results, find_ppc_results, parse_sbc_ks_results,
    parse_sbc_ranks, parse_empirical_fit_results, parse_ppc_coverage,
    parse_detailed_ppc_coverage, get_subjects_from_detailed_coverage,
    get_subject_coverage_data, plot_ks_test_results, plot_rank_histogram,
    plot_parameter_distributions, plot_parameter_correlation, plot_ppc_coverage,
    plot_subject_ppc_coverage
)
from gui.state import SessionState

def render():
    """
    Render the results viewer page.
    """
    page_header(
        "Results Viewer",
        "View and analyze results from completed NES Co-Pilot experiments."
    )
    
    # Get the results directory
    results_dir = SessionState.get_results_dir()
    
    # Allow the user to set or change the results directory
    with st.expander("Set Results Directory"):
        new_results_dir = st.text_input(
            "Results Directory",
            value=results_dir if results_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"),
            help="Directory containing run outputs."
        )
        
        if st.button("Set Directory"):
            SessionState.set_results_dir(new_results_dir)
            st.success(f"Results directory set to: {new_results_dir}")
            st.rerun()
    
    # Check if the results directory exists
    if not results_dir or not os.path.exists(results_dir):
        st.warning("Results directory not found. Please set a valid directory.")
        return
    
    # Create a two-column layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # List run directories
        run_dirs = list_run_directories(results_dir)
        
        if not run_dirs:
            st.info("No runs found in the results directory.")
            return
        
        # Get metadata for each run
        runs_metadata = [get_run_metadata(run_dir) for run_dir in run_dirs]
        
        # Sort by timestamp (newest first)
        runs_metadata.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Create a selectbox for run selection
        st.subheader("Select Run")
        
        # Create a list of run names with timestamps
        run_options = [
            f"{run['name']} - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run['timestamp']))}"
            for run in runs_metadata
        ]
        
        selected_index = st.selectbox(
            "Run",
            range(len(run_options)),
            format_func=lambda i: run_options[i]
        )
        
        if selected_index is not None:
            selected_run = runs_metadata[selected_index]
            selected_run_dir = selected_run['path']
            
            # Set the selected run directory in session state
            SessionState.set_selected_run_dir(selected_run_dir)
            
            # Display available result types
            st.subheader("Available Results")
            
            result_types = []
            if selected_run['has_sbc_results']:
                result_types.append("SBC")
            if selected_run['has_empirical_fit_results']:
                result_types.append("Empirical Fit")
            if selected_run['has_ppc_results']:
                result_types.append("PPC")
                
            if not result_types:
                st.info("No results available for this run.")
                return
                
            # Create a radio button for result type selection
            selected_result_type = st.radio("Result Type", result_types)
            
            # Set the selected result type in session state
            SessionState.set_selected_result_type(selected_result_type)
    
    with col2:
        # Get the selected run directory and result type from session state
        selected_run_dir = SessionState.get_selected_run_dir()
        selected_result_type = SessionState.get_selected_result_type()
        
        if not selected_run_dir or not selected_result_type:
            st.info("Select a run and result type to view results.")
            return
        
        # Display the selected result type
        if selected_result_type == "SBC":
            render_sbc_results(selected_run_dir)
        elif selected_result_type == "Empirical Fit":
            render_empirical_fit_results(selected_run_dir)
        elif selected_result_type == "PPC":
            render_ppc_results(selected_run_dir)

def render_sbc_results(run_dir: str):
    """
    Render SBC results.
    
    Args:
        run_dir: Path to the run directory.
    """
    st.subheader("SBC Results")
    
    # Find SBC result files
    sbc_results = find_sbc_results(run_dir)
    
    if not sbc_results:
        st.warning("No SBC results found in this run directory.")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Summary", "Detailed"])
    
    with tab1:
        # Display KS test results if available
        if 'ks_test' in sbc_results:
            try:
                ks_results = parse_sbc_ks_results(sbc_results['ks_test'])
                display_ks_test_results(ks_results)
                
                # Create and display a plot of the KS test results
                fig = plot_ks_test_results(ks_results)
                display_matplotlib_figure(fig, caption="KS Test p-values")
            except Exception as e:
                st.error(f"Error parsing KS test results: {str(e)}")
        
        # Display summary plot if available
        if 'summary_plot' in sbc_results:
            try:
                display_image(sbc_results['summary_plot'], caption="SBC Diagnostic Plot")
            except Exception as e:
                st.error(f"Error displaying summary plot: {str(e)}")
    
    with tab2:
        # Display rank histograms if available
        if 'ranks' in sbc_results:
            try:
                ranks_df = parse_sbc_ranks(sbc_results['ranks'])
                
                # Get parameter columns (exclude any non-parameter columns)
                param_columns = [col for col in ranks_df.columns if col not in ['index', 'run_id']]
                
                if param_columns:
                    # Create a selectbox for parameter selection
                    selected_param = st.selectbox("Parameter", param_columns)
                    
                    # Create and display a rank histogram for the selected parameter
                    fig = plot_rank_histogram(ranks_df, selected_param)
                    display_matplotlib_figure(fig, caption=f"Rank Histogram for {selected_param}")
                    
                    # Display the ranks as a table
                    st.subheader("Ranks Data")
                    st.dataframe(ranks_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error parsing ranks data: {str(e)}")
        
        # Display ECDF plots if available
        if 'ecdf_plots' in sbc_results:
            try:
                st.subheader("ECDF Plots")
                display_multiple_images(sbc_results['ecdf_plots'], columns=2)
            except Exception as e:
                st.error(f"Error displaying ECDF plots: {str(e)}")
        
        # Display rank histogram plots if available
        if 'rank_plots' in sbc_results:
            try:
                st.subheader("Rank Histogram Plots")
                display_multiple_images(sbc_results['rank_plots'], columns=2)
            except Exception as e:
                st.error(f"Error displaying rank histogram plots: {str(e)}")

def render_empirical_fit_results(run_dir: str):
    """
    Render empirical fitting results.
    
    Args:
        run_dir: Path to the run directory.
    """
    st.subheader("Empirical Fitting Results")
    
    # Find empirical fit result files
    fit_results = find_empirical_fit_results(run_dir)
    
    if not fit_results:
        st.warning("No empirical fitting results found in this run directory.")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Summary", "Detailed"])
    
    with tab1:
        # Display fit results if available
        if 'fit_results' in fit_results:
            try:
                fit_df = parse_empirical_fit_results(fit_results['fit_results'])
                display_empirical_fit_results(fit_df)
            except Exception as e:
                st.error(f"Error parsing empirical fit results: {str(e)}")
    
    with tab2:
        # Display parameter distributions and correlations if available
        if 'fit_results' in fit_results:
            try:
                fit_df = parse_empirical_fit_results(fit_results['fit_results'])
                
                # Get parameter columns (exclude subject column)
                param_columns = [col for col in fit_df.columns if col != 'subject']
                
                if param_columns:
                    # Create a selectbox for parameter selection
                    st.subheader("Parameter Distribution")
                    selected_param = st.selectbox("Parameter", param_columns)
                    
                    # Create and display a histogram for the selected parameter
                    fig = plot_parameter_distributions(fit_df, selected_param)
                    display_matplotlib_figure(fig, caption=f"Distribution of {selected_param}")
                    
                    # Create a section for parameter correlations
                    st.subheader("Parameter Correlations")
                    
                    # Create selectboxes for parameter selection
                    col1, col2 = st.columns(2)
                    with col1:
                        param_x = st.selectbox("Parameter X", param_columns, key="param_x")
                    with col2:
                        param_y = st.selectbox("Parameter Y", param_columns, key="param_y")
                    
                    # Create and display a scatter plot for the selected parameters
                    fig = plot_parameter_correlation(fit_df, param_x, param_y)
                    display_matplotlib_figure(fig, caption=f"Correlation between {param_x} and {param_y}")
            except Exception as e:
                st.error(f"Error analyzing empirical fit results: {str(e)}")

def render_ppc_results(run_dir: str):
    """
    Render PPC results.
    
    Args:
        run_dir: Path to the run directory.
    """
    st.subheader("PPC Results")
    
    # Find PPC result files
    ppc_results = find_ppc_results(run_dir)
    
    if not ppc_results:
        st.warning("No PPC results found in this run directory.")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Summary", "Subject-Specific"])
    
    with tab1:
        # Display coverage summary if available
        if 'coverage_summary' in ppc_results:
            try:
                coverage_df = parse_ppc_coverage(ppc_results['coverage_summary'])
                display_ppc_coverage_summary(coverage_df)
                
                # Create and display a plot of the coverage summary
                fig = plot_ppc_coverage(coverage_df)
                display_matplotlib_figure(fig, caption="PPC Coverage Summary")
            except Exception as e:
                st.error(f"Error parsing PPC coverage summary: {str(e)}")
        
        # Display coverage plot if available
        if 'coverage_plot' in ppc_results:
            try:
                display_image(ppc_results['coverage_plot'], caption="PPC Coverage Plot")
            except Exception as e:
                st.error(f"Error displaying coverage plot: {str(e)}")
    
    with tab2:
        # Display subject-specific PPC results if available
        if 'detailed_coverage' in ppc_results:
            try:
                # Get list of subjects
                subjects = get_subjects_from_detailed_coverage(ppc_results['detailed_coverage'])
                
                if subjects:
                    # Create a subject selector
                    selected_subject = display_subject_selector(subjects, on_select=lambda s: SessionState.set_selected_subject(s))
                    
                    if selected_subject:
                        # Get the subject's coverage data
                        subject_data = get_subject_coverage_data(ppc_results['detailed_coverage'], selected_subject)
                        
                        # Display the subject's coverage data
                        st.subheader(f"Coverage for Subject {selected_subject}")
                        st.dataframe(subject_data, use_container_width=True)
                        
                        # Get the statistics for this subject
                        statistics = subject_data['statistic'].unique().tolist() if 'statistic' in subject_data.columns else []
                        
                        if statistics:
                            # Create a selectbox for statistic selection
                            selected_stat = st.selectbox("Statistic", statistics)
                            
                            # Create and display a plot for the selected statistic
                            fig = plot_subject_ppc_coverage(subject_data, selected_stat)
                            display_matplotlib_figure(fig, caption=f"PPC for {selected_stat}")
                else:
                    st.info("No subject-specific data available.")
            except Exception as e:
                st.error(f"Error analyzing subject-specific PPC results: {str(e)}")
        
        # Display individual PPC plots if available
        if 'ppc_plots' in ppc_results:
            try:
                st.subheader("PPC Plots")
                display_multiple_images(ppc_results['ppc_plots'], columns=2)
            except Exception as e:
                st.error(f"Error displaying PPC plots: {str(e)}")
