"""
Result tables component for the NES Co-Pilot Mission Control GUI.

This component provides reusable UI elements for displaying tabular data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

def display_dataframe(df: pd.DataFrame, title: Optional[str] = None, 
                     use_container_width: bool = True, height: Optional[int] = None):
    """
    Display a DataFrame with optional formatting.
    
    Args:
        df: DataFrame to display.
        title: Optional title for the DataFrame.
        use_container_width: Whether to use the full container width.
        height: Optional height for the DataFrame.
    """
    if title:
        st.subheader(title)
        
    st.dataframe(df, use_container_width=use_container_width, height=height)

def display_dataframe_with_download(df: pd.DataFrame, title: Optional[str] = None, 
                                   filename: str = "data.csv", 
                                   use_container_width: bool = True, 
                                   height: Optional[int] = None):
    """
    Display a DataFrame with a download button.
    
    Args:
        df: DataFrame to display.
        title: Optional title for the DataFrame.
        filename: Filename for the downloaded CSV.
        use_container_width: Whether to use the full container width.
        height: Optional height for the DataFrame.
    """
    if title:
        st.subheader(title)
        
    st.dataframe(df, use_container_width=use_container_width, height=height)
    
    # Create a download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def display_ks_test_results(ks_results: Dict[str, Any], title: str = "KS Test Results"):
    """
    Display KS test results in a formatted table.
    
    Args:
        ks_results: Dictionary containing KS test results.
        title: Title for the table.
    """
    st.subheader(title)
    
    # Create a DataFrame from the results
    df = pd.DataFrame({
        'Parameter': ks_results['parameters'],
        'KS Statistic': ks_results['ks_statistics'],
        'p-value': ks_results['p_values'],
        'Is Uniform': ks_results['is_uniform']
    })
    
    # Format the p-values
    df['p-value'] = df['p-value'].apply(lambda x: f"{x:.4f}")
    
    # Format the KS statistics
    df['KS Statistic'] = df['KS Statistic'].apply(lambda x: f"{x:.4f}")
    
    # Add styling based on uniformity
    def highlight_non_uniform(val):
        if val == False:
            return 'background-color: #ffcccc'
        return ''
    
    # Display the styled DataFrame
    st.dataframe(df.style.applymap(highlight_non_uniform, subset=['Is Uniform']), 
                use_container_width=True)

def display_empirical_fit_results(df: pd.DataFrame, title: str = "Empirical Fitting Results"):
    """
    Display empirical fitting results with formatting.
    
    Args:
        df: DataFrame containing empirical fitting results.
        title: Title for the table.
    """
    st.subheader(title)
    
    # Check if we have the expected columns
    if 'subject' in df.columns:
        # Display the DataFrame
        display_dataframe_with_download(df, filename="empirical_fit_results.csv")
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Calculate summary statistics
            summary = df[numeric_cols].describe()
            
            # Display the summary
            st.dataframe(summary, use_container_width=True)
    else:
        st.error("DataFrame does not contain expected columns.")

def display_ppc_coverage_summary(df: pd.DataFrame, title: str = "PPC Coverage Summary"):
    """
    Display PPC coverage summary with formatting.
    
    Args:
        df: DataFrame containing PPC coverage summary.
        title: Title for the table.
    """
    st.subheader(title)
    
    # Check if we have the expected columns
    if 'statistic' in df.columns and 'coverage_percentage' in df.columns:
        # Sort by coverage percentage
        df_sorted = df.sort_values('coverage_percentage', ascending=False)
        
        # Format the coverage percentage
        df_sorted['coverage_percentage'] = df_sorted['coverage_percentage'].apply(lambda x: f"{x:.2f}%")
        
        # Display the DataFrame
        display_dataframe_with_download(df_sorted, filename="ppc_coverage_summary.csv")
        
        # Calculate overall coverage
        if 'covered' in df.columns and 'total' in df.columns:
            total_covered = df['covered'].sum()
            total_total = df['total'].sum()
            overall_coverage = (total_covered / total_total) * 100 if total_total > 0 else 0
            
            st.info(f"Overall Coverage: {overall_coverage:.2f}% ({total_covered}/{total_total} statistics covered)")
    else:
        st.error("DataFrame does not contain expected columns.")

def display_subject_selector(subjects: List[str], on_select=None):
    """
    Create a subject selector.
    
    Args:
        subjects: List of subject IDs.
        on_select: Callback function to call when a subject is selected.
    """
    st.subheader("Select Subject")
    
    if not subjects:
        st.warning("No subjects available.")
        return None
    
    selected_subject = st.selectbox("Subject", subjects)
    
    if selected_subject and on_select:
        on_select(selected_subject)
        
    return selected_subject
