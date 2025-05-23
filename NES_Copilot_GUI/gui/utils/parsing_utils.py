"""
Parsing utilities for the NES Co-Pilot Mission Control GUI.

This module provides utilities for parsing output files from the NES Co-Pilot backend.
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union

def parse_sbc_ks_results(file_path: str) -> Dict[str, Any]:
    """
    Parse SBC KS test results from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Dictionary containing parsed KS test results.
    """
    with open(file_path, 'r') as f:
        ks_results = json.load(f)
        
    # Process the results for display
    parsed_results = {
        'parameters': [],
        'ks_statistics': [],
        'p_values': [],
        'is_uniform': []
    }
    
    for param_name, param_results in ks_results.items():
        parsed_results['parameters'].append(param_name)
        parsed_results['ks_statistics'].append(param_results.get('ks_statistic', np.nan))
        parsed_results['p_values'].append(param_results.get('p_value', np.nan))
        parsed_results['is_uniform'].append(param_results.get('is_uniform', False))
        
    return parsed_results

def parse_sbc_ranks(file_path: str) -> pd.DataFrame:
    """
    Parse SBC ranks from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the ranks.
    """
    return pd.read_csv(file_path)

def parse_empirical_fit_results(file_path: str) -> pd.DataFrame:
    """
    Parse empirical fitting results from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the empirical fitting results.
    """
    df = pd.read_csv(file_path)
    
    # Pivot the data for easier display if needed
    if 'subject' in df.columns and 'parameter' in df.columns:
        # Check if we have the expected columns
        value_cols = [col for col in df.columns if col not in ['subject', 'parameter']]
        if value_cols:
            # Create a pivot table with subjects as rows and parameters as columns
            pivot_df = df.pivot(index='subject', columns='parameter')
            
            # Flatten the column multi-index
            pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
            
            # Reset the index to make 'subject' a regular column
            pivot_df = pivot_df.reset_index()
            
            return pivot_df
    
    # If we can't pivot or don't need to, return the original DataFrame
    return df

def parse_ppc_coverage(file_path: str) -> pd.DataFrame:
    """
    Parse PPC coverage summary from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the PPC coverage summary.
    """
    df = pd.read_csv(file_path)
    
    # Sort by coverage percentage if it exists
    if 'coverage_percentage' in df.columns:
        df = df.sort_values('coverage_percentage', ascending=False)
        
    return df

def parse_detailed_ppc_coverage(file_path: str) -> pd.DataFrame:
    """
    Parse detailed PPC coverage from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the detailed PPC coverage.
    """
    df = pd.read_csv(file_path)
    
    # Group by subject and calculate summary statistics
    if 'subject' in df.columns and 'covered' in df.columns:
        subject_summary = df.groupby('subject')['covered'].agg(['mean', 'sum', 'count']).reset_index()
        subject_summary = subject_summary.rename(columns={
            'mean': 'coverage_percentage',
            'sum': 'num_covered',
            'count': 'num_total'
        })
        subject_summary['coverage_percentage'] = subject_summary['coverage_percentage'] * 100
        
        return subject_summary
        
    return df

def get_subjects_from_detailed_coverage(file_path: str) -> List[str]:
    """
    Get list of subjects from detailed PPC coverage file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        List of subject IDs.
    """
    df = pd.read_csv(file_path)
    
    if 'subject' in df.columns:
        return sorted(df['subject'].unique().tolist())
        
    return []

def get_subject_coverage_data(file_path: str, subject: str) -> pd.DataFrame:
    """
    Get coverage data for a specific subject.
    
    Args:
        file_path: Path to the CSV file.
        subject: Subject ID.
        
    Returns:
        DataFrame containing the subject's coverage data.
    """
    df = pd.read_csv(file_path)
    
    if 'subject' in df.columns:
        return df[df['subject'] == subject]
        
    return pd.DataFrame()
