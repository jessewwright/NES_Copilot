"""
File utilities for the NES Co-Pilot Mission Control GUI.

This module provides utilities for file operations such as reading configurations,
logs, and results files.
"""

import os
import yaml
import json
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import glob

def list_config_files(config_dir: str) -> List[str]:
    """
    List all configuration files in the specified directory.
    
    Args:
        config_dir: Directory containing configuration files.
        
    Returns:
        List of configuration file paths.
    """
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    yml_files = glob.glob(os.path.join(config_dir, "*.yml"))
    json_files = glob.glob(os.path.join(config_dir, "*.json"))
    
    return sorted(yaml_files + yml_files + json_files)

def read_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Dictionary containing the JSON data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the CSV data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    return pd.read_csv(file_path)

def read_log_file(file_path: str, max_lines: int = 100) -> List[str]:
    """
    Read the last N lines of a log file.
    
    Args:
        file_path: Path to the log file.
        max_lines: Maximum number of lines to read.
        
    Returns:
        List of log lines.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        return []
        
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]

def list_run_directories(base_dir: str) -> List[str]:
    """
    List all run directories in the specified base directory.
    
    Args:
        base_dir: Base directory containing run directories.
        
    Returns:
        List of run directory paths.
    """
    if not os.path.exists(base_dir):
        return []
        
    # Look for directories that contain the expected subdirectories
    run_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if this looks like a run directory (has models, data, plots, or logs subdirs)
            subdirs = os.listdir(item_path)
            if any(subdir in subdirs for subdir in ['models', 'data', 'plots', 'logs']):
                run_dirs.append(item_path)
                
    return sorted(run_dirs, key=os.path.getmtime, reverse=True)

def get_run_metadata(run_dir: str) -> Dict[str, Any]:
    """
    Get metadata for a run directory.
    
    Args:
        run_dir: Path to the run directory.
        
    Returns:
        Dictionary containing run metadata.
    """
    metadata = {
        'name': os.path.basename(run_dir),
        'path': run_dir,
        'timestamp': os.path.getmtime(run_dir),
        'has_models': os.path.exists(os.path.join(run_dir, 'models')),
        'has_data': os.path.exists(os.path.join(run_dir, 'data')),
        'has_plots': os.path.exists(os.path.join(run_dir, 'plots')),
        'has_logs': os.path.exists(os.path.join(run_dir, 'logs')),
    }
    
    # Try to find specific result files
    data_dir = os.path.join(run_dir, 'data')
    if os.path.exists(data_dir):
        # Check for SBC results
        sbc_ks_path = os.path.join(data_dir, 'sbc_ks_test_results.json')
        metadata['has_sbc_results'] = os.path.exists(sbc_ks_path)
        
        # Check for empirical fit results
        emp_fit_path = os.path.join(data_dir, 'empirical_fitting_results.csv')
        metadata['has_empirical_fit_results'] = os.path.exists(emp_fit_path)
        
        # Check for PPC results
        ppc_path = os.path.join(data_dir, 'ppc_coverage_summary.csv')
        metadata['has_ppc_results'] = os.path.exists(ppc_path)
    else:
        metadata['has_sbc_results'] = False
        metadata['has_empirical_fit_results'] = False
        metadata['has_ppc_results'] = False
        
    return metadata

def find_sbc_results(run_dir: str) -> Dict[str, str]:
    """
    Find SBC result files in a run directory.
    
    Args:
        run_dir: Path to the run directory.
        
    Returns:
        Dictionary mapping file types to file paths.
    """
    results = {}
    
    data_dir = os.path.join(run_dir, 'data')
    plots_dir = os.path.join(run_dir, 'plots')
    
    # Look for KS test results
    ks_test_path = os.path.join(data_dir, 'sbc_ks_test_results.json')
    if os.path.exists(ks_test_path):
        results['ks_test'] = ks_test_path
        
    # Look for ranks CSV
    ranks_path = os.path.join(data_dir, 'sbc_ranks.csv')
    if os.path.exists(ranks_path):
        results['ranks'] = ranks_path
        
    # Look for SBC plots
    if os.path.exists(plots_dir):
        # Look for ECDF plots
        ecdf_plots = glob.glob(os.path.join(plots_dir, 'ecdf_*.png'))
        if ecdf_plots:
            results['ecdf_plots'] = ecdf_plots
            
        # Look for rank histogram plots
        rank_plots = glob.glob(os.path.join(plots_dir, 'rank_histogram_*.png'))
        if rank_plots:
            results['rank_plots'] = rank_plots
            
        # Look for summary plot
        summary_plot = os.path.join(plots_dir, 'ks_test_summary.png')
        if os.path.exists(summary_plot):
            results['summary_plot'] = summary_plot
            
    return results

def find_empirical_fit_results(run_dir: str) -> Dict[str, str]:
    """
    Find empirical fit result files in a run directory.
    
    Args:
        run_dir: Path to the run directory.
        
    Returns:
        Dictionary mapping file types to file paths.
    """
    results = {}
    
    data_dir = os.path.join(run_dir, 'data')
    plots_dir = os.path.join(run_dir, 'plots')
    
    # Look for empirical fit results CSV
    fit_results_path = os.path.join(data_dir, 'empirical_fitting_results.csv')
    if os.path.exists(fit_results_path):
        results['fit_results'] = fit_results_path
        
    # Look for posterior samples
    posterior_samples_dir = os.path.join(data_dir, 'posterior_samples')
    if os.path.exists(posterior_samples_dir):
        results['posterior_samples_dir'] = posterior_samples_dir
        
    # Look for metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        results['metadata'] = metadata_path
        
    return results

def find_ppc_results(run_dir: str) -> Dict[str, str]:
    """
    Find PPC result files in a run directory.
    
    Args:
        run_dir: Path to the run directory.
        
    Returns:
        Dictionary mapping file types to file paths.
    """
    results = {}
    
    data_dir = os.path.join(run_dir, 'data')
    plots_dir = os.path.join(run_dir, 'plots')
    
    # Look for PPC coverage summary
    coverage_path = os.path.join(data_dir, 'ppc_coverage_summary.csv')
    if os.path.exists(coverage_path):
        results['coverage_summary'] = coverage_path
        
    # Look for detailed coverage
    detailed_path = os.path.join(data_dir, 'detailed_coverage.csv')
    if os.path.exists(detailed_path):
        results['detailed_coverage'] = detailed_path
        
    # Look for PPC plots
    if os.path.exists(plots_dir):
        # Look for coverage summary plot
        coverage_plot = os.path.join(plots_dir, 'coverage_summary.png')
        if os.path.exists(coverage_plot):
            results['coverage_plot'] = coverage_plot
            
        # Look for individual PPC plots
        ppc_plots = glob.glob(os.path.join(plots_dir, 'ppc_*.png'))
        if ppc_plots:
            results['ppc_plots'] = ppc_plots
            
    return results
