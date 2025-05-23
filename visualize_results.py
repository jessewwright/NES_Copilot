#!/usr/bin/env python3
"""
Visualization script for NPE analysis results.

This script generates various plots to visualize the results of the NPE analysis,
including posterior distributions, parameter recovery, and model diagnostics.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any

def plot_posterior_distributions(posterior_samples: pd.DataFrame, 
                               output_dir: Path) -> None:
    """Plot posterior distributions for all parameters."""
    n_params = len(posterior_samples.columns)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, param in enumerate(posterior_samples.columns):
        ax = axes[i]
        sns.histplot(posterior_samples[param], ax=ax, kde=True)
        ax.set_title(f'Posterior: {param}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plot_path = output_dir / 'posterior_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved posterior distributions plot to: {plot_path}")

def plot_parameter_recovery(true_params: Dict[str, float], 
                          posterior_samples: pd.DataFrame,
                          output_dir: Path) -> None:
    """Plot parameter recovery (true vs. posterior estimates)."""
    n_params = len(true_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, (param, true_val) in enumerate(true_params.items()):
        if param in posterior_samples.columns:
            ax = axes[i]
            sns.histplot(posterior_samples[param], ax=ax, kde=True)
            ax.axvline(true_val, color='r', linestyle='--', label='True Value')
            ax.set_title(f'Parameter: {param}')
            ax.set_xlabel('Parameter Value')
            ax.legend()
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plot_path = output_dir / 'parameter_recovery.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter recovery plot to: {plot_path}")

def plot_training_curve(losses: np.ndarray, output_dir: Path) -> None:
    """Plot the training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plot_path = output_dir / 'training_curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curve plot to: {plot_path}")

def plot_corner_plot(posterior_samples: pd.DataFrame, output_dir: Path) -> None:
    """Generate a corner plot of the posterior samples."""
    try:
        import corner
        
        # Convert to numpy array for corner plot
        samples = posterior_samples.values
        
        # Create corner plot
        fig = corner.corner(
            samples,
            labels=posterior_samples.columns,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        plot_path = output_dir / 'corner_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved corner plot to: {plot_path}")
        
    except ImportError:
        print("corner package not found. Install with: pip install corner")

def main(results_dir: str):
    """Main function to generate visualizations."""
    # Set up output directory
    results_dir = Path(results_dir)
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Set plot style
    sns.set_theme(style='whitegrid')
    
    # Load results
    posterior_path = results_dir / 'posterior_samples.csv'
    if posterior_path.exists():
        posterior_samples = pd.read_csv(posterior_path)
        
        # Plot posterior distributions
        plot_posterior_distributions(posterior_samples, plots_dir)
        
        # Plot corner plot (if corner package is available)
        plot_corner_plot(posterior_samples, plots_dir)
    
    # Load training losses if available
    losses_path = results_dir / 'training_losses.npy'
    if losses_path.exists():
        losses = np.load(losses_path)
        plot_training_curve(losses, plots_dir)
    
    print(f"\nAll visualizations saved to: {plots_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for NPE analysis.')
    parser.add_argument('results_dir', type=str, 
                       help='Path to the directory containing the NPE results')
    
    args = parser.parse_args()
    main(args.results_dir)
