"""
Analysis Module for NES Copilot

This module handles analysis and visualization of results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
from scipy import stats

from nes_copilot.module_base import ModuleBase


class AnalysisModule(ModuleBase):
    """
    Analysis module for the NES Copilot system.
    
    Handles analysis and visualization of results.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the analysis module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        super().__init__(config_manager, data_manager, logging_manager)
        
        # Get module configuration
        self.config = self.config_manager.get_module_config('analysis')
        
    def run(self, fitting_results: Optional[str] = None, ppc_results: Optional[str] = None, 
            empirical_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis and generate visualizations.
        
        Args:
            fitting_results: Optional override for the fitting results path.
            ppc_results: Optional override for the PPC results path.
            empirical_data: Optional override for the empirical data path.
            **kwargs: Additional arguments (not used).
            
        Returns:
            Dict containing the results of the analysis.
        """
        self.logger.info("Starting analysis")
        
        # Get fitting results path
        if fitting_results is None:
            fitting_results = self.config.get('empirical_fit_results_file')
            if fitting_results is None:
                # Try to find the latest fitting results
                fitting_results_path = self.data_manager.get_output_path('empirical_fit', 'posterior_summaries.csv')
                if os.path.exists(fitting_results_path):
                    fitting_results = fitting_results_path
                else:
                    raise ValueError("Fitting results not specified and no default found")
                    
        # Load fitting results
        self.logger.info(f"Loading fitting results from {fitting_results}")
        fitting_df = pd.read_csv(fitting_results)
        
        # Get PPC results path
        if ppc_results is None:
            ppc_results = self.config.get('ppc_results_file')
            if ppc_results is None:
                # Try to find the latest PPC results
                ppc_results_path = self.data_manager.get_output_path('ppc', 'coverage_summary.csv')
                if os.path.exists(ppc_results_path):
                    ppc_results = ppc_results_path
                else:
                    self.logger.warning("PPC results not specified and no default found, skipping PPC analysis")
                    ppc_results = None
                    
        # Load PPC results if available
        ppc_df = None
        if ppc_results is not None:
            self.logger.info(f"Loading PPC results from {ppc_results}")
            ppc_df = pd.read_csv(ppc_results)
            
        # Load empirical data if not provided
        if empirical_data is None:
            empirical_data_path = self.data_manager.get_output_path('data_prep', 'empirical_data.csv')
            if os.path.exists(empirical_data_path):
                empirical_data = pd.read_csv(empirical_data_path)
            else:
                raise FileNotFoundError(f"Empirical data not found: {empirical_data_path}")
                
        # Validate inputs
        self.validate_inputs(fitting_df=fitting_df, empirical_data=empirical_data)
        
        # Perform analyses
        correlation_results = self._analyze_correlations(fitting_df, empirical_data)
        parameter_distribution_results = self._analyze_parameter_distributions(fitting_df)
        
        # Generate plots
        plot_paths = self._generate_plots(
            fitting_df=fitting_df,
            ppc_df=ppc_df,
            empirical_data=empirical_data,
            correlation_results=correlation_results,
            parameter_distribution_results=parameter_distribution_results
        )
        
        # Save outputs
        output_paths = self.save_outputs({
            'correlation_results': correlation_results,
            'parameter_distribution_results': parameter_distribution_results,
            'plot_paths': plot_paths
        })
        
        # Return results
        results = {
            'correlation_results': correlation_results,
            'parameter_distribution_results': parameter_distribution_results,
            'output_paths': output_paths
        }
        
        self.logger.info("Analysis completed successfully")
        return results
        
    def validate_inputs(self, fitting_df: pd.DataFrame, empirical_data: pd.DataFrame, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            fitting_df: Fitting results DataFrame.
            empirical_data: Empirical data DataFrame.
            **kwargs: Additional arguments (not used).
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        # Check if fitting results are provided
        if fitting_df is None or len(fitting_df) == 0:
            raise ValueError("Fitting results not provided or empty")
            
        # Check if fitting results have required columns
        required_columns = ['subject', 'parameter', 'mean']
        for col in required_columns:
            if col not in fitting_df.columns:
                raise ValueError(f"Fitting results missing required column: {col}")
                
        # Check if empirical data is provided
        if empirical_data is None or len(empirical_data) == 0:
            raise ValueError("Empirical data not provided or empty")
            
        # Check if empirical data has required columns
        required_columns = ['subject', 'choice', 'rt', 'frame', 'cond']
        for col in required_columns:
            if col not in empirical_data.columns:
                raise ValueError(f"Empirical data missing required column: {col}")
                
        return True
        
    def _analyze_correlations(self, fitting_df: pd.DataFrame, empirical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between fitted parameters and behavioral metrics.
        
        Args:
            fitting_df: Fitting results DataFrame.
            empirical_data: Empirical data DataFrame.
            
        Returns:
            Dict containing correlation results.
        """
        # Get correlation configuration
        correlation_config = self.config.get('correlations', {})
        correlations_to_compute = correlation_config.get('compute', [])
        
        # Calculate behavioral metrics for each subject
        subject_metrics = self._calculate_subject_metrics(empirical_data)
        
        # Pivot fitting results to have parameters as columns
        fitting_pivot = fitting_df.pivot(index='subject', columns='parameter', values='mean')
        
        # Merge with subject metrics
        analysis_df = pd.merge(fitting_pivot, subject_metrics, left_index=True, right_index=True)
        
        # Calculate correlations
        correlation_results = {}
        
        for param_metric_pair in correlations_to_compute:
            if len(param_metric_pair) != 2:
                continue
                
            param, metric = param_metric_pair
            
            if param not in fitting_pivot.columns or metric not in subject_metrics.columns:
                self.logger.warning(f"Skipping correlation between {param} and {metric}: not found in data")
                continue
                
            # Calculate correlation
            r, p = stats.pearsonr(analysis_df[param], analysis_df[metric])
            
            correlation_results[f"{param}_{metric}"] = {
                'parameter': param,
                'metric': metric,
                'r': r,
                'p': p,
                'significant': p < 0.05
            }
            
        return {
            'correlations': correlation_results,
            'analysis_df': analysis_df
        }
        
    def _calculate_subject_metrics(self, empirical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate behavioral metrics for each subject.
        
        Args:
            empirical_data: Empirical data DataFrame.
            
        Returns:
            DataFrame with subject metrics.
        """
        # Initialize results
        metrics = []
        
        # Process each subject
        for subject in empirical_data['subject'].unique():
            subject_data = empirical_data[empirical_data['subject'] == subject]
            
            # Calculate metrics
            subject_metrics = {
                'subject': subject,
                'num_trials': len(subject_data),
                'mean_rt': subject_data['rt'].mean(),
                'std_rt': subject_data['rt'].std(),
                'p_gamble': subject_data['choice'].mean()
            }
            
            # Calculate framing index
            gain_data = subject_data[subject_data['frame'] == 'gain']
            loss_data = subject_data[subject_data['frame'] == 'loss']
            
            if len(gain_data) > 0 and len(loss_data) > 0:
                p_gamble_gain = gain_data['choice'].mean()
                p_gamble_loss = loss_data['choice'].mean()
                subject_metrics['framing_index'] = p_gamble_loss - p_gamble_gain
            else:
                subject_metrics['framing_index'] = np.nan
                
            # Calculate time pressure index
            tc_data = subject_data[subject_data['cond'] == 'tc']
            ntc_data = subject_data[subject_data['cond'] == 'ntc']
            
            if len(tc_data) > 0 and len(ntc_data) > 0:
                p_gamble_tc = tc_data['choice'].mean()
                p_gamble_ntc = ntc_data['choice'].mean()
                subject_metrics['time_pressure_index'] = p_gamble_tc - p_gamble_ntc
            else:
                subject_metrics['time_pressure_index'] = np.nan
                
            # Add to results
            metrics.append(subject_metrics)
            
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)
        metrics_df.set_index('subject', inplace=True)
        
        return metrics_df
        
    def _analyze_parameter_distributions(self, fitting_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distributions of fitted parameters.
        
        Args:
            fitting_df: Fitting results DataFrame.
            
        Returns:
            Dict containing parameter distribution results.
        """
        # Get unique parameters
        parameters = fitting_df['parameter'].unique()
        
        # Calculate statistics for each parameter
        param_stats = {}
        
        for param in parameters:
            param_data = fitting_df[fitting_df['parameter'] == param]['mean']
            
            # Calculate statistics
            param_stats[param] = {
                'mean': param_data.mean(),
                'median': param_data.median(),
                'std': param_data.std(),
                'min': param_data.min(),
                'max': param_data.max(),
                'q25': param_data.quantile(0.25),
                'q75': param_data.quantile(0.75)
            }
            
        return {
            'parameter_stats': param_stats,
            'parameters': parameters
        }
        
    def _generate_plots(self, fitting_df: pd.DataFrame, ppc_df: Optional[pd.DataFrame], 
                       empirical_data: pd.DataFrame, correlation_results: Dict[str, Any],
                       parameter_distribution_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate analysis plots.
        
        Args:
            fitting_df: Fitting results DataFrame.
            ppc_df: PPC results DataFrame (optional).
            empirical_data: Empirical data DataFrame.
            correlation_results: Correlation analysis results.
            parameter_distribution_results: Parameter distribution analysis results.
            
        Returns:
            Dict mapping plot names to their paths.
        """
        plot_paths = {}
        
        # Create plots directory
        plots_dir = self.data_manager.get_output_path('analysis', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get plot configuration
        plot_config = self.config.get('plots', {})
        
        # Generate scatter plots
        if 'scatter' in plot_config:
            scatter_plots = plot_config['scatter']
            for param_metric_pair in scatter_plots:
                if len(param_metric_pair) != 2:
                    continue
                    
                param, metric = param_metric_pair
                plot_path = self._generate_scatter_plot(
                    correlation_results['analysis_df'],
                    param,
                    metric,
                    correlation_results['correlations'].get(f"{param}_{metric}"),
                    plots_dir
                )
                
                if plot_path:
                    plot_paths[f"scatter_{param}_{metric}"] = plot_path
                    
        # Generate parameter distribution plots
        if plot_config.get('parameter_distributions', True):
            for param in parameter_distribution_results['parameters']:
                plot_path = self._generate_parameter_distribution_plot(
                    fitting_df,
                    param,
                    parameter_distribution_results['parameter_stats'][param],
                    plots_dir
                )
                
                if plot_path:
                    plot_paths[f"dist_{param}"] = plot_path
                    
        # Generate PPC coverage plot
        if plot_config.get('ppc_coverage', True) and ppc_df is not None:
            plot_path = self._generate_ppc_coverage_plot(ppc_df, plots_dir)
            if plot_path:
                plot_paths["ppc_coverage"] = plot_path
                
        return plot_paths
        
    def _generate_scatter_plot(self, analysis_df: pd.DataFrame, param: str, metric: str, 
                              correlation_info: Optional[Dict[str, Any]], plots_dir: str) -> Optional[str]:
        """
        Generate a scatter plot for a parameter-metric pair.
        
        Args:
            analysis_df: DataFrame with parameters and metrics.
            param: Parameter name.
            metric: Metric name.
            correlation_info: Correlation information.
            plots_dir: Directory to save the plot.
            
        Returns:
            Path to the saved plot, or None if plot could not be generated.
        """
        if param not in analysis_df.columns or metric not in analysis_df.columns:
            self.logger.warning(f"Cannot generate scatter plot for {param} vs {metric}: not found in data")
            return None
            
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        sns.scatterplot(x=param, y=metric, data=analysis_df)
        
        # Add regression line
        sns.regplot(x=param, y=metric, data=analysis_df, scatter=False, line_kws={'color': 'red'})
        
        # Add correlation information if available
        if correlation_info:
            r = correlation_info['r']
            p = correlation_info['p']
            plt.title(f"{param} v
(Content truncated due to size limit. Use line ranges to read in chunks)