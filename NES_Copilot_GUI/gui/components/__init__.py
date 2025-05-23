"""
__init__.py for components package

This file initializes the components package.
"""

from .config_display import display_config, config_selector
from .log_viewer import display_log, log_tail
from .plot_display import (
    display_image, display_matplotlib_figure, display_base64_image,
    display_multiple_images, display_plot_with_download
)
from .result_tables import (
    display_dataframe, display_dataframe_with_download, display_ks_test_results,
    display_empirical_fit_results, display_ppc_coverage_summary, display_subject_selector
)
from .navigation import sidebar_navigation, create_tabs, page_header

__all__ = [
    # Config display
    'display_config', 'config_selector',
    
    # Log viewer
    'display_log', 'log_tail',
    
    # Plot display
    'display_image', 'display_matplotlib_figure', 'display_base64_image',
    'display_multiple_images', 'display_plot_with_download',
    
    # Result tables
    'display_dataframe', 'display_dataframe_with_download', 'display_ks_test_results',
    'display_empirical_fit_results', 'display_ppc_coverage_summary', 'display_subject_selector',
    
    # Navigation
    'sidebar_navigation', 'create_tabs', 'page_header'
]
