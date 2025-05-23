"""
Logging Manager for NES Copilot

This module provides centralized logging functionality for the NES Copilot system.
"""

import os
import logging
import datetime
from typing import Optional, Dict, Any


class LoggingManager:
    """
    Logging manager for the NES Copilot system.
    
    Provides centralized logging functionality, including console and file logging,
    and report generation.
    """
    
    def __init__(self, data_manager, log_level: int = logging.INFO):
        """
        Initialize the logging manager.
        
        Args:
            data_manager: Data manager instance.
            log_level: Logging level (default: INFO).
        """
        self.data_manager = data_manager
        self.log_level = log_level
        
        # Get the logs directory
        self.logs_dir = self.data_manager.get_module_dir('logs')
        
        # Create the main logger
        self.logger = logging.getLogger('nes_copilot')
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        log_file = os.path.join(self.logs_dir, 'nes_copilot.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Dictionary to store module-specific loggers
        self.module_loggers = {}
        
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger for a specific module.
        
        Args:
            module_name: Name of the module (optional).
            
        Returns:
            Logger instance.
        """
        if not module_name:
            return self.logger
            
        # Check if a logger for this module already exists
        if module_name in self.module_loggers:
            return self.module_loggers[module_name]
            
        # Create a new logger for this module
        module_logger = logging.getLogger(f'nes_copilot.{module_name}')
        module_logger.setLevel(self.log_level)
        module_logger.handlers = []  # Clear any existing handlers
        
        # Create file handler for module-specific logs
        module_log_file = os.path.join(self.logs_dir, f'{module_name}.log')
        module_file_handler = logging.FileHandler(module_log_file)
        module_file_handler.setLevel(self.log_level)
        module_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        module_file_handler.setFormatter(module_format)
        module_logger.addHandler(module_file_handler)
        
        # Store the logger
        self.module_loggers[module_name] = module_logger
        
        return module_logger
        
    def log_module_start(self, module_name: str, config: Dict[str, Any]):
        """
        Log the start of a module's execution.
        
        Args:
            module_name: Name of the module.
            config: Module configuration.
        """
        logger = self.get_logger(module_name)
        logger.info(f"Starting {module_name} module")
        logger.info(f"Configuration: {config}")
        
    def log_module_end(self, module_name: str, results: Dict[str, Any]):
        """
        Log the end of a module's execution.
        
        Args:
            module_name: Name of the module.
            results: Module results.
        """
        logger = self.get_logger(module_name)
        logger.info(f"Completed {module_name} module")
        logger.info(f"Results: {results}")
        
    def log_error(self, module_name: str, error: Exception):
        """
        Log an error that occurred during module execution.
        
        Args:
            module_name: Name of the module.
            error: Exception that occurred.
        """
        logger = self.get_logger(module_name)
        logger.error(f"Error in {module_name} module: {str(error)}", exc_info=True)
        
    def generate_html_report(self, title: str, content: str) -> str:
        """
        Generate an HTML report.
        
        Args:
            title: Report title.
            content: Report content (HTML).
            
        Returns:
            Path to the generated report.
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"{title.lower().replace(' ', '_')}_{timestamp}.html"
        report_path = os.path.join(self.logs_dir, report_filename)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                .timestamp {{ color: #666666; font-size: 0.8em; }}
                .content {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="timestamp">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="content">
                {content}
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_template)
            
        return report_path
