"""
Module Base Class for NES Copilot

This module defines the base class for all NES Copilot modules.
"""

from typing import Dict, Any, Optional


class ModuleBase:
    """
    Base class for all NES Copilot modules.
    
    Provides common functionality and interface for all modules.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the module.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging_manager.get_logger(self.__class__.__name__)
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the module's main functionality.
        
        Args:
            **kwargs: Additional arguments specific to the module.
            
        Returns:
            Dict containing the results of the module execution.
        """
        raise NotImplementedError("Subclasses must implement run()")
        
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate that all required inputs are available and correctly formatted.
        
        Args:
            **kwargs: Additional arguments specific to the module.
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        return True
        
    def save_outputs(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save module outputs to disk using the data manager.
        
        Args:
            results: Results to save.
            
        Returns:
            Dict mapping output names to their paths.
        """
        return {}
