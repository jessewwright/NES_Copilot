"""
Main entry point for NES Copilot

This script runs the complete NES Copilot pipeline based on the master configuration.
"""

import os
import sys
import argparse
from nes_copilot.config_manager import ConfigManager
from nes_copilot.data_manager import DataManager
from nes_copilot.logging_manager import LoggingManager
from nes_copilot.workflow_manager import WorkflowManager


def main():
    """
    Main entry point for the NES Copilot pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the NES Copilot pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to master configuration file')
    args = parser.parse_args()
    
    # Check if configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
        
    # Initialize managers
    config_manager = ConfigManager(args.config)
    data_manager = DataManager(config_manager)
    logging_manager = LoggingManager(data_manager)
    
    # Get logger
    logger = logging_manager.get_logger()
    logger.info(f"Starting NES Copilot pipeline with configuration: {args.config}")
    
    # Initialize workflow manager
    workflow_manager = WorkflowManager(config_manager, data_manager, logging_manager)
    
    # Run pipeline
    try:
        results = workflow_manager.run_pipeline()
        logger.info("NES Copilot pipeline completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error running NES Copilot pipeline: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
