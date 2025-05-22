"""
Workflow Manager for NES Copilot

This module orchestrates the execution of modules based on the configuration.
"""

import os
import time
import importlib
from typing import Dict, Any, List, Optional, Callable

class WorkflowManager:
    """
    Workflow manager for the NES Copilot system.
    
    Orchestrates the execution of modules based on the configuration,
    manages dependencies between modules, and handles execution flow.
    """
    
    def __init__(self, config_manager, data_manager, logging_manager):
        """
        Initialize the workflow manager.
        
        Args:
            config_manager: Configuration manager instance.
            data_manager: Data manager instance.
            logging_manager: Logging manager instance.
        """
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging_manager.get_logger('workflow_manager')
        self.logging_manager = logging_manager
        
        # Get the modules to run from the configuration
        self.run_modules = self.config_manager.get_param('run_modules', {})
        
        # Define module dependencies
        self.module_dependencies = {
            'data_prep': [],
            'npe': ['data_prep'],
            'sbc': ['npe'],
            'empirical_fit': ['npe', 'data_prep'],
            'ppc': ['empirical_fit', 'npe', 'data_prep'],
            'analysis': ['empirical_fit', 'ppc']
        }
        
        # Track module execution status
        self.module_status = {module: 'pending' for module in self.run_modules}
        self.module_results = {}
        
    def run_pipeline(self):
        """
        Run the complete pipeline based on the configuration.
        
        Returns:
            Dict containing the results of all executed modules.
        """
        self.logger.info("Starting NES Copilot pipeline")
        
        # Save configuration snapshot
        config_dir = self.data_manager.get_module_dir('config')
        self.config_manager.save_config_snapshot(config_dir)
        
        # Determine execution order based on dependencies
        execution_order = self._determine_execution_order()
        self.logger.info(f"Module execution order: {execution_order}")
        
        # Execute modules in order
        for module_name in execution_order:
            if not self.run_modules.get(module_name, False):
                self.logger.info(f"Skipping {module_name} module (disabled in configuration)")
                continue
                
            self.logger.info(f"Executing {module_name} module")
            try:
                # Execute the module
                results = self._execute_module(module_name)
                
                # Update module status and results
                self.module_status[module_name] = 'completed'
                self.module_results[module_name] = results
                
                self.logger.info(f"Module {module_name} completed successfully")
            except Exception as e:
                self.logger.error(f"Error executing {module_name} module: {str(e)}", exc_info=True)
                self.module_status[module_name] = 'failed'
                self.module_results[module_name] = {'error': str(e)}
                
                # Determine if we should continue or abort
                if self._should_abort_on_failure(module_name):
                    self.logger.error(f"Aborting pipeline due to failure in {module_name} module")
                    break
        
        # Generate final report
        self._generate_pipeline_report()
        
        self.logger.info("NES Copilot pipeline completed")
        return self.module_results
    
    def _determine_execution_order(self) -> List[str]:
        """
        Determine the order in which modules should be executed based on dependencies.
        
        Returns:
            List of module names in execution order.
        """
        # Start with modules that have no dependencies
        execution_order = []
        remaining_modules = list(self.run_modules.keys())
        
        while remaining_modules:
            # Find modules whose dependencies have been satisfied
            ready_modules = [
                module for module in remaining_modules
                if all(dep not in remaining_modules or dep not in self.run_modules
                      for dep in self.module_dependencies.get(module, []))
            ]
            
            if not ready_modules:
                # Circular dependency or missing module
                self.logger.error(f"Circular dependency or missing module detected: {remaining_modules}")
                break
                
            # Add ready modules to execution order
            execution_order.extend(ready_modules)
            
            # Remove ready modules from remaining modules
            for module in ready_modules:
                remaining_modules.remove(module)
                
        return execution_order
    
    def _execute_module(self, module_name: str) -> Dict[str, Any]:
        """
        Execute a specific module.
        
        Args:
            module_name: Name of the module to execute.
            
        Returns:
            Dict containing the module results.
        """
        # Get module configuration
        module_config = self.config_manager.get_module_config(module_name)
        
        # Log module start
        self.logging_manager.log_module_start(module_name, module_config)
        
        # Import the module class
        try:
            module_class = self._import_module_class(module_name)
        except ImportError as e:
            self.logger.error(f"Error importing module {module_name}: {str(e)}")
            raise
        
        # Initialize the module
        module_instance = module_class(
            self.config_manager,
            self.data_manager,
            self.logging_manager
        )
        
        # Get dependency results
        dependency_results = {
            dep: self.module_results.get(dep, {})
            for dep in self.module_dependencies.get(module_name, [])
            if dep in self.module_results
        }
        
        # Execute the module
        start_time = time.time()
        results = module_instance.run(**dependency_results)
        end_time = time.time()
        
        # Add execution time to results
        results['execution_time'] = end_time - start_time
        
        # Log module end
        self.logging_manager.log_module_end(module_name, results)
        
        return results
    
    def _import_module_class(self, module_name: str) -> type:
        """
        Import the module class for a specific module.
        
        Args:
            module_name: Name of the module.
            
        Returns:
            Module class.
        """
        # Convert module_name to class name (e.g., 'data_prep' -> 'DataPrepModule')
        class_name = ''.join(word.capitalize() for word in module_name.split('_')) + 'Module'
        
        # Import the module
        module_path = f"nes_copilot.{module_name}.{module_name}_module"
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Error importing {class_name} from {module_path}: {str(e)}")
            raise ImportError(f"Could not import {class_name} from {module_path}: {str(e)}")
    
    def _should_abort_on_failure(self, failed_module: str) -> bool:
        """
        Determine if the pipeline should abort due to a module failure.
        
        Args:
            failed_module: Name of the failed module.
            
        Returns:
            True if the pipeline should abort, False otherwise.
        """
        # Get modules that depend on the failed module
        dependent_modules = [
            module for module, deps in self.module_dependencies.items()
            if failed_module in deps and module in self.run_modules and self.run_modules[module]
        ]
        
        # If any dependent modules are enabled, we should abort
        if dependent_modules:
            self.logger.warning(f"Modules {dependent_modules} depend on {failed_module} and cannot be executed")
            return True
            
        return False
    
    def _generate_pipeline_report(self):
        """
        Generate a report summarizing the pipeline execution.
        """
        # Create HTML content for the report
        html_content = """
        <h2>Pipeline Execution Summary</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>Module</th>
                <th>Status</th>
                <th>Execution Time</th>
                <th>Details</th>
            </tr>
        """
        
        for module_name in self.module_status:
            status = self.module_status[module_name]
            execution_time = self.module_results.get(module_name, {}).get('execution_time', 'N/A')
            if execution_time != 'N/A':
                execution_time = f"{execution_time:.2f} seconds"
                
            details = ""
            if status == 'failed':
                error = self.module_results.get(module_name, {}).get('error', 'Unknown error')
                details = f"Error: {error}"
            elif status == 'completed':
                # Extract key results for the report
                results = self.module_results.get(module_name, {})
                if 'output_paths' in results:
                    details = "Output files:<br>"
                    for key, path in results['output_paths'].items():
                        details += f"- {key}: {path}<br>"
            
            html_content += f"""
            <tr>
                <td>{module_name}</td>
                <td>{status}</td>
                <td>{execution_time}</td>
                <td>{details}</td>
            </tr>
            """
            
        html_content += "</table>"
        
        # Generate the report
        report_path = self.logging_manager.generate_html_report(
            "NES Copilot Pipeline Execution Report",
            html_content
        )
        
        self.logger.info(f"Pipeline execution report generated: {report_path}")
