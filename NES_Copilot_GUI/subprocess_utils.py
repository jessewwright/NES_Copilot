"""
Subprocess utilities for the NES Co-Pilot Mission Control GUI.

This module provides utilities for managing subprocess calls to run_pipeline.py
and other backend scripts.
"""

import subprocess
import os
import threading
import time
from typing import Optional, Dict, Any, List, Tuple

def launch_run(config_file: str, output_dir: Optional[str] = None) -> Tuple[subprocess.Popen, str]:
    """
    Launch a run using the specified configuration file.
    
    Args:
        config_file: Path to the configuration file.
        output_dir: Optional output directory. If not specified, the default from the config will be used.
        
    Returns:
        Tuple of (subprocess handle, log file path).
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        subprocess.SubprocessError: If the subprocess fails to start.
    """
    # Ensure the configuration file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    # Get the script path
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              "scripts", "run_pipeline.py")
    
    # Ensure the script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Run pipeline script not found: {script_path}")
        
    # Construct the command
    cmd = ["python", script_path, "--config", config_file]
    
    # Add output directory if specified
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
        
    # Create the output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Create logs directory within output directory
    logs_dir = os.path.join(output_dir, "logs") if output_dir else None
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        
    # Create log file path
    log_file = os.path.join(logs_dir, "run.log") if logs_dir else None
    
    # Launch the subprocess
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
    return process, log_file

def check_process_status(process: subprocess.Popen) -> str:
    """
    Check the status of a subprocess.
    
    Args:
        process: Subprocess handle.
        
    Returns:
        Status string: "running", "completed", or "failed".
    """
    if process.poll() is None:
        return "running"
    elif process.returncode == 0:
        return "completed"
    else:
        return "failed"

def terminate_process(process: subprocess.Popen) -> bool:
    """
    Terminate a subprocess.
    
    Args:
        process: Subprocess handle.
        
    Returns:
        True if the process was terminated, False otherwise.
    """
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return True
    return False

class ProcessMonitor:
    """
    Monitor a subprocess and update its status.
    """
    
    def __init__(self, process: subprocess.Popen, status_callback=None):
        """
        Initialize the process monitor.
        
        Args:
            process: Subprocess handle.
            status_callback: Optional callback function to call when the process status changes.
        """
        self.process = process
        self.status_callback = status_callback
        self.running = False
        self.thread = None
        
    def start(self):
        """
        Start monitoring the process.
        """
        if self.thread is not None and self.thread.is_alive():
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """
        Stop monitoring the process.
        """
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
            
    def _monitor(self):
        """
        Monitor the process and update its status.
        """
        while self.running:
            status = check_process_status(self.process)
            
            if status != "running":
                # Process has completed or failed
                if self.status_callback:
                    self.status_callback(status)
                self.running = False
                break
                
            # Sleep for a short time before checking again
            time.sleep(1)
