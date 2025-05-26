#!/usr/bin/env python
"""
Wrapper script for empirical NES model fitting without SBC.
This script calls fit_nes_to_roberts_data_sbc_focused.py with recommended defaults for
empirical fitting (skipping SBC) and allows specifying a pre-trained NPE.
"""
import argparse
import subprocess
import sys
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run empirical NES fit (skip SBC)')
    parser.add_argument('--npe_file', type=str, required=True, 
                      help='Path to pre-trained NPE file')
    parser.add_argument('--npe_posterior_samples', type=int, default=500, 
                      help='Number of posterior samples per subject')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed')
    parser.add_argument('--output_dir', type=str, default='empirical_fit_output',
                      help='Output directory for empirical fitting results')
    parser.add_argument('--force_cpu', action='store_true',
                      help='Force CPU usage even if GPU is available')
    parser.add_argument('--roberts_data_file', type=str, 
                      default='roberts_framing_data/ftp_osf_data.csv',
                      help='Path to Roberts et al. data file')
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    log_file = output_dir / 'empirical_fit.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting empirical fitting with NPE: {args.npe_file}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Determine path to main script
    main_script = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 'fit_nes_to_roberts_data_sbc_focused.py'
    ))
    
    # Build command
    cmd = [
        sys.executable, 
        main_script,
        '--npe_train_sims', '0',  # Skip training, we'll use the provided checkpoint
        '--npe_architecture', 'nsf',  # Specify the architecture used for the pre-trained model
        '--output_base_name', str(output_dir / 'empirical_fit'),
        '--roberts_data_file', str(Path(args.roberts_data_file).absolute()),
        '--sbc_posterior_samples', str(args.npe_posterior_samples),
        '--seed', str(args.seed),
        '--sbc_debug_mode',  # Run with minimal settings for empirical fitting
    ]
    
    # Add the pre-trained model path
    cmd.extend(['--npe_file', str(Path(args.npe_file).absolute())])
    
    if args.force_cpu:
        cmd.append('--force_cpu')
    else:
        # Ensure we don't accidentally use GPU if not forced
        cmd.extend(['--force_cpu'])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info("Empirical fitting completed successfully!")
        logger.debug(f"Script output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Script warnings/errors:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Empirical fitting failed with return code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        logger.error(f"Output:\n{e.stdout}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during empirical fitting: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
