"""
Test script for empirical fitting of the NES model to Roberts et al. data.

This script runs a small-scale test of the empirical fitting process to verify that
the NPE training and SBC pipeline is working correctly.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_empirical_fit():
    """Run a small test of the empirical fitting process."""
    from fit_nes_to_roberts_data_sbc_focused import main as fit_main
    
    # Create a test output directory
    test_output_dir = Path("test_empirical_fit_output")
    test_output_dir.mkdir(exist_ok=True)
    
    # Set up test arguments for SBC run
    sbc_output_dir = test_output_dir / "sbc_run"
    test_args = [
        "--npe_train_sims", "100",  # Small number for testing
        "--template_trials", "20",   # Reduced number of trials
        "--sbc_datasets", "5",       # Minimal number of SBC datasets
        "--sbc_posterior_samples", "10",  # Few samples per dataset
        "--output_base_name", str(sbc_output_dir),
        "--seed", "42",              # Fixed seed for reproducibility
        "--npe_architecture", "maf", # Use MAF for faster training
        "--force_cpu",               # Ensure we're using CPU
        "--force_retrain_npe",       # Ensure NPE is trained and saved
    ]
    
    # Add Roberts data file path if not in default location
    roberts_data = Path("roberts_framing_data/ftp_osf_data.csv")
    if not roberts_data.exists():
        logger.warning(f"Roberts data file not found at {roberts_data}. Please specify with --roberts_data_file")
        test_args.extend(["--roberts_data_file", str(roberts_data)])
    
    # Parse arguments
    sys.argv = [sys.argv[0]] + test_args
    
    try:
        logger.info("Starting test empirical fitting...")
        fit_main()
        logger.info("Test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False
    
    # Check if expected output files were created
    required_files = [
        sbc_output_dir / "data" / "simulated_summary_stats.csv",
        sbc_output_dir / "data" / "training_stat_means.npy",
        sbc_output_dir / "data" / "training_stat_stds.npy"
    ]
    
    # Find the NPE checkpoint file (it has a dynamic name with simulation details)
    npe_checkpoint_pattern = f"nes_npe_sims*_template*_seed*.pt"
    npe_checkpoint_files = list((sbc_output_dir / "models").glob(npe_checkpoint_pattern))
    
    if not npe_checkpoint_files:
        logger.error(f"No NPE checkpoint files found matching pattern: {npe_checkpoint_pattern} in {sbc_output_dir / 'models'}")
        return False
    
    # Use the first matching checkpoint file
    npe_checkpoint = npe_checkpoint_files[0]
    logger.info(f"Found NPE checkpoint: {npe_checkpoint}")
    
    # Check if all required files exist
    all_files_exist = all(f.exists() for f in required_files)
    if all_files_exist:
        logger.info("All expected SBC output files were created successfully!")
        logger.info(f"Running empirical fitting with NPE checkpoint: {npe_checkpoint}")
        
        # Run the empirical fitting using the new script
        from run_empirical_fit import main as empirical_main
        
        # Set up arguments for empirical fitting
        sys.argv = [
            sys.argv[0],
            "--npe_checkpoint", str(npe_checkpoint),
            "--output_dir", str(test_output_dir / "empirical_fit"),
            "--num_posterior_samples", "50",  # Small number for testing
            "--seed", "42"
        ]
        
        try:
            success = empirical_main()
            if success:
                logger.info("Empirical fitting completed successfully!")
                # Check for output files
                output_dir = test_output_dir / "empirical_fit"
                if output_dir.exists():
                    logger.info(f"Empirical fit results saved to: {output_dir}")
                    # Check for expected output files
                    expected_files = [
                        output_dir / "empirical_fit.log",
                        output_dir / "posterior_samples.csv"
                    ]
                    if all(f.exists() for f in expected_files):
                        logger.info("All expected output files found")
                        return True
                    else:
                        missing = [f for f in expected_files if not f.exists()]
                        logger.error(f"Missing expected output files: {missing}")
                        return False
                else:
                    logger.error(f"Output directory not found: {output_dir}")
                    return False
            else:
                logger.error("Empirical fitting failed (returned False)")
                return False
        except Exception as e:
            logger.error(f"Empirical fitting failed with exception: {e}", exc_info=True)
            return False
    else:
        missing = [str(f) for f in required_files if not f.exists()]
        logger.error(f"Cannot run empirical fitting - missing required SBC output files: {missing}")
        return False

if __name__ == "__main__":
    success = test_empirical_fit()
    sys.exit(0 if success else 1)
