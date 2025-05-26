"""
Smoke test for the SBC (Simulation-Based Calibration) pipeline.

This script verifies that the NPE training and SBC pipeline runs end-to-end
with minimal settings. It's designed to catch major issues before running
longer, more resource-intensive training runs.
"""

import logging
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_sbc_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Set environment variable to avoid numexpr thread warnings
os.environ['NUMEXPR_MAX_THREADS'] = '8'

def run_smoke_test():
    """Run a smoke test of the SBC pipeline."""
    from fit_nes_to_roberts_data_sbc_focused import main as sbc_main
    
    # Create a test output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path(f"test_sbc_smoke_{timestamp}")
    test_output_dir.mkdir(exist_ok=True)
    
    # Set up test arguments for a minimal run
    test_args = [
        "--npe_train_sims", "100",       # Minimal number of training simulations
        "--template_trials", "20",       # Reduced number of trials
        "--sbc_datasets", "5",           # Minimal number of SBC datasets
        "--sbc_posterior_samples", "10",  # Few samples per dataset
        "--output_base_name", str(test_output_dir / "sbc_smoke_test"),
        "--seed", "42",                  # Fixed seed for reproducibility
        "--npe_architecture", "maf",     # Use MAF for faster training
        "--force_cpu",                   # Ensure we're using CPU
        "--sbc_debug_mode",              # Enable debug mode for SBC
        "--force_retrain_npe",           # Always retrain for the test
    ]
    
    # Use the same default path as in the main script
    default_data_path = Path("./roberts_framing_data/ftp_osf_data.csv")
    if not default_data_path.exists():
        logger.warning(f"Roberts data file not found at {default_data_path}.")
        logger.warning("Please specify the correct path with --roberts_data_file")
        test_args.extend(["--roberts_data_file", str(default_data_path)])
    
    # Parse arguments
    sys_argv_original = sys.argv
    sys.argv = [sys.argv[0]] + test_args
    
    try:
        logger.info("Starting SBC pipeline smoke test...")
        logger.info(f"Output will be saved to: {test_output_dir}")
        logger.info(f"Command: {' '.join(sys.argv)}")
        
        # Run the main SBC pipeline
        sbc_main()
        
        # Check for expected output files
        output_dir = test_output_dir / "sbc_smoke_test"
        required_files = [
            # NPE training outputs
            output_dir / "data" / "simulated_summary_stats.csv",
            output_dir / "models" / "density_estimator.pt",
            output_dir / "data" / "training_stat_means.npy",
            output_dir / "data" / "training_stat_stds.npy",
            # SBC outputs (if SBC ran)
            output_dir / "sbc" / "sbc_ranks.csv",
            output_dir / "sbc" / "sbc_ks_test_results.json"
        ]
        
        # Verify files exist
        missing_files = [str(f) for f in required_files if not f.exists()]
        
        if not missing_files:
            logger.info("✅ Smoke test passed! All expected output files were created.")
            return True
        else:
            logger.warning("⚠️  Smoke test completed, but some expected files are missing:")
            for f in missing_files:
                logger.warning(f"  - {f}")
            return False
            
    except Exception as e:
        logger.error("❌ Smoke test failed with error:", exc_info=True)
        return False
    finally:
        # Restore original command line arguments
        sys.argv = sys_argv_original

if __name__ == "__main__":
    success = run_smoke_test()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
