import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from nes_copilot.summary_stats_module import SummaryStatsModule
from nes_copilot.config_manager import ConfigManager
from nes_copilot.data_manager import DataManager
from nes_copilot.logging_manager import LoggingManager

def generate_test_data(n_trials=1000):
    """Generate test data with realistic patterns."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate frame and condition
    frames = np.random.choice(['gain', 'loss'], size=n_trials, p=[0.5, 0.5])
    conditions = np.random.choice(['tc', 'ntc'], size=n_trials, p=[0.5, 0.5])
    
    # Generate choices with realistic patterns
    choices = np.zeros(n_trials, dtype=int)
    rt = np.zeros(n_trials)
    
    # Base probabilities for gambling
    p_gamble = {
        'gain': {'tc': 0.3, 'ntc': 0.4},
        'loss': {'tc': 0.5, 'ntc': 0.6}
    }
    
    # Base RT parameters (in seconds)
    rt_params = {
        'gain': {'tc': (1.2, 0.3), 'ntc': (1.4, 0.4)},
        'loss': {'tc': (1.1, 0.3), 'ntc': (1.3, 0.4)}
    }
    
    for i in range(n_trials):
        frame = frames[i]
        cond = conditions[i]
        
        # Generate choice
        p = p_gamble[frame][cond]
        choices[i] = np.random.binomial(1, p)
        
        # Generate RT based on condition and choice
        mu, sigma = rt_params[frame][cond]
        if choices[i] == 1:  # Gamble RT is slightly faster
            rt[i] = np.random.normal(mu * 0.9, sigma * 0.8)
        else:  # Sure choice RT
            rt[i] = np.random.normal(mu, sigma)
    
    # Ensure RT is positive
    rt = np.maximum(0.1, rt)
    
    # Create DataFrame
    df = pd.DataFrame({
        'frame': frames,
        'cond': conditions,
        'choice': choices,
        'rt': rt
    })
    
    return df

def test_summary_stats():
    """Test the summary statistics calculation."""
    print("Generating test data...")
    test_data = generate_test_data(n_trials=1000)
    
    print("\nSample of test data:")
    print(test_data.head())
    
    print("\nSetting up dependencies...")
    
    # Set up configuration
    config = {
        'output_dir': 'test_output',
        'logging': {
            'log_level': 'INFO',
            'log_file': 'test_summary_stats.log'
        }
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize dependencies
    logging_manager = LoggingManager(config)
    data_manager = DataManager(config, logging_manager)
    config_manager = ConfigManager(config, logging_manager)
    
    print("\nCalculating summary statistics...")
    stats_module = SummaryStatsModule(config_manager, data_manager, logging_manager)
    
    # Get all available stat keys
    stat_keys = [
        'prop_gamble_overall',
        'mean_rt_overall',
        'rt_q10_overall',
        'rt_q50_overall',
        'rt_q90_overall'
    ]
    
    # Add condition-specific stats
    for cond in ['Gain_TC', 'Gain_NTC', 'Loss_TC', 'Loss_NTC']:
        stat_keys.extend([
            f'prop_gamble_{cond}',
            f'mean_rt_{cond}',
            f'rt_q10_{cond}',
            f'rt_q50_{cond}',
            f'rt_q90_{cond}'
        ])
    
    # Calculate statistics
    results = stats_module.run(test_data, stat_keys=stat_keys)
    
    # Print results
    print("\nSummary Statistics:")
    for stat, value in results['summary_stats'].items():
        print(f"{stat}: {value:.4f}")
    
    # Calculate and print composite indices
    print("\nComposite Indices:")
    composites = stats_module.calculate_composite_indices(results['summary_stats'])
    for stat, value in composites.items():
        print(f"{stat}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    test_summary_stats()
