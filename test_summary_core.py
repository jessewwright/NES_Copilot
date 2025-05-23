import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from nes_copilot.stats_schema import validate_summary_stats

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

def calculate_summary_stats(df):
    """Calculate summary statistics from trial data."""
    stats = {}
    
    # Overall statistics
    stats['prop_gamble_overall'] = df['choice'].mean()
    stats['mean_rt_overall'] = df['rt'].mean()
    
    # RT percentiles
    q10, q50, q90 = df['rt'].quantile([0.1, 0.5, 0.9])
    stats['rt_q10_overall'] = q10
    stats['rt_q50_overall'] = q50
    stats['rt_q90_overall'] = q90
    
    # Condition-specific statistics
    conditions = {
        'Gain_TC': (df['frame'] == 'gain') & (df['cond'] == 'tc'),
        'Gain_NTC': (df['frame'] == 'gain') & (df['cond'] == 'ntc'),
        'Loss_TC': (df['frame'] == 'loss') & (df['cond'] == 'tc'),
        'Loss_NTC': (df['frame'] == 'loss') & (df['cond'] == 'ntc')
    }
    
    for cond_name, mask in conditions.items():
        cond_df = df[mask]
        if len(cond_df) > 0:
            stats[f'prop_gamble_{cond_name}'] = cond_df['choice'].mean()
            stats[f'mean_rt_{cond_name}'] = cond_df['rt'].mean()
            
            if len(cond_df) >= 2:
                try:
                    q10, q50, q90 = cond_df['rt'].quantile([0.1, 0.5, 0.9])
                    stats[f'rt_q10_{cond_name}'] = q10
                    stats[f'rt_q50_{cond_name}'] = q50
                    stats[f'rt_q90_{cond_name}'] = q90
                except:
                    pass
    
    return stats

def main():
    print("Generating test data...")
    test_data = generate_test_data(n_trials=1000)
    
    print("\nSample of test data:")
    print(test_data.head())
    
    print("\nCalculating summary statistics...")
    stats = calculate_summary_stats(test_data)
    
    print("\nSummary Statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.4f}")
    
    # Validate the statistics
    print("\nValidating statistics...")
    is_valid, message = validate_summary_stats(stats)
    if is_valid:
        print("[OK] All statistics are valid!")
    else:
        print(f"[ERROR] Validation failed: {message}")
    
    # Calculate composite indices
    print("\nComposite Indices:")
    
    # Framing index
    if all(k in stats for k in ['prop_gamble_Loss_TC', 'prop_gamble_Gain_TC',
                              'prop_gamble_Loss_NTC', 'prop_gamble_Gain_NTC']):
        p_gain = (stats['prop_gamble_Gain_TC'] + stats['prop_gamble_Gain_NTC']) / 2
        p_loss = (stats['prop_gamble_Loss_TC'] + stats['prop_gamble_Loss_NTC']) / 2
        print(f"Framing index (Loss - Gain): {p_loss - p_gain:.4f}")
    
    # Time pressure index
    if all(k in stats for k in ['prop_gamble_Gain_TC', 'prop_gamble_Gain_NTC',
                              'prop_gamble_Loss_TC', 'prop_gamble_Loss_NTC']):
        p_tc = (stats['prop_gamble_Gain_TC'] + stats['prop_gamble_Loss_TC']) / 2
        p_ntc = (stats['prop_gamble_Gain_NTC'] + stats['prop_gamble_Loss_NTC']) / 2
        print(f"Time pressure index (TC - NTC): {p_tc - p_ntc:.4f}")

if __name__ == "__main__":
    main()
