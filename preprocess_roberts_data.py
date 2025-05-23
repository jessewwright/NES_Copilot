import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_roberts_data(input_path, output_dir):
    """
    Preprocess the Roberts et al. dataset for NPE/SBC analysis.
    
    Args:
        input_path: Path to the raw CSV file
        output_dir: Directory to save the processed data
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print("\nOriginal data shape:", df.shape)
    print("\nFirst few rows of raw data:")
    print(df.head())
    
    # Filter for target trials only (exclude catch trials)
    print("\nFiltering for target trials...")
    original_count = len(df)
    df = df[df['trialType'] == 'target'].copy()
    print(f"  - Kept {len(df)} of {original_count} rows after filtering")
    
    # Basic data cleaning
    print("\nPerforming data cleaning...")
    df['frame'] = df['frame'].str.lower()
    df['cond'] = df['cond'].str.lower()
    
    # Handle missing values in RT
    missing_rt = df['rt'].isna().sum()
    if missing_rt > 0:
        print(f"  - Found {missing_rt} missing RT values, these will be excluded")
    
    # Create a clean dataset with selected columns
    processed_df = df[[
        'subject', 'trial', 'block', 'frame', 'cond', 
        'choice', 'rt', 'prob', 'sureOutcome', 'endow'
    ]].rename(columns={
        'rt': 'reaction_time',
        'prob': 'gamble_prob',
        'sureOutcome': 'sure_amount',
        'endow': 'endowment',
        'choice': 'gamble_chosen'  # 1 = gamble, 0 = sure
    }).copy()
    
    # Add a trial ID
    processed_df['trial_id'] = processed_df['subject'].astype(str) + '_' + processed_df['trial'].astype(str)
    
    # Save the processed data
    output_path = Path(output_dir) / 'roberts_processed.csv'
    processed_df.to_csv(output_path, index=False)
    
    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"Processed data saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(f"Number of participants: {processed_df['subject'].nunique()}")
    print(f"Number of trials per participant: {len(processed_df) / processed_df['subject'].nunique():.1f}")
    print("\nGambling rates by condition:")
    print(processed_df.groupby(['frame', 'cond'])['gamble_chosen'].mean().unstack())
    
    return processed_df

if __name__ == "__main__":
    # Define paths
    input_path = r"C:\Users\jesse\Hegemonikon Project\hegemonikon\Roberts_Framing_Data\ftp_osf_data.csv"
    output_dir = r"C:\Users\jesse\Hegemonikon Project\NES_Copilot\data"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run preprocessing
    preprocess_roberts_data(input_path, output_dir)
