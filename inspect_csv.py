import pandas as pd
import json

# Path to the CSV file
csv_path = "empirical_fit_results/empirical_fitting_results.csv"

# First, read the file as plain text
print("\n=== Raw file content (first 10 lines) ===")
with open(csv_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 10:  # Print first 10 lines
            print(f"{i}: {line.strip()}")
        else:
            break

# Try to read with different encodings if needed
encodings = ['utf-8', 'latin-1', 'cp1252']
df = None

for encoding in encodings:
    try:
        print(f"\nTrying to read with encoding: {encoding}")
        df = pd.read_csv(csv_path, encoding=encoding)
        print("Success!")
        break
    except Exception as e:
        print(f"Failed with {encoding}: {str(e)}")

if df is not None:
    print("\n=== DataFrame Info ===")
    print(df.info())
    
    print("\n=== First few rows ===")
    print(df.head().to_string())
    
    print("\n=== Column dtypes ===")
    print(df.dtypes)
    
    # Check for any non-numeric columns that should be numeric
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    
    print("\n=== Non-numeric columns ===")
    for col in non_numeric_cols:
        print(f"\nColumn: {col}")
        print(f"Sample values: {df[col].head(3).tolist()}")
        
        # Try to convert to numeric
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            num_nulls = converted.isna().sum()
            print(f"Converted to numeric with {num_nulls} nulls")
            print(f"Converted sample: {converted.head(3).tolist()}")
        except Exception as e:
            print(f"Could not convert to numeric: {str(e)}")
else:
    print("\nFailed to read the CSV file with any encoding")
