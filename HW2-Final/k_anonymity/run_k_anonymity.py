# run_k_anonymity.py

import pandas as pd
import os
from k_anonymizer import apply_k_anonymity

# Set paths
RAW_DATA_PATH = 'data/colorado.csv'   
OUTPUT_DIR = 'output/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Step 1: Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Step 2: Applying K-Anonymity...")
    quasi_identifiers = ['AGE', 'SEX', 'COUNTY', 'CITY', 'EDUC']
    k = 5

    k_anon_df = apply_k_anonymity(df, quasi_identifiers, k)

    print(f"Step 3: Saving anonymized data with {k}-Anonymity...")
    k_anon_df.to_csv(os.path.join(OUTPUT_DIR, 'k_anonymity_output.csv'), index=False)
    print("K-Anonymized data saved!")

if __name__ == "__main__":
    main()
