import pandas as pd
from tqdm import tqdm
import os

file_path = "IAIR7EFL.DTA"
output_csv = "odisha_womens_data.csv"

# Comprehensive list of features for women's anaemia prediction
columns_to_keep = [
    'v012',  # Age
    'v445',  # BMI
    'v453',  # Hemoglobin level (Hb)
    'v042',  # Anaemia level (Target)
    'v106',  # Education level
    'v190',  # Wealth index
    'v213',  # Currently pregnant
    'v024',  # Region (21 = Odisha)
    'v025',  # Residence type (urban/rural)
    'v130',  # Religion
    'v481',  # Covered by health insurance
    'v157',  # Frequency of reading newspaper/magazine
    'v158',  # Frequency of listening to radio
    'v159'   # Frequency of watching TV
]

print(f"Extracting columns: {columns_to_keep}")
print("Filtering for Region (v024) == 21 (Odisha)...")

chunksize = 100000 
chunks = []
total_rows_extracted = 0

try:
    with pd.io.stata.StataReader(file_path, chunksize=chunksize, convert_categoricals=False) as reader:
        for i, chunk in enumerate(tqdm(reader, desc="Processing chunks")):
            # Get available columns in this dataset
            available_cols = [c for c in columns_to_keep if c in chunk.columns]
            
            # Filter the chunk for Odisha (v024 == 21)
            # Make sure v024 exists in the chunk first
            if 'v024' in available_cols:
                # Filter rows where region is Odisha
                odisha_chunk = chunk[chunk['v024'] == 21]
                
                # Keep only relevant columns
                odisha_chunk = odisha_chunk[available_cols]
                
                if not odisha_chunk.empty:
                    chunks.append(odisha_chunk)
                    total_rows_extracted += len(odisha_chunk)
            
    print(f"\nFinished scanning. Found {total_rows_extracted} records for Odisha.")
    
    if chunks:
        print(f"Concatenating and saving to {output_csv}...")
        data = pd.concat(chunks, ignore_index=True)
        data.to_csv(output_csv, index=False)
        print(f"Success! Saved {data.shape[0]} rows and {data.shape[1]} columns to {output_csv}.")
        print("Data preview:")
        print(data.head())
    else:
        print("No records found for Odisha (v024 == 21). Please check the region code for NFHS-5.")
        
except Exception as e:
    print(f"Error during extraction: {e}")
