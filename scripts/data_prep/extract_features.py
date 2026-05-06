import pandas as pd
from tqdm import tqdm

file_path = "IAIR7EFL.DTA"
output_csv = "odisha_womens_features.csv"

# Comprehensive list of cost-aware demographic and health features (No blood work needed)
columns_to_keep = [
    'v012',  # Age
    'v445',  # BMI
    'v106',  # Education level
    'v190',  # Wealth index
    'v213',  # Currently pregnant
    'v404',  # Currently breastfeeding
    'v208',  # Births in last 5 years
    'v024',  # Region (21 = Odisha)
    'v025',  # Residence type (1=urban, 2=rural)
    'v130',  # Religion
    'v481',  # Covered by health insurance
    'v157',  # Frequency of reading newspaper/magazine
    'v158',  # Frequency of listening to radio
    'v159',  # Frequency of watching TV
    'v414e', # Frequency of eating eggs
    'v414f', # Frequency of eating meat
    'v453'   # Hemoglobin level (STRICTLY FOR DEFINING TARGET, WILL BE DROPPED LATER)
]

print(f"Extracting columns for cost-aware prediction: {columns_to_keep}")
print("Filtering for Region (v024) == 21 (Odisha)...")

chunksize = 100000 
chunks = []
total_rows_extracted = 0

try:
    with pd.io.stata.StataReader(file_path, chunksize=chunksize, convert_categoricals=False) as reader:
        for i, chunk in enumerate(tqdm(reader, desc="Processing chunks")):
            available_cols = [c for c in columns_to_keep if c in chunk.columns]
            
            if 'v024' in available_cols:
                # Filter rows where region is Odisha
                odisha_chunk = chunk[chunk['v024'] == 21]
                # Keep only relevant columns
                odisha_chunk = odisha_chunk[available_cols]
                
                if not odisha_chunk.empty:
                    chunks.append(odisha_chunk)
                    total_rows_extracted += len(odisha_chunk)
            
    if chunks:
        data = pd.concat(chunks, ignore_index=True)
        data.to_csv(output_csv, index=False)
        print(f"\nSuccess! Saved {data.shape[0]} rows and {data.shape[1]} columns to {output_csv}.")
    else:
        print("\nNo records found for Odisha.")
        
except Exception as e:
    print(f"Error during extraction: {e}")
