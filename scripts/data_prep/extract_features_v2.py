"""
V2 Feature Extraction: Expanded non-invasive feature set for improved anemia prediction.
Adds 12 new high-signal features to the original 16.
"""
import pandas as pd
from tqdm import tqdm

file_path = "e:/FRP PROJECT/Aneamia Detection/data/IAIR7EFL.DTA"
output_csv = "e:/FRP PROJECT/Aneamia Detection/data/odisha_womens_features_v2.csv"

# ============================================================
# ORIGINAL 16 FEATURES (kept for backward compatibility)
# ============================================================
original_features = [
    'v012',   # Age
    'v445',   # BMI
    'v106',   # Education level
    'v190',   # Wealth index
    'v213',   # Currently pregnant
    'v404',   # Currently breastfeeding
    'v208',   # Births in last 5 years
    'v024',   # Region (21 = Odisha)
    'v025',   # Residence type (urban/rural)
    'v130',   # Religion
    'v481',   # Health insurance
    'v157',   # Reads newspaper
    'v158',   # Listens to radio
    'v159',   # Watches TV
    'v414e',  # Eats eggs
    'v414f',  # Eats meat
]

# ============================================================
# NEW FEATURES: 12 high-signal, non-invasive additions
# Selection criteria: >95% fill rate, strong biological/socioeconomic relevance
# ============================================================
new_features = [
    # --- Sanitation & Environment (parasitic/infection-driven anemia) ---
    'v113',   # Source of drinking water (contaminated water → parasites → anemia)
    'v116',   # Type of toilet facility (sanitation → infection risk)
    'v161',   # Type of cooking fuel (indoor air pollution → chronic disease)
    
    # --- Household structure ---
    'v136',   # Number of household members (resource dilution)
    'v137',   # Number of children 5 and under (maternal depletion)
    'v201',   # Total children ever born (cumulative iron loss)
    
    # --- Reproductive health ---
    'v501',   # Marital status
    'v312',   # Current contraceptive method (hormonal contraceptives affect iron)
    
    # --- Healthcare access (the "Access-Aware" narrative) ---
    'v467b',  # Distance to health facility
    'v467d',  # Getting to health facility a big problem
    
    # --- Tobacco use (affects nutrient absorption) ---
    'v463z',  # Uses no tobacco (inverse flag)
    
    # --- Physical measurements ---
    'v438',   # Respondent height (cm) — stunting indicator, independent of BMI
]

# Target variable (will be used to define labels, then dropped)
target_var = ['v453']  # Hemoglobin

columns_to_keep = original_features + new_features + target_var

print(f"Extracting {len(columns_to_keep)} columns ({len(original_features)} original + {len(new_features)} new + 1 target)")
print("Filtering for Region (v024) == 21 (Odisha)...")

chunksize = 100000
chunks = []
total_rows = 0

try:
    with pd.io.stata.StataReader(file_path, chunksize=chunksize, convert_categoricals=False) as reader:
        for i, chunk in enumerate(tqdm(reader, desc="Processing chunks")):
            available_cols = [c for c in columns_to_keep if c in chunk.columns]
            
            if 'v024' in available_cols:
                odisha_chunk = chunk[chunk['v024'] == 21][available_cols]
                
                if not odisha_chunk.empty:
                    chunks.append(odisha_chunk)
                    total_rows += len(odisha_chunk)

    if chunks:
        data = pd.concat(chunks, ignore_index=True)
        data.to_csv(output_csv, index=False)
        print(f"\nSuccess! Saved {data.shape[0]} rows and {data.shape[1]} columns to {output_csv}")
        print(f"\nColumn fill rates:")
        for col in data.columns:
            fill = data[col].notna().mean() * 100
            print(f"  {col:8s}: {fill:5.1f}%")
    else:
        print("No records found for Odisha.")
        
except Exception as e:
    print(f"Error: {e}")
