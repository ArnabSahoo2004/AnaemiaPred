import pandas as pd
from tqdm import tqdm

file_path = "IAIR7EFL.DTA"

# Standard DHS Recode Variables we likely need for ML Anemia Detection
# v012 = Age
# v445 = Body Mass Index (BMI)
# v453 = Hemoglobin level (Hb)
# v106 = Highest year of education
# v190 = Wealth index
# v213 = Currently pregnant
# v024 = Region/State (Crucial for Odisha filtering)
columns_to_keep = ['v012', 'v445', 'v453', 'v106', 'v190', 'v213', 'v024']

print(f"Reading file in chunks, extracting only necessary variables: {columns_to_keep}")
chunksize = 100000 
chunks = []

# Load in chunks, using StataReader directly to get a progress bar with known total
try:
    with pd.io.stata.StataReader(file_path, chunksize=chunksize, convert_categoricals=False) as reader:
        for chunk in tqdm(reader, desc="Loading specific columns form DTA"):
            # Only keep the columns we actually care about to save memory
            # Check if columns exist in this dataset version before filtering
            available_cols = [c for c in columns_to_keep if c in chunk.columns]
            chunks.append(chunk[available_cols])

    print("\nConcatenating chunks...")
    data = pd.concat(chunks, ignore_index=True)

    print(f"\nFinal Shape: {data.shape}")
    print(f"Extracted Columns: {list(data.columns)}")
    print(data.head())
    
except Exception as e:
    print(f"Error reading chunk: {e}")
    # Let's read just the first row to see exactly what columns exist in your specific IAIR7EFL file
    print("\nAttempting to read just the header to find correct column names...")
    sample = pd.read_stata(file_path, convert_categoricals=False, convert_dates=False)
