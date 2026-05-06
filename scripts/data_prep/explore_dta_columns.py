"""
Scan the NFHS-5 DTA file to identify all available columns
and find promising non-invasive features for anemia prediction.
"""
import pandas as pd
import os

file_path = "e:/FRP PROJECT/Aneamia Detection/data/IAIR7EFL.DTA"

print("Reading DTA file metadata (first chunk only)...")
with pd.io.stata.StataReader(file_path, chunksize=1000, convert_categoricals=False) as reader:
    chunk = next(iter(reader))

print(f"Total columns in DTA file: {len(chunk.columns)}")
print(f"Sample rows: {len(chunk)}")

# Filter for Odisha to see what data looks like
odisha = chunk[chunk['v024'] == 21]
print(f"Odisha rows in this chunk: {len(odisha)}")

# Key variable groups we want to explore (non-invasive, cost-aware)
# Reference: https://dhsprogram.com/pubs/pdf/DHSG4/Recode7_DHS_10Sep2018_DHSG4.pdf
target_vars = {
    # --- ALREADY USING ---
    'v012': 'Age',
    'v445': 'BMI',
    'v106': 'Education level',
    'v190': 'Wealth index',
    'v213': 'Currently pregnant',
    'v404': 'Currently breastfeeding',
    'v208': 'Births in last 5 years',
    'v024': 'Region',
    'v025': 'Urban/Rural',
    'v130': 'Religion',
    'v481': 'Health insurance',
    'v157': 'Reads newspaper',
    'v158': 'Listens to radio',
    'v159': 'Watches TV',
    'v414e': 'Eats eggs',
    'v414f': 'Eats meat',
    'v453': 'Hemoglobin (TARGET ONLY)',
    
    # --- POTENTIAL NEW FEATURES (Non-invasive) ---
    'v113': 'Source of drinking water',
    'v116': 'Type of toilet facility',
    'v127': 'Main floor material',
    'v161': 'Type of cooking fuel',
    'v136': 'Number of household members',
    'v137': 'Number of children 5 and under',
    'v201': 'Total children ever born',
    'v212': 'Age at first birth',
    'v222': 'Months since last birth (birth interval)',
    'v228': 'Time since last menstrual period',
    'v301': 'Knowledge of any contraceptive method',
    'v312': 'Current contraceptive method',
    'v463a': 'Smokes cigarettes',
    'v463b': 'Smokes pipe',
    'v463e': 'Chews tobacco',
    'v463z': 'Uses no tobacco',
    'v467b': 'Distance to health facility',
    'v467d': 'Getting to health facility a big problem',
    'v414g': 'Eats dark green leafy vegetables',
    'v414h': 'Eats fruits',
    'v414i': 'Eats fish',
    'v414j': 'Eats beans/lentils',
    'v414k': 'Eats milk/cheese',
    'v414l': 'Eats oil/fats',
    'v414n': 'Eats nuts/seeds',
    'v409': 'Drinks plain water',
    's712a': 'Iron tablets/syrup during pregnancy',
    's712b': 'Number of days took iron tablets',
    'v426': 'When child put to breast',
    'v437': 'Respondent weight (kg)',
    'v438': 'Respondent height (cm)',
    'v440': 'Height/Age percentile',
    'v501': 'Marital status',
    'v502': 'Currently/formerly married',
    'v714': 'Currently working',
    'v717': 'Occupation type',
    'v731': 'Working in last 12 months',
    'v151': 'Sex of household head',
    'v152': 'Age of household head',
    'v171a': 'Has mobile phone',
    'v169a': 'Owns a house',
    'v171b': 'Uses internet',
    'v191': 'Wealth index factor score',
    'v467e': 'Not wanting to go alone to health facility',
}

print("\n" + "="*80)
print("CHECKING WHICH POTENTIAL FEATURES EXIST IN THE DATASET")
print("="*80)

existing_new = {}
missing = []

for var, desc in target_vars.items():
    if var in chunk.columns:
        # Get basic stats
        col = chunk[var]
        non_null = col.notna().sum()
        pct = (non_null / len(chunk)) * 100
        unique = col.nunique()
        existing_new[var] = {
            'description': desc,
            'non_null_pct': pct,
            'unique_values': unique,
            'sample_values': sorted(col.dropna().unique()[:10].tolist())
        }
    else:
        missing.append(f"{var}: {desc}")

print(f"\n--- AVAILABLE ({len(existing_new)} vars) ---")
for var, info in sorted(existing_new.items()):
    already = "  [ALREADY USING]" if var in ['v012','v445','v106','v190','v213','v404','v208','v024','v025','v130','v481','v157','v158','v159','v414e','v414f','v453'] else "  [NEW]"
    print(f"{var:8s} | {info['description']:45s} | {info['non_null_pct']:5.1f}% filled | {info['unique_values']:4d} unique{already}")

print(f"\n--- NOT FOUND ({len(missing)} vars) ---")
for m in missing:
    print(f"  {m}")
