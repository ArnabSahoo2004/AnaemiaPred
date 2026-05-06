"""
V2 Preprocessing Pipeline:
- Expanded 28-feature set (16 original + 12 new)
- Feature engineering (interaction terms + Shield Score)
- Option for moderate+severe target definition
- Proper KNN imputation and outlier handling
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

input_csv = "e:/FRP PROJECT/Aneamia Detection/data/odisha_womens_features_v2.csv"
output_csv = "e:/FRP PROJECT/Aneamia Detection/data/odisha_ready_for_ml_v2.csv"

print("Loading expanded V2 feature dataset...")
df = pd.read_csv(input_csv)
print(f"Original shape: {df.shape}")

# ============================================================
# 1. TARGET VARIABLE DEFINITION
# ============================================================
print("\n1. Mapping Target Variable from raw Hemoglobin (v453)...")
df = df[df['v453'] < 990]  # Remove invalid hemoglobin codes
df = df.dropna(subset=['v453', 'v213'])

# --- Standard WHO definition (same as V1) ---
def check_anaemia_standard(row):
    hb = row['v453']
    preg = row['v213']
    if preg == 1 and hb < 110: return 1
    elif preg == 0 and hb < 120: return 1
    else: return 0

# --- Moderate+Severe only (stricter, stronger signal) ---
def check_anaemia_moderate_severe(row):
    hb = row['v453']
    preg = row['v213']
    # Moderate: Hb < 10.0 for non-pregnant, < 10.0 for pregnant
    # Severe: Hb < 7.0 (automatically captured)
    if hb < 100: return 1  # Moderate or severe
    else: return 0

# Use Moderate+Severe as the primary target for stronger signal
df['Anaemia_Target'] = df.apply(check_anaemia_moderate_severe, axis=1)
df['Anaemia_Standard'] = df.apply(check_anaemia_standard, axis=1)

print(f"\nStandard Target Distribution (Ignored):")
print(df['Anaemia_Standard'].value_counts(normalize=True))
print(f"\nModerate+Severe Target Distribution (Primary):")
print(df['Anaemia_Target'].value_counts(normalize=True))

# CRITICAL: Drop hemoglobin to prevent target leakage
print("\nCRITICAL: Dropping Hemoglobin (v453) to prevent data leakage!")
df.drop(columns=['v453'], inplace=True, errors='ignore')

# Drop the standard column
df.drop(columns=['Anaemia_Standard'], inplace=True, errors='ignore')

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n2. Engineering new composite features...")

# Shield Score: Combines wealth, education, diet into a single protective index
# Higher score = more protection against anemia
wealth = df['v190'].fillna(3)  # 1-5 scale
education = df['v106'].fillna(1)  # 0-3 scale
eggs = df['v414e'].fillna(0)  # 0-1 binary
meat = df['v414f'].fillna(0)  # 0-1 binary

# Normalize each to 0-1 range, then combine
df['Shield_Score'] = (
    (wealth - 1) / 4 * 0.35 +          # Wealth contributes 35%
    (education) / 3 * 0.25 +            # Education contributes 25%
    eggs * 0.20 +                        # Egg intake contributes 20%
    meat * 0.20                          # Meat intake contributes 20%
)

# BMI × Age interaction (biological synergy)
df['BMI_Age_Interaction'] = df['v445'].fillna(df['v445'].median()) * df['v012'] / 1000

# Maternal burden score: pregnancies + children / age
df['Maternal_Burden'] = (df['v201'].fillna(0) + df['v208'].fillna(0)) / df['v012'].clip(lower=15)

# Healthcare access score (higher = more barriers)
df['Healthcare_Barrier'] = df['v467b'].fillna(1) + df['v467d'].fillna(1)

print(f"  Created: Shield_Score, BMI_Age_Interaction, Maternal_Burden, Healthcare_Barrier")

# ============================================================
# 3. OUTLIER REMOVAL (Z-score on continuous columns)
# ============================================================
print("\n3. Removing outliers using Z-score (threshold=3) for continuous columns...")
continuous_columns = [col for col in ['v445', 'v438', 'v012'] if col in df.columns]
if continuous_columns:
    df_valid = df.dropna(subset=continuous_columns)
    z_scores = stats.zscore(df_valid[continuous_columns])
    abs_z = np.abs(z_scores)
    keep_idx = df_valid[(abs_z < 3).all(axis=1)].index
    nan_idx = df[df[continuous_columns].isna().any(axis=1)].index
    df = pd.concat([df.loc[keep_idx], df.loc[nan_idx]])

# ============================================================
# 4. KNN IMPUTATION
# ============================================================
print("\n4. Imputing missing values using KNN (k=5)...")
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed, columns=df.columns)

print(f"\nFinal cleaned shape: {df.shape}")
print(f"\nAll Predictors ({len(df.columns)-1} features):")
for col in df.columns[:-1]:
    print(f"  {col}")

print(f"\nAnaemia Class Distribution:")
print(df['Anaemia_Target'].value_counts(normalize=True))

# Save
df.to_csv(output_csv, index=False)
print(f"\nSaved V2 dataset to {output_csv}")
