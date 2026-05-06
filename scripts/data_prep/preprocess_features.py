import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

input_csv = "odisha_womens_features.csv"
output_csv = "odisha_ready_for_ml.csv"

print("Loading expanded Odisha feature dataset...")
df = pd.read_csv(input_csv)
print(f"Original shape: {df.shape}")

print("\n1. Mapping Target Variable from raw Hemoglobin (v453)...")
# Drop rows with missing Hemoglobin data since we can't define target without it
df = df[df['v453'] < 990]
df = df.dropna(subset=['v453', 'v213'])

def check_anaemia(row):
    hb = row['v453']
    preg = row['v213']
    if preg == 1 and hb < 110: return 1 # Anaemic
    elif preg == 0 and hb < 120: return 1 # Anaemic
    else: return 0 # Not Anaemic

df['Anaemia_Target'] = df.apply(check_anaemia, axis=1)

print("\nCRITICAL: Dropping Hemoglobin (v453) to prevent data leakage!")
# By dropping v453, the model is forced to learn patterns across demographics
# and other accessible health indicators like BMI, Pregnancy, Wealth, etc.
df.drop(columns=['v453'], inplace=True, errors='ignore')

print("\n2. Removing outliers using Z-score (threshold=3) for BMI...")
# Only numeric continuous columns get Z-score. The rest are categorical codes.
numeric_columns = [col for col in ['v445'] if col in df.columns]
if numeric_columns:
    # We only compute Z-score on non-null rows to avoid dropping everything if there's NaNs
    # But since we will KNN impute later, we should handle outliers first on valid values
    df_valid_bmi = df.dropna(subset=numeric_columns)
    z_scores = stats.zscore(df_valid_bmi[numeric_columns])
    abs_z_scores = np.abs(z_scores)
    
    # Identify indices to KEEP
    keep_indices = df_valid_bmi[(abs_z_scores < 3).all(axis=1)].index
    # Also keep all rows that HAD NaNs (we will impute them later, they are not outliers)
    nan_indices = df[df[numeric_columns].isna().any(axis=1)].index
    
    df = pd.concat([df.loc[keep_indices], df.loc[nan_indices]])

print("\n3. Imputing missing predictor values using KNN...")
# K=5 as per Tanzania methodology
imputer = KNNImputer(n_neighbors=5)
df_imputed_array = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed_array, columns=df.columns)

print(f"\nFinal cleaned shape ready for modelling: {df.shape}")
print("\nFinal Predictors (X):")
print(list(df.columns[:-1])) # Exclude target

print("\nAnaemia Class Distribution:")
print(df['Anaemia_Target'].value_counts(normalize=True))

# Save the final cleaned, ready-to-model dataset
df.to_csv(output_csv, index=False)
print(f"\nSaved cost-aware modelling dataset to {output_csv}")
