import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

input_csv = "odisha_womens_data.csv"
output_csv = "odisha_cleaned_data.csv"

# Load the lightweight Odisha dataset
print("Loading Odisha dataset...")
df = pd.read_csv(input_csv)
print(f"Original shape: {df.shape}")

# 1. Define Target Variable
# v042 is the Anaemia level categorical variable in DHS
# 0 = Not anaemic, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Not tested/missing
print("\nMapping Target Variable...")
# Since v042 simply indicates all respondents were tested (value 1), we must generate the Anaemia Target manually
# according to literal WHO guidelines using Hemoglobin values (v453):
# v453 is provided in g/dL * 10 (so 120 = 12.0 g/dL)
# Pregnant women (v213 == 1): Anaemic if Hb < 11.0 g/dL (v453 < 110)
# Non-pregnant women (v213 == 0): Anaemic if Hb < 12.0 g/dL (v453 < 120)

print("\nMapping Target Variable from raw Hemoglobin (v453)...")
# Drop rows with missing Hemoglobin data to define target 
# v453 code 998 and 999 usually means missing/not measured in DHS
df = df[df['v453'] < 990]
df = df.dropna(subset=['v453', 'v213'])

def check_anaemia(row):
    hb = row['v453']
    preg = row['v213']
    if preg == 1 and hb < 110:
        return 1 # Anaemic
    elif preg == 0 and hb < 120:
        return 1 # Anaemic
    else:
        return 0 # Not Anaemic

df['Anaemia_Target'] = df.apply(check_anaemia, axis=1)
df.drop(columns=['v042'], inplace=True, errors='ignore') 

# 2. Impute Missing Values with KNN (as per Tanzania methodology)
print("\nImputing missing values using KNN...")
# K=5 is standard for KNN Imputing 
imputer = KNNImputer(n_neighbors=5)
# Impute all columns (KNN works on numerics, our dataset is all numeric codes right now)
df_imputed_array = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed_array, columns=df.columns)
print(f"Missing values handled. Current shape: {df.shape}")

# 3. Handle Outliers using Z-Score (as per Tanzania methodology)
# "Outlier values were checked using the Z-score method with a threshold of greater than positive or negative three Standard deviations."
print("\nRemoving outliers using Z-score (threshold=3)...")
numeric_columns = ['v445', 'v453'] # BMI and Hemoglobin are the main continuous variables needing outlier removal
z_scores = stats.zscore(df[numeric_columns])
abs_z_scores = np.abs(z_scores)

# Filter rows where all absolute Z-scores are less than 3
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

print(f"Removed outliers. Final cleaned shape: {df.shape}")
print("\nAnaemia Class Distribution:")
print(df['Anaemia_Target'].value_counts(normalize=True))

# Save the final cleaned, ready-to-model dataset
df.to_csv(output_csv, index=False)
print(f"\nSaved cleaned dataset for modelling to {output_csv}")
