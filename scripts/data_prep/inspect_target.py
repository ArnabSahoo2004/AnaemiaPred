import pandas as pd

input_csv = "odisha_womens_data.csv"

# Load the lightweight Odisha dataset
print("Loading Odisha dataset...")
df = pd.read_csv(input_csv)

print("\nv042 (Anaemia level) raw values before mapping:")
print(df['v042'].value_counts(dropna=False))

print("\nv453 (Hemoglobin level) summary:")
print(df['v453'].describe())
