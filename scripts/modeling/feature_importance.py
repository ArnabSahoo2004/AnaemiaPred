import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.ensemble import RandomForestClassifier

# Setup styles
plt.style.use('default')

import os
dataset_path = "../../data/odisha_ready_for_ml.csv"
if not os.path.exists(dataset_path):
    dataset_path = "e:/FRP PROJECT/Aneamia Detection/data/odisha_ready_for_ml.csv"
    
print("Loading preprocessed dataset for Feature Importance Analysis...")
df = pd.read_csv(dataset_path)

X = df.drop(columns=['Anaemia_Target'])
y = df['Anaemia_Target']

# Mapping DHS column names to human-readable names
feature_names_mapping = {
    'v012': 'Age',
    'v445': 'BMI',
    'v106': 'Education Level',
    'v190': 'Wealth Index',
    'v213': 'Currently Pregnant',
    'v404': 'Currently Breastfeeding',
    'v208': 'Births in Last 5 Years',
    'v024': 'Region (Odisha)',
    'v025': 'Residence (Urban/Rural)',
    'v130': 'Religion',
    'v481': 'Health Insurance Coverage',
    'v157': 'Reads Newspaper',
    'v158': 'Listens to Radio',
    'v159': 'Watches TV',
    'v414e': 'Eats Eggs Frequency',
    'v414f': 'Eats Meat Frequency'
}

X.rename(columns=feature_names_mapping, inplace=True)

print("Training Random Forest to extract Gini feature importances...")
# We use Random Forest because it uses the Gini index out-of-the-box which is standard for feature importance
rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
rf.fit(X, y)

# Extract and Sort Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [X.columns[i] for i in indices]
sorted_importances = [importances[i] for i in indices]

# Display to console
print("\n--- Feature Importance Ranking ---")
for f, imp in zip(sorted_features, sorted_importances):
    print(f"{f}: {imp:.4f}")

# Plotting
plt.figure(figsize=(10, 8))
plt_sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")

# Add the red dashed threshold line exactly like the reference paper
plt.axvline(x=0.02, color='red', linestyle='--', label='Threshold (>0.02)')
plt.legend(loc='lower right')

plt.title('Feature Importance for Predicting Anaemia (Odisha Women)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score (Gini Index)', fontsize=12, fontweight='bold')
plt.ylabel('Cost-Aware Features', fontsize=12, fontweight='bold')
plt.tight_layout()

# Save plot appropriately accounting for directory execution context
import os
output_path = "../../results/feature_importance.png"
if not os.path.exists("../../results"):
    output_path = "e:/FRP PROJECT/Aneamia Detection/results/feature_importance.png"
if not os.path.exists(os.path.dirname(output_path)):
    output_path = "feature_importance.png" # fallback

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\nSaved chart to feature_importance.png")
