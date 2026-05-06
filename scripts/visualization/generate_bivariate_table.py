import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency
import os

# Ensure we are reading from the correct path
dataset_path = "../../data/odisha_ready_for_ml.csv"
if not os.path.exists(dataset_path):
    dataset_path = "data/odisha_ready_for_ml.csv" # fallback
if not os.path.exists(dataset_path):
    dataset_path = "e:/FRP PROJECT/Aneamia Detection/data/odisha_ready_for_ml.csv"

print("Loading data for Bivariate Analysis...")
df = pd.read_csv(dataset_path)

# Feature dictionaries mapping column codes to human-readable names and types
feature_types = {
    'v012': ('Age', 'Continuous'),
    'v445': ('BMI', 'Continuous'),
    'v208': ('Births in 5 yrs', 'Continuous'),
    'v106': ('Education Level', 'Categorical'),
    'v190': ('Wealth Index', 'Categorical'),
    'v213': ('Pregnant', 'Categorical'),
    'v404': ('Breastfeeding', 'Categorical'),
    'v024': ('Region', 'Categorical'),
    'v025': ('Residence Type', 'Categorical'),
    'v130': ('Religion', 'Categorical'),
    'v481': ('Health Insurance', 'Categorical'),
    'v157': ('Reads Newspaper', 'Categorical'),
    'v158': ('Listens to Radio', 'Categorical'),
    'v159': ('Watches TV', 'Categorical'),
    'v414e': ('Eats Eggs', 'Categorical'),
    'v414f': ('Eats Meat', 'Categorical')
}

target = 'Anaemia_Target'
results = []

for col, (readable_name, f_type) in feature_types.items():
    if f_type == 'Continuous':
        # ANOVA
        group0 = df[df[target] == 0][col].dropna()
        group1 = df[df[target] == 1][col].dropna()
        f_stat, p_val = f_oneway(group0, group1)
        test_name = "ANOVA"
        test_val = f_stat
        dof = 1 # k-1 where k is number of groups (2)
    else:
        # Chi-Square
        contingency_table = pd.crosstab(df[col], df[target])
        chi2_stat, p_val, dof, ex = chi2_contingency(contingency_table)
        test_name = "Chi-Square"
        test_val = chi2_stat
        
    significant = "TRUE" if p_val <= 0.05 else "FALSE"
    
    # Format p-value cleanly
    if p_val < 0.0001:
        p_val_str = f"{p_val:.4E}"
    else:
        p_val_str = f"{p_val:.4f}"
        
    results.append({
        "Feature": readable_name,
        "Test": test_name,
        "Test Value": f"{test_val:.4f}",
        "p-value": p_val_str,
        "Degrees of Freedom": dof,
        "Significant (p <= 0.05)": significant
    })

# Convert to DataFrame and sort by p-value
results_df = pd.DataFrame(results)
# Sort to bring TRUE significance to the top
results_df['p_val_num'] = results_df['p-value'].astype(float)
results_df = results_df.sort_values(by='p_val_num', ascending=True).drop('p_val_num', axis=1)

print("\n--- Bivariate Analysis Results ---")
print(results_df)

# Render as Image
fig, ax = plt.subplots(figsize=(12, 6)) # Adjust size as needed
ax.axis('off')
ax.axis('tight')

# Create a clear title above the table
plt.suptitle('Table 1\nBivariate Analysis of anaemia status among adult women in all features', 
             fontweight='bold', x=0.05, y=0.95, ha='left', fontsize=12)

# Create table
table = ax.table(cellText=results_df.values, 
                 colLabels=results_df.columns, 
                 cellLoc='left', 
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the headers to look like the reference image
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f2f2f2')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5) # Thicker top/bottom borders for header
    else:
        # Remove vertical borders to mimic academic style
        cell.set_edgecolor('none')
        # Add thin line at the bottom
        if row == len(results_df):
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5) 
            cell.visible_edges = 'B'

# Save the table image
output_path = "../../results/bivariate_analysis_table.png"
if not os.path.exists("../../results"):
    output_path = "e:/FRP PROJECT/Aneamia Detection/results/bivariate_analysis_table.png"

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
