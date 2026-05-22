"""
Statistical Significance Analysis for Section 4 of Research Paper
Runs ANOVA (numerical) and Chi-square (categorical) tests on all 32 features
and generates a publication-quality significance table.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(r'e:\FRP PROJECT\Aneamia Detection\data\odisha_ready_for_ml_v2.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target distribution:\n{df['Anaemia_Target'].value_counts()}")

# ─── Feature metadata ────────────────────────────────────────────────────────
FEATURE_NAMES = {
    'v012': 'Age (Years)',
    'v445': 'Body Mass Index (BMI)',
    'v438': 'Height (cm)',
    'v106': 'Education Level',
    'v190': 'Wealth Index',
    'v213': 'Currently Pregnant',
    'v404': 'Breastfeeding Status',
    'v208': 'Births in Last 5 Years',
    'v024': 'State Code (Odisha)',
    'v025': 'Residence (Urban/Rural)',
    'v130': 'Religion',
    'v481': 'Health Insurance',
    'v157': 'Reads Newspaper',
    'v158': 'Listens to Radio',
    'v159': 'Watches Television',
    'v414e': 'Egg Consumption Frequency',
    'v414f': 'Meat Consumption Frequency',
    'v113': 'Drinking Water Source',
    'v116': 'Toilet Facility Type',
    'v161': 'Cooking Fuel Type',
    'v136': 'Household Size',
    'v137': 'Children Under 5 in HH',
    'v201': 'Total Children Ever Born',
    'v501': 'Marital Status',
    'v312': 'Contraceptive Use',
    'v467b': 'Distance to Hospital',
    'v467d': 'Hospital Access Difficulty',
    'v463z': 'No Tobacco Use',
    'Shield_Score': 'Socio-Nutritional Shield Score',
    'BMI_Age_Interaction': 'BMI × Age Interaction',
    'Maternal_Burden': 'Maternal Burden Score',
    'Healthcare_Barrier': 'Healthcare Access Barrier Index',
}

# Continuous (ANOVA) vs Categorical (Chi-square)
NUMERICAL = ['v012', 'v445', 'v438', 'BMI_Age_Interaction',
             'Shield_Score', 'Maternal_Burden', 'Healthcare_Barrier']
CATEGORICAL = [f for f in FEATURE_NAMES if f not in NUMERICAL]

CATEGORY_GROUPS = {
    'v012': 'Biological', 'v445': 'Biological', 'v438': 'Biological',
    'BMI_Age_Interaction': 'Biological',
    'v213': 'Reproductive', 'v404': 'Reproductive', 'v208': 'Reproductive',
    'v201': 'Reproductive', 'v137': 'Reproductive', 'Maternal_Burden': 'Reproductive',
    'v312': 'Reproductive',
    'v190': 'Socio-Economic', 'v106': 'Socio-Economic', 'v157': 'Socio-Economic',
    'v158': 'Socio-Economic', 'v159': 'Socio-Economic', 'v414e': 'Socio-Economic',
    'v414f': 'Socio-Economic', 'v116': 'Socio-Economic', 'v161': 'Socio-Economic',
    'v501': 'Socio-Economic', 'v136': 'Socio-Economic', 'Shield_Score': 'Socio-Economic',
    'v481': 'Healthcare Access', 'v467b': 'Healthcare Access', 'v467d': 'Healthcare Access',
    'Healthcare_Barrier': 'Healthcare Access',
    'v113': 'Household', 'v025': 'Household', 'v130': 'Household',
    'v463z': 'Lifestyle', 'v024': 'Geographic',
}

# ─── Run statistical tests ───────────────────────────────────────────────────
anaemic = df[df['Anaemia_Target'] == 1.0]
healthy  = df[df['Anaemia_Target'] == 0.0]

results = []

for feat, label in FEATURE_NAMES.items():
    if feat not in df.columns:
        continue

    group = CATEGORY_GROUPS.get(feat, 'Other')

    if feat in NUMERICAL:
        test_type = 'ANOVA (F-test)'
        g1 = anaemic[feat].dropna()
        g2 = healthy[feat].dropna()
        f_stat, p_val = stats.f_oneway(g1, g2)
        stat_val = f_stat
        mean_anaemic = f"{g1.mean():.2f} ± {g1.std():.2f}"
        mean_healthy  = f"{g2.mean():.2f} ± {g2.std():.2f}"
    else:
        test_type = 'Chi-square'
        ct = pd.crosstab(df[feat], df['Anaemia_Target'])
        if ct.shape[1] < 2:
            continue
        chi2, p_val, _, _ = stats.chi2_contingency(ct)
        stat_val = chi2
        mean_anaemic = f"n={len(anaemic[feat].dropna())}"
        mean_healthy  = f"n={len(healthy[feat].dropna())}"

    results.append({
        'Feature': label,
        'Category': group,
        'Test': test_type,
        'Statistic': round(stat_val, 3),
        'p-value': p_val,
        'p-value (fmt)': f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
        'Significant': p_val <= 0.05,
        'Anaemic Group': mean_anaemic,
        'Healthy Group': mean_healthy,
    })

results_df = pd.DataFrame(results).sort_values(['Category', 'p-value'])
sig_df = results_df[results_df['Significant'] == True]
not_sig_df = results_df[results_df['Significant'] == False]

print(f"\nSIGNIFICANT features (p <= 0.05): {len(sig_df)}")
print(sig_df[['Feature', 'Category', 'Test', 'p-value (fmt)']].to_string(index=False))
print(f"\nNOT SIGNIFICANT (p > 0.05): {len(not_sig_df)}")
print(not_sig_df[['Feature', 'Category', 'p-value (fmt)']].to_string(index=False))

# ─── Save CSV for reference ──────────────────────────────────────────────────
results_df[['Feature','Category','Test','Statistic','p-value (fmt)','Significant']].to_csv(
    r'e:\FRP PROJECT\Aneamia Detection\results\feature_significance_table.csv', index=False)
print("\nSaved: results/feature_significance_table.csv")

# ─── Generate publication-quality table image ────────────────────────────────
fig, ax = plt.subplots(figsize=(16, len(results_df) * 0.42 + 1.5))
ax.axis('off')

table_data = []
col_labels = ['#', 'Feature', 'Category', 'Statistical Test', 'Test Statistic', 'p-value', 'Significant']

for i, row in results_df.iterrows():
    sig_str = "YES" if row['Significant'] else "NO"
    table_data.append([
        len(table_data) + 1,
        row['Feature'],
        row['Category'],
        row['Test'],
        f"{row['Statistic']:.3f}",
        row['p-value (fmt)'],
        sig_str
    ])

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='left'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1a1a2e')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Style rows by significance
for i, row in enumerate(table_data):
    is_sig = row[6] == "YES"
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        if is_sig:
            cell.set_facecolor('#e8f5e9')  # light green
        else:
            cell.set_facecolor('#fafafa')
        # Color the Significant column
        if j == 6:
            if is_sig:
                table[i+1, j].set_text_props(color='#2e7d32', fontweight='bold')
            else:
                table[i+1, j].set_text_props(color='#c62828')

# Column widths
table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6])

# Legend patches
sig_patch   = mpatches.Patch(color='#e8f5e9', label='Statistically Significant (p ≤ 0.05)')
nosig_patch = mpatches.Patch(color='#fafafa', label='Not Significant (p > 0.05)')
ax.legend(handles=[sig_patch, nosig_patch], loc='upper right', fontsize=9)

plt.title(
    'Table 1: Statistical Significance of Features Associated with Moderate-to-Severe Anaemia\n'
    '(NFHS-5 Odisha Dataset, n = 26,687 Women aged 15–49)',
    fontsize=11, fontweight='bold', pad=15
)

plt.tight_layout()
out_path = r'e:\FRP PROJECT\Aneamia Detection\results\v2_significance_table.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
print("\n[SUCCESS] Significance analysis complete!")
