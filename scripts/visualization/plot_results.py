import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots using standard matplotlib functions
plt.style.use('default')  # Basic style
# Define slightly custom colors
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

data_path = "final_model_comparison.csv"
df = pd.read_csv(data_path)

# 1. Plot: Sensitivity vs Specificity (The Balance)
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(df))

plt.bar([i - bar_width/2 for i in index], df['Sensitivity (Recall)'], bar_width, label='Sensitivity', color=colors[0])
plt.bar([i + bar_width/2 for i in index], df['Specificity'], bar_width, label='Specificity', color=colors[1])

plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Score (0 to 1)', fontsize=12, fontweight='bold')
plt.title('Sensitivity vs Specificity Comparison (Imbalance Resolution)', fontsize=14, fontweight='bold')
plt.xticks(index, df['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('sensitivity_specificity_comparison.png', dpi=300)
print("Saved: sensitivity_specificity_comparison.png")

# 2. Plot: AUC Score Comparison
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['AUC'], color=colors[2])
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('AUC Score', fontsize=12, fontweight='bold')
plt.title('Area Under the ROC Curve (AUC) Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.4, 0.65) # Zoom in to see differences clearly
plt.tight_layout()
plt.savefig('auc_comparison.png', dpi=300)
print("Saved: auc_comparison.png")

print("Visualizations generated successfully!")
