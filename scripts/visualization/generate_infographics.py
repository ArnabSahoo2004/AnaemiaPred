import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

dataset_path = "odisha_ready_for_ml.csv"
print("Loading data for visualizations...")
df = pd.read_csv(dataset_path)

# --- 1. Target Class Distribution Pie Chart ---
plt.figure(figsize=(8, 8))
class_counts = df['Anaemia_Target'].value_counts()
labels = ['Anaemic (High Risk)', 'Healthy (Low Risk)']
plt.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=[colors[3], colors[2]], wedgeprops={'edgecolor': 'white'})
plt.title('The Class Imbalance Problem in Odisha Women', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution_pie.png', dpi=300)
print("Saved: class_distribution_pie.png")


# --- 2. Feature Correlation Heatmap ---
# Select a subset of interesting features to avoid clutter
cols_to_plot = ['Anaemia_Target', 'v012', 'v445', 'v190', 'v106', 'v213', 'v025']
rename_dict = {
    'Anaemia_Target': 'Anaemia Risk',
    'v012': 'Age',
    'v445': 'BMI',
    'v190': 'Wealth Index',
    'v106': 'Education',
    'v213': 'Pregnant',
    'v025': 'Residence (Rural)'
}
corr_df = df[cols_to_plot].rename(columns=rename_dict)
corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Key Socioeconomic Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
print("Saved: correlation_heatmap.png")


# --- 3. Age vs BMI Density (KDE) Plot ---
# To show the interaction of the top 2 features identified by the model
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['Anaemia_Target'] == 0.0], x='v012', y='v445', cmap="Greens", fill=True, alpha=0.5, label='Healthy')
sns.kdeplot(data=df[df['Anaemia_Target'] == 1.0], x='v012', y='v445', cmap="Reds", fill=True, alpha=0.5, label='Anaemic')
plt.title('Density Landscape: BMI vs Age in Anaemia Risk', fontsize=16, fontweight='bold')
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('BMI', fontsize=12)
# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.5, label='Anaemic'), Patch(facecolor='green', alpha=0.5, label='Healthy')]
plt.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.savefig('age_bmi_density.png', dpi=300)
print("Saved: age_bmi_density.png")


# --- 4. Predictive Confusion Matrices (Simulated from final results) ---
# We know the final metrics to recreate standard 100-patient visualization grids
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Tanzania Hybrid (Sens 80.5%, Spec 26.9%)
tanzania_cm = np.array([[26.9, 73.1], [19.5, 80.5]]) # Percentages
sns.heatmap(tanzania_cm, annot=True, fmt=".1f", cmap="Blues", ax=axes[0], cbar=False, 
            xticklabels=['Predict Healthy', 'Predict Anaemic'], yticklabels=['Actual Healthy', 'Actual Anaemic'])
axes[0].set_title("Tanzania Baseline Model Bias\n(Over-predicts Anaemia)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("AI Prediction")
axes[0].set_ylabel("True Condition")

# Proposed CatBoost + SMOTE (Sens 49.5%, Spec 60.7%)
proposed_cm = np.array([[60.7, 39.3], [50.5, 49.5]]) # Percentages
sns.heatmap(proposed_cm, annot=True, fmt=".1f", cmap="Greens", ax=axes[1], cbar=False,
            xticklabels=['Predict Healthy', 'Predict Anaemic'], yticklabels=['Actual Healthy', 'Actual Anaemic'])
axes[1].set_title("Proposed SMOTE Architecture\n(Balanced Detection)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("AI Prediction")
axes[1].set_ylabel("True Condition")

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300)
print("Saved: confusion_matrix_comparison.png")

print("All new infographics generated successfully!")
