import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib

dataset_path = "odisha_ready_for_ml.csv"
print("Loading preprocessed dataset for final model export...")
df = pd.read_csv(dataset_path)

X = df.drop(columns=['Anaemia_Target'])
y = df['Anaemia_Target']

# Identify categorical features for CatBoost
continuous_cols = ['v012', 'v445', 'v208'] 
cat_features = [i for i, col in enumerate(X.columns) if col not in continuous_cols]

# We want the final deployed model to learn from the ENTIRE dataset using SMOTE
# so it has the absolute maximum amount of training data possible before hitting production.
print("Applying SMOTE to balance the entire dataset...")
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"Original shape: {X.shape}. SMOTE balanced shape: {X_smote.shape}")

print("Training final Production CatBoost Model...")
# Retrain the CatBoost model directly (skipping the base-learners for the lightweight production app)
# A standalone CatBoost model is much faster for a web interface and handles the categories natively
final_model = CatBoostClassifier(
    iterations=800, 
    depth=6, 
    learning_rate=0.05, 
    loss_function='Logloss', 
    eval_metric='Accuracy',
    cat_features=cat_features,
    random_seed=42, 
    verbose=100
)

# Convert all categorical columns back to INT for CatBoost categorical parsing
for col in X_smote.columns:
    if col not in continuous_cols:
        X_smote[col] = X_smote[col].astype(int)

final_model.fit(X_smote, y_smote)

print("Saving model to disk as 'final_anaemia_model.cbm'...")
# Save the model using CatBoost's native format for high performance
final_model.save_model("final_anaemia_model.cbm")

# Also save the exact column names so the Streamlit app knows the exact feature order
joblib.dump(list(X.columns), "model_features.pkl")

print("Production model exported successfully!")
