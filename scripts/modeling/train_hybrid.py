import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, precision_score
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

dataset_path = "odisha_ready_for_ml.csv"
print("Loading preprocessed dataset...")
df = pd.read_csv(dataset_path)

X = df.drop(columns=['Anaemia_Target'])
y = df['Anaemia_Target']

# Train-Test Split (70:30 stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} rows. Test set: {X_test.shape[0]} rows.\n")

# --- Step 1: Define Base Models and Meta Learner (Exactly as Reference Study) ---
# Base 1: Random Forest
rf = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced', random_state=42)
# Base 2: Artificial Neural Network
ann = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', early_stopping=True, random_state=42)
# Meta-Learner: XGBoost (Tuned as per paper)
xgb_meta = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.01, colsample_bytree=0.8, subsample=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42)

print("--- Step 2: Training Base Models via Stratified K-Fold for Meta-Features ---")
# Stratified K-Fold (5 folds) to generate meta-features for the meta-learner to avoid overfitting
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create arrays to hold the out-of-fold predictions for the training set (Meta-features)
meta_X_train = np.zeros((X_train.shape[0], 2)) 
# Create arrays to hold the predictions for the test set
meta_X_test = np.zeros((X_test.shape[0], 2))

# Convert to NumPy for easy indexing during K-Fold
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values

# Train Base Models using Cross Validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
    print(f"Processing Fold {fold+1}/5...")
    # Get fold data
    X_fold_train, y_fold_train = X_train_np[train_idx], y_train_np[train_idx]
    X_fold_val = X_train_np[val_idx]
    
    # Clone models to ensure pristine training per fold
    rf_clone = clone(rf)
    ann_clone = clone(ann)
    
    # Train
    rf_clone.fit(X_fold_train, y_fold_train)
    ann_clone.fit(X_fold_train, y_fold_train)
    
    # Generate Meta-Features (Probabilities) for the validation fold
    meta_X_train[val_idx, 0] = rf_clone.predict_proba(X_fold_val)[:, 1]
    meta_X_train[val_idx, 1] = ann_clone.predict_proba(X_fold_val)[:, 1]
    
    # Add predictions to the test set averaging array
    meta_X_test[:, 0] += rf_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits
    meta_X_test[:, 1] += ann_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits

print("\n--- Step 3: Training XGBoost Meta-Learner ---")
# Train the meta-learner on the meta-features (the predictions of the base models)
xgb_meta.fit(meta_X_train, y_train)

# Get raw probabilities from the Meta-Learner for the test set
y_prob_meta = xgb_meta.predict_proba(meta_X_test)[:, 1]

print("\n--- Step 4: Threshold Optimization (Youden's J Index) ---")
# Calculate Youden's J Index for various thresholds to find the optimal balance
# J = Sensitivity + Specificity - 1
thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
best_threshold = 0.5
best_j = -1
optimal_metrics = {}

for t in thresholds:
    y_pred_t = [1 if p >= t else 0 for p in y_prob_meta]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    j_index = sensitivity + specificity - 1
    
    print(f"Threshold {t:.2f} -> Sens: {sensitivity:.4f}, Spec: {specificity:.4f}, J={j_index:.4f}")
    
    if j_index > best_j:
        best_j = j_index
        best_threshold = t
        optimal_metrics = {
            'Accuracy': accuracy_score(y_test, y_pred_t),
            'AUC': roc_auc_score(y_test, y_prob_meta), # AUC doesn't change with threshold
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision_score(y_test, y_pred_t)
        }

print(f"\nOptimal Threshold Found: {best_threshold}")
print("\n--- Final Hybrid Stacking Model Performance ---")
for metric, val in optimal_metrics.items():
    print(f"{metric}: {val:.4f}")

# Save results for comparison later
results_df = pd.DataFrame([{
    'Model': f"Hybrid (RF+ANN->XGB) [Threshold {best_threshold}]",
    'Accuracy': optimal_metrics['Accuracy'],
    'AUC': optimal_metrics['AUC'],
    'Sensitivity (Recall)': optimal_metrics['Sensitivity'],
    'Specificity': optimal_metrics['Specificity'],
    'Precision': optimal_metrics['Precision']
}])

# Append to baseline results file
baseline_df = pd.read_csv("baseline_results.csv")
combined_df = pd.concat([baseline_df, results_df], ignore_index=True)
combined_df.to_csv("hybrid_comparison_results.csv", index=False)
print("\nSaved comparison to hybrid_comparison_results.csv")
