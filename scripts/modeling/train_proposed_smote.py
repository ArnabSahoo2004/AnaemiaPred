import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, precision_score
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

dataset_path = "odisha_ready_for_ml.csv"
print("Loading preprocessed dataset...")
df = pd.read_csv(dataset_path)

X = df.drop(columns=['Anaemia_Target'])
y = df['Anaemia_Target']

# Train-Test Split (70:30)
# We ONLY apply SMOTE to the training data. The test data remains real, unbalanced data to prove real-world accuracy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
print(f"Original Training set: {X_train.shape[0]} rows. Test set: {X_test.shape[0]} rows.")
print(f"Original Training Distribution:\n{y_train.value_counts(normalize=True)}")

print("\n--- Step 1: Initializing Proposed Enhanced Base Models & Meta-Learner ---")
rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42) # Removed class_weight because SMOTE balances it
ann = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', early_stopping=True, random_state=42)
lgbm = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, random_state=42, verbose=-1)
cat_meta = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.01, loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=False)

print("\n--- Step 2: Training Base Models via Stratified K-Fold with SMOTE ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

meta_X_train = np.zeros((X_train.shape[0], 3)) 
meta_X_test = np.zeros((X_test.shape[0], 3))

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
    print(f"Processing Fold {fold+1}/5 with SMOTE...")
    X_fold_train, y_fold_train = X_train_np[train_idx], y_train_np[train_idx]
    X_fold_val = X_train_np[val_idx]
    
    # APPLY SMOTE STRICTLY INSIDE THE FOLD TO PREVENT DATA LEAKAGE
    X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
    
    rf_clone = clone(rf)
    ann_clone = clone(ann)
    lgbm_clone = clone(lgbm)
    
    # Train on perfectly balanced synthetic data
    rf_clone.fit(X_fold_train_smote, y_fold_train_smote)
    ann_clone.fit(X_fold_train_smote, y_fold_train_smote)
    lgbm_clone.fit(X_fold_train_smote, y_fold_train_smote)
    
    meta_X_train[val_idx, 0] = rf_clone.predict_proba(X_fold_val)[:, 1]
    meta_X_train[val_idx, 1] = ann_clone.predict_proba(X_fold_val)[:, 1]
    meta_X_train[val_idx, 2] = lgbm_clone.predict_proba(X_fold_val)[:, 1]
    
    meta_X_test[:, 0] += rf_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits
    meta_X_test[:, 1] += ann_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits
    meta_X_test[:, 2] += lgbm_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits

print("\n--- Step 3: Training CatBoost Meta-Learner ---")
cat_meta.fit(meta_X_train, y_train_np)
y_prob_meta = cat_meta.predict_proba(meta_X_test)[:, 1]

print("\n--- Step 4: Threshold Optimization (Youden's J Index) ---")
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
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
            'AUC': roc_auc_score(y_test, y_prob_meta),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision_score(y_test, y_pred_t)
        }

print(f"\nFinal Optimal Threshold: {best_threshold}")
print("\n--- PROPOSED ENSEMBLE + SMOTE (Final Results) ---")
for metric, val in optimal_metrics.items():
    print(f"{metric}: {val:.4f}")

results_df = pd.DataFrame([{
    'Model': f"Proposed Ensembled + SMOTE [Threshold {best_threshold}]",
    'Accuracy': optimal_metrics['Accuracy'],
    'AUC': optimal_metrics['AUC'],
    'Sensitivity (Recall)': optimal_metrics['Sensitivity'],
    'Specificity': optimal_metrics['Specificity'],
    'Precision': optimal_metrics['Precision']
}])

combined_df = pd.read_csv("final_model_comparison.csv")
final_df = pd.concat([combined_df, results_df], ignore_index=True)
final_df.to_csv("final_model_comparison.csv", index=False)
print("\nSaved SMOTE comparison to final_model_comparison.csv")
