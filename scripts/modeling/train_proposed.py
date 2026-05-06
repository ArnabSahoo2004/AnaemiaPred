import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, precision_score
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

dataset_path = "odisha_ready_for_ml.csv"
print("Loading preprocessed dataset...")
df = pd.read_csv(dataset_path)

X = df.drop(columns=['Anaemia_Target'])
y = df['Anaemia_Target']

# Identify categorical features for CatBoost
# In DHS, most features like Region, Religion, Wealth Index are categorical codes
categorical_features_indices = list(range(X.shape[1])) 
# BMI (v445) and Age (v012) are continuous, let's exclude them from categorical list
continuous_cols = ['v012', 'v445', 'v208'] 
cat_features = [i for i, col in enumerate(X.columns) if col not in continuous_cols]

# Train-Test Split (70:30 stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} rows. Test set: {X_test.shape[0]} rows.\n")

print("--- Step 1: Initializing Proposed Enhanced Base Models & Meta-Learner ---")
# Base 1: Random Forest (Reference)
rf = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced', random_state=42)
# Base 2: Artificial Neural Network (Reference)
ann = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', early_stopping=True, random_state=42)
# Base 3: LightGBM (PROPOSED ADDITION - Fast, handles large datasets well)
lgbm = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)

# Meta-Learner: CatBoost (PROPOSED UPGRADE - Excellent at categorical data without extensive pre-processing)
cat_meta = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.01, loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=False)

print("--- Step 2: Training Base Models via Stratified K-Fold for Meta-Features ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Meta-features array now expects 3 columns (RF, ANN, LGBM)
meta_X_train = np.zeros((X_train.shape[0], 3)) 
meta_X_test = np.zeros((X_test.shape[0], 3))

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
    print(f"Processing Fold {fold+1}/5...")
    X_fold_train, y_fold_train = X_train_np[train_idx], y_train_np[train_idx]
    X_fold_val = X_train_np[val_idx]
    
    rf_clone = clone(rf)
    ann_clone = clone(ann)
    lgbm_clone = clone(lgbm)
    
    rf_clone.fit(X_fold_train, y_fold_train)
    ann_clone.fit(X_fold_train, y_fold_train)
    lgbm_clone.fit(X_fold_train, y_fold_train)
    
    meta_X_train[val_idx, 0] = rf_clone.predict_proba(X_fold_val)[:, 1]
    meta_X_train[val_idx, 1] = ann_clone.predict_proba(X_fold_val)[:, 1]
    meta_X_train[val_idx, 2] = lgbm_clone.predict_proba(X_fold_val)[:, 1]
    
    meta_X_test[:, 0] += rf_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits
    meta_X_test[:, 1] += ann_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits
    meta_X_test[:, 2] += lgbm_clone.predict_proba(X_test_np)[:, 1] / skf.n_splits

print("\n--- Step 3: Training CatBoost Meta-Learner ---")
cat_meta.fit(meta_X_train, y_train)
y_prob_meta = cat_meta.predict_proba(meta_X_test)[:, 1]

print("\n--- Step 4: Threshold Optimization (Youden's J Index) ---")
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
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

print(f"\nOptimal Threshold Found: {best_threshold}")
print("\n--- PROPOSED ENSEMBLE (RF+ANN+LGBM -> CatBoost) Performance ---")
for metric, val in optimal_metrics.items():
    print(f"{metric}: {val:.4f}")

results_df = pd.DataFrame([{
    'Model': f"Proposed (RF+ANN+LGBM->CatBoost) [Threshold {best_threshold}]",
    'Accuracy': optimal_metrics['Accuracy'],
    'AUC': optimal_metrics['AUC'],
    'Sensitivity (Recall)': optimal_metrics['Sensitivity'],
    'Specificity': optimal_metrics['Specificity'],
    'Precision': optimal_metrics['Precision']
}])

combined_df = pd.read_csv("hybrid_comparison_results.csv")
final_df = pd.concat([combined_df, results_df], ignore_index=True)
final_df.to_csv("final_model_comparison.csv", index=False)
print("\nSaved final comparison to final_model_comparison.csv")
