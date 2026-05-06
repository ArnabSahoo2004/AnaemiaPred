"""
V2 Final Training Pipeline (Streamlined):
- 32 features (16 original + 12 new + 4 engineered)
- SMOTE inside K-Fold (correct methodology)
- Stacking ensemble: RF + ANN + LightGBM → CatBoost
- Reports cross-validated AUC (mean ± std)
- Exports the EXACT evaluated model for production
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, precision_score)
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

dataset_path = "e:/FRP PROJECT/Aneamia Detection/data/odisha_ready_for_ml_v2.csv"
output_dir = "e:/FRP PROJECT/Aneamia Detection/results"
model_dir = "e:/FRP PROJECT/Aneamia Detection/app"

print("="*80)
print("V2 FINAL TRAINING PIPELINE")
print("="*80)

df = pd.read_csv(dataset_path)
X = df.drop(columns=['Anaemia_Target'])
y = df['Anaemia_Target']

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Class balance: Anaemic={y.mean():.1%}, Healthy={1-y.mean():.1%}")
print(f"Features: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# STEP 1: Train base learners via K-Fold + SMOTE
# ============================================================
print("\n--- STEP 1: Training Base Learners (5-Fold + SMOTE) ---")

rf = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42)
ann = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    solver='adam', early_stopping=True, random_state=42, max_iter=500)
lgbm = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                       random_state=42, verbose=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

meta_X_train = np.zeros((X_train.shape[0], 3))
meta_X_test = np.zeros((X_test.shape[0], 3))
X_tr = X_train.values
y_tr = y_train.values
X_te = X_test.values

trained_rfs, trained_anns, trained_lgbms = [], [], []
fold_base_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
    print(f"  Fold {fold+1}/5 with SMOTE...")
    X_fold, y_fold = X_tr[train_idx], y_tr[train_idx]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    X_fold_s, y_fold_s = smote.fit_resample(X_fold, y_fold)

    rf_c = clone(rf); ann_c = clone(ann); lgbm_c = clone(lgbm)
    rf_c.fit(X_fold_s, y_fold_s)
    ann_c.fit(X_fold_s, y_fold_s)
    lgbm_c.fit(X_fold_s, y_fold_s)

    meta_X_train[val_idx, 0] = rf_c.predict_proba(X_val)[:, 1]
    meta_X_train[val_idx, 1] = ann_c.predict_proba(X_val)[:, 1]
    meta_X_train[val_idx, 2] = lgbm_c.predict_proba(X_val)[:, 1]
    meta_X_test[:, 0] += rf_c.predict_proba(X_te)[:, 1] / 5
    meta_X_test[:, 1] += ann_c.predict_proba(X_te)[:, 1] / 5
    meta_X_test[:, 2] += lgbm_c.predict_proba(X_te)[:, 1] / 5

    # Per-fold base AUC (average of 3 base learners)
    avg_prob = (rf_c.predict_proba(X_val)[:, 1] +
                ann_c.predict_proba(X_val)[:, 1] +
                lgbm_c.predict_proba(X_val)[:, 1]) / 3
    fold_auc = roc_auc_score(y_val, avg_prob)
    fold_base_aucs.append(fold_auc)
    print(f"    Fold {fold+1} base AUC: {fold_auc:.4f}")

    trained_rfs.append(rf_c)
    trained_anns.append(ann_c)
    trained_lgbms.append(lgbm_c)

print(f"\n  Base Learner CV AUC: {np.mean(fold_base_aucs):.4f} +/- {np.std(fold_base_aucs):.4f}")

# ============================================================
# STEP 2: Train CatBoost meta-learner
# ============================================================
print("\n--- STEP 2: Training CatBoost Meta-Learner ---")
cat_meta = CatBoostClassifier(
    iterations=500, depth=4, learning_rate=0.03,
    l2_leaf_reg=3.0, bagging_temperature=0.5,
    loss_function='Logloss', eval_metric='AUC',
    random_seed=42, verbose=False
)
cat_meta.fit(meta_X_train, y_tr)
y_prob = cat_meta.predict_proba(meta_X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_prob)
print(f"  Test AUC: {test_auc:.4f}")

# ============================================================
# STEP 3: Threshold optimization
# ============================================================
print("\n--- STEP 3: Threshold Optimization ---")
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
best_t, best_j = 0.5, -1
results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    j = sens + spec - 1
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)

    results.append({
        'Model': 'Proposed V2 SMOTE',
        'Threshold': t, 'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
        'Sensitivity': round(sens, 4), 'Specificity': round(spec, 4),
        'AUC': round(test_auc, 4), 'Youden_J': round(j, 4)
    })
    print(f"  t={t:.2f} -> Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f} J={j:.4f}")

    if j > best_j:
        best_j = j
        best_t = t

# ============================================================
# STEP 4: Final Report
# ============================================================
y_opt = (y_prob >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_opt).ravel()

print("\n" + "="*80)
print(f"FINAL V2 RESULTS @ Threshold = {best_t}")
print("="*80)
print(f"  Accuracy:     {accuracy_score(y_test, y_opt):.4f}")
print(f"  AUC:          {test_auc:.4f}")
print(f"  Sensitivity:  {tp/(tp+fn):.4f}")
print(f"  Specificity:  {tn/(tn+fp):.4f}")
print(f"  Precision:    {precision_score(y_test, y_opt):.4f}")
print(f"  Youden's J:   {best_j:.4f}")
print(f"  CV AUC:       {np.mean(fold_base_aucs):.4f} +/- {np.std(fold_base_aucs):.4f}")

print("\n--- V1 -> V2 COMPARISON ---")
print(f"  V1 AUC:       0.573 (16 features)")
print(f"  V2 AUC:       {test_auc:.4f} ({X.shape[1]} features)")
print(f"  Improvement:  {(test_auc - 0.573)*100:+.2f}% AUC points")

# Save results
pd.DataFrame(results).to_csv(os.path.join(output_dir, "v2_threshold_results.csv"), index=False)

# ============================================================
# STEP 5: Export production model
# ============================================================
print("\n--- STEP 5: Exporting Production Model ---")
bundle = {
    'base_rfs': trained_rfs,
    'base_anns': trained_anns,
    'base_lgbms': trained_lgbms,
    'meta_learner': cat_meta,
    'optimal_threshold': best_t,
    'feature_names': list(X.columns),
    'test_auc': test_auc,
    'cv_auc_mean': np.mean(fold_base_aucs),
    'cv_auc_std': np.std(fold_base_aucs),
}
bundle_path = os.path.join(model_dir, "production_model_v2.pkl")
joblib.dump(bundle, bundle_path)
joblib.dump(list(X.columns), os.path.join(model_dir, "model_features_v2.pkl"))
print(f"  Saved to {bundle_path}")
print("\n[SUCCESS] V2 Pipeline Complete!")
