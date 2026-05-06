import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, precision_score
import warnings
warnings.filterwarnings('ignore')

dataset_path = "odisha_ready_for_ml.csv"
print("Loading preprocessed dataset...")
df = pd.read_csv(dataset_path)

# Separate predictors (X) and target (y)
X = df.drop(columns=['Anaemia_Target'])
y = df['Anaemia_Target']

print(f"Features: {X.shape[1]}")
print(f"Target distribution: \n{y.value_counts()}")

# Train-Test Split (70:30) as per Reference Study
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
print(f"\nTraining set: {X_train.shape[0]} rows. Test set: {X_test.shape[0]} rows.")

# Dictionary to hold models and their hyperparameters 
# Note: RF and ANN parameters are matched to the Tanzania reference paper where possible
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    
    # 500 trees, max_depth 10, balanced class weight (Reference paper 5.2.1)
    'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced', random_state=42),
    
    # 3 Hidden layers: 128, 64, 32. ReLU activation. Adam optimizer. Early stopping. (Reference paper 5.2.2)
    'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', early_stopping=True, random_state=42),
    
    # Basic XGBoost for baseline comparison
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []

print("\n--- Training Baseline Models ---")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability for AUC
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    sensitivity = recall_score(y_test, y_pred) # Also known as Recall
    precision = precision_score(y_test, y_pred)
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Sensitivity (Recall)': round(sensitivity, 4),
        'Specificity': round(specificity, 4),
        'Precision': round(precision, 4)
    })

# Display Results
results_df = pd.DataFrame(results)
print("\n--- Baseline Model Performance Comparison ---")
print(results_df.to_string(index=False))

results_df.to_csv("baseline_results.csv", index=False)
print("\nSaved baseline performance to baseline_results.csv")
