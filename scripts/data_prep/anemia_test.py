import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
data = pd.read_excel("Anemia Dataset.xlsx")

# ----------------------------
# 2️⃣ Remove Hb (avoid leakage)
# ----------------------------
X = data.drop(columns=["Decision_Class", "Hb"])
y = data["Decision_Class"]

# Convert Gender to numeric
X["Gender"] = X["Gender"].map({"m": 1, "f": 0})

# ----------------------------
# 3️⃣ Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

results = []

# ----------------------------
# Function to Evaluate Model
# ----------------------------
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    results.append([name, acc, auc, sensitivity, specificity])

# ----------------------------
# 4️⃣ Train Individual Models
# ----------------------------

evaluate_model("Logistic Regression",
               LogisticRegression(max_iter=1000))

evaluate_model("Random Forest",
               RandomForestClassifier(n_estimators=500,
                                       max_depth=10,
                                       random_state=42))

evaluate_model("XGBoost",
               XGBClassifier(n_estimators=500,
                             max_depth=4,
                             learning_rate=0.01,
                             random_state=42,
                             use_label_encoder=False,
                             eval_metric="logloss"))

evaluate_model("ANN (MLP)",
               MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                             max_iter=500,
                             random_state=42))

# ----------------------------
# 5️⃣ Hybrid Model (Stacking)
# ----------------------------

# Base models
rf = RandomForestClassifier(n_estimators=500,
                            max_depth=10,
                            random_state=42)

ann = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                    max_iter=500,
                    random_state=42)

rf.fit(X_train, y_train)
ann.fit(X_train, y_train)

rf_train_prob = rf.predict_proba(X_train)[:, 1]
ann_train_prob = ann.predict_proba(X_train)[:, 1]

rf_test_prob = rf.predict_proba(X_test)[:, 1]
ann_test_prob = ann.predict_proba(X_test)[:, 1]

X_train_stack = np.column_stack((rf_train_prob, ann_train_prob))
X_test_stack = np.column_stack((rf_test_prob, ann_test_prob))

meta = XGBClassifier(n_estimators=500,
                     max_depth=4,
                     learning_rate=0.01,
                     random_state=42,
                     use_label_encoder=False,
                     eval_metric="logloss")

meta.fit(X_train_stack, y_train)

y_pred = meta.predict(X_test_stack)
y_prob = meta.predict_proba(X_test_stack)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

results.append(["Hybrid (RF+ANN→XGB)", acc, auc, sensitivity, specificity])

# ----------------------------
# 6️⃣ Print Comparison Table
# ----------------------------

results_df = pd.DataFrame(results,
                          columns=["Model",
                                   "Accuracy",
                                   "AUC",
                                   "Sensitivity",
                                   "Specificity"])

print("\n=== FULL MODEL COMPARISON (Without Hb) ===\n")
print(results_df.sort_values(by="Accuracy", ascending=False))