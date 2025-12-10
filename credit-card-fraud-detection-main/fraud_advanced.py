import matplotlib
matplotlib.use("TkAgg")  # Force popup windows

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.utils import resample
import joblib


print("Loading dataset...")
data = pd.read_csv("creditcard.csv")
print("Dataset loaded! Shape:", data.shape)

# Split X and y
X = data.drop('Class', axis=1)
y = data['Class']

# Scale features
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nOriginal training class distribution:")
print(y_train.value_counts())

# ==========================
#  FAST UNDERSAMPLING METHOD
# ==========================

train_data = pd.concat([X_train, y_train], axis=1)

non_fraud = train_data[train_data.Class == 0]
fraud = train_data[train_data.Class == 1]

non_fraud_sampled = resample(
    non_fraud,
    replace=False,
    n_samples=len(fraud),
    random_state=42
)

balanced_data = pd.concat([non_fraud_sampled, fraud])

X_train_bal = balanced_data.drop("Class", axis=1)
y_train_bal = balanced_data["Class"]

print("\nBalanced training class distribution:")
print(y_train_bal.value_counts())

# ==========================
#  TRAIN RANDOM FOREST MODEL
# ==========================

print("\nTraining RandomForest model...")
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_bal, y_train_bal)
print("Model training complete!")

# ==========================
#  PREDICTION & EVALUATION
# ==========================

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print("\nAUC Score:", auc)

# ==========================
#  ROC CURVE
# ==========================

print("\nPlotting ROC Curve...")
plt.figure()
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve")
plt.show()

# ==========================
#  PRECISION–RECALL CURVE
# ==========================

print("\nPlotting Precision–Recall Curve...")
plt.figure()
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision–Recall Curve")
plt.show()

# ==========================
#  SHAP EXPLAINABILITY
# ==========================

print("\nCalculating SHAP values (may take some time)...")

X_explain = X_test.sample(50, random_state=42)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_explain)

print("Generating SHAP summary plot...")
plt.figure()

# Handle multi-output and single-output
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_explain, plot_type="dot", show=False)
else:
    shap.summary_plot(shap_values, X_explain, plot_type="dot", show=False)

# Save shap to file
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
print("SHAP plot saved as shap_summary.png")

plt.show()

# ==========================
#  SAVE MODEL
# ==========================

joblib.dump(rf, "fraud_model.pkl")
print("\nModel saved as fraud_model.pkl")
