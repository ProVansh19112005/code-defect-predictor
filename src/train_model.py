import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("data/final_dataset.csv")

X = df.drop("is_bug_fix", axis=1)
y = df["is_bug_fix"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Handle Class Imbalance
# -------------------------------
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# -------------------------------
# Train XGBoost Model
# -------------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# Evaluation
# -------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc)

# -------------------------------
# Feature Importance
# -------------------------------
feature_names = X.columns
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

print("\nFeature Importances:\n")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

plt.figure()
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# SHAP Explainability
# -------------------------------
print("\nGenerating SHAP explanations...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    matplotlib=True
)

# -------------------------------
# Save Experiment Results
# -------------------------------
report = classification_report(y_test, y_pred, output_dict=True)

results = {
    "roc_auc": roc,
    "precision_bug": report["1"]["precision"],
    "recall_bug": report["1"]["recall"],
    "f1_bug": report["1"]["f1-score"]
}

results_df = pd.DataFrame([results])
results_df.to_csv("data/experiment_results.csv", index=False)

print("\nExperiment results saved to data/experiment_results.csv")