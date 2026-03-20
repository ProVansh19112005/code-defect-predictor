import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Load processed data
X, y = pickle.load(open("processed_data.pkl", "rb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model (best for text)
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc)

# Save model
pickle.dump(model, open("model_numeric.pkl", "wb"))

print("\nModel saved as model.pkl")