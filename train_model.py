
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

print("Loading dataset...")

# STEP 1: Load original dataset
data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)
print("\nOriginal class distribution:")
print(data["Class"].value_counts())

# STEP 2: Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# STEP 3: Split original dataset BEFORE undersampling
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining distribution before sampling:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())

# STEP 4: Undersample ONLY training data
train_data = X_train.copy()
train_data["Class"] = y_train

legit = train_data[train_data["Class"] == 0]
fraud = train_data[train_data["Class"] == 1]

legit_sample = legit.sample(
    n=len(fraud),
    random_state=42
)

balanced_train = pd.concat(
    [legit_sample, fraud]
).sample(frac=1, random_state=42)

X_train_balanced = balanced_train.drop("Class", axis=1)
y_train_balanced = balanced_train["Class"]

print("\nTraining distribution after sampling:")
print(y_train_balanced.value_counts())

# STEP 5: Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# STEP 6: Train Logistic Regression
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    solver="saga",
    max_iter=5000,
    random_state=42
)

model.fit(X_train_scaled, y_train_balanced)

# STEP 7: Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# STEP 8: Evaluation
print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC (Average Precision):",
      average_precision_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# STEP 9: Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")
