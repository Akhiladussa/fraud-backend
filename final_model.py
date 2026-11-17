
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Step 1: Split into training and temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

# Step 2: Split temporary data into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print("\nCombining training and validation data...")

# Use both training and validation data for final training
X_final_train = pd.concat([X_train, X_val])
y_final_train = pd.concat([y_train, y_val])

print("Final training distribution:")
print(y_final_train.value_counts())

print("\nTraining final Weighted Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_final_train, y_final_train)

print("\nModel training completed!")

# Step 3: Save model and threshold together
threshold = 0.4

model_bundle = {
    "model": model,
    "threshold": threshold,
    "features": list(X.columns)
}

joblib.dump(model_bundle, "fraud_model_bundle.pkl")

print("\nModel saved as fraud_model_bundle.pkl")

# Step 4: Evaluate on the held-out test data
print("\n========== FINAL TEST EVALUATION ==========")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)

print("Threshold:", threshold)

print("Accuracy:",
      accuracy_score(y_test, predictions))

print("Precision:",
      precision_score(y_test, predictions, zero_division=0))

print("Recall:",
      recall_score(y_test, predictions, zero_division=0))

print("F1 Score:",
      f1_score(y_test, predictions, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(y_test, probabilities))

print("Average Precision:",
      average_precision_score(y_test, probabilities))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nFinal model implementation completed!")
