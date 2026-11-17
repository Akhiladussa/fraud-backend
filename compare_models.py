
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# STEP 1: Load dataset
print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# STEP 2: Split original dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# STEP 3: Undersample only training data
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

# STEP 4: Create Random Forest model
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# STEP 5: Train model
rf_model.fit(X_train_balanced, y_train_balanced)

# STEP 6: Make predictions
y_pred = rf_model.predict(X_test)

y_prob = rf_model.predict_proba(X_test)[:, 1]

# STEP 7: Evaluate model
print("\n========== RANDOM FOREST RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(
    y_test, y_pred, zero_division=0
))

print("Recall:", recall_score(
    y_test, y_pred, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, y_pred, zero_division=0
))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("PR-AUC:", average_precision_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# STEP 8: Save separately
joblib.dump(rf_model, "random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
