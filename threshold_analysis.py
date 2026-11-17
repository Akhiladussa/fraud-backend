
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

# 1. Load dataset
print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# 2. Split into training and temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

# 3. Split temporary data into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nValidation distribution:")
print(y_val.value_counts())

print("\nFinal test distribution:")
print(y_test.value_counts())

# 4. Undersample training data ONLY
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

X_balanced = balanced_train.drop("Class", axis=1)
y_balanced = balanced_train["Class"]

# 5. Train Logistic Regression
scaler = StandardScaler()

X_balanced_scaled = scaler.fit_transform(X_balanced)
X_val_scaled = scaler.transform(X_val)

lr_model = LogisticRegression(
    solver="saga",
    max_iter=5000,
    random_state=42
)

lr_model.fit(X_balanced_scaled, y_balanced)

# 6. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_balanced, y_balanced)

# 7. Get fraud probabilities
lr_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
rf_prob = rf_model.predict_proba(X_val)[:, 1]

print("\n========== VALIDATION AVERAGE PRECISION ==========")

print("Logistic Regression:",
      average_precision_score(y_val, lr_prob))

print("Random Forest:",
      average_precision_score(y_val, rf_prob))

# 8. Compare classification thresholds
thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

for model_name, probabilities in [
    ("Logistic Regression", lr_prob),
    ("Random Forest", rf_prob)
]:

    print(f"\n========== {model_name} ==========")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        print(f"\nThreshold: {threshold}")

        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        print("False Positives:", fp)
        print("False Negatives:", fn)

print("\nValidation threshold analysis completed.")
print("Final test set has NOT been used.")


# ==========================================
# STEP 9: FINAL TEST EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("\n========== FINAL TEST EVALUATION ==========")

# Selected thresholds based on validation results
selected_models = [
    ("Logistic Regression", lr_model, scaler, 0.30),
    ("Random Forest", rf_model, None, 0.40)
]

for name, model, model_scaler, threshold in selected_models:

    print(f"\n========== {name} ==========")

    # Transform test data only when needed
    if model_scaler is not None:
        X_final = model_scaler.transform(X_test)
    else:
        X_final = X_test

    # Get fraud probabilities
    probabilities = model.predict_proba(X_final)[:, 1]

    # Apply selected threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Calculate metrics
    print("Selected Threshold:", threshold)

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

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nTrue Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

print("\nFinal test evaluation completed!")
