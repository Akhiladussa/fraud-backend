
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

# STEP 1: Load dataset
print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# STEP 2: Create training, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

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

# STEP 3: Scale Logistic Regression features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# STEP 4: Train weighted Logistic Regression
print("\nTraining weighted Logistic Regression...")

lr_model = LogisticRegression(
    solver="saga",
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

# STEP 5: Train weighted Random Forest
print("\nTraining weighted Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# STEP 6: Get validation probabilities
lr_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
rf_prob = rf_model.predict_proba(X_val)[:, 1]

print("\n========== VALIDATION RESULTS ==========")

print("\nLogistic Regression Average Precision:",
      average_precision_score(y_val, lr_prob))

print("Random Forest Average Precision:",
      average_precision_score(y_val, rf_prob))

# STEP 7: Compare thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for name, probabilities in [
    ("Weighted Logistic Regression", lr_prob),
    ("Weighted Random Forest", rf_prob)
]:

    print(f"\n========== {name} ==========")

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

print("\nClass weighting experiment completed.")
print("Final test data was not evaluated.")
