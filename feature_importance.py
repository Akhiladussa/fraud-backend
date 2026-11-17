
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Split the dataset
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

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nCalculating Permutation Importance...")

result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance": result.importances_mean
})

permutation_df = permutation_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Permutation Important Features:")
print(permutation_df.head(10))


# Extract feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Visualize feature importance

# Compare both feature importance methods

built_in_top = importance.head(10).sort_values(
    by="Importance",
    ascending=True
)

permutation_top = permutation_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Built-in importance
axes[0].barh(
    built_in_top["Feature"],
    built_in_top["Importance"]
)

axes[0].set_title("Random Forest Feature Importance")
axes[0].set_xlabel("Importance")

# Permutation importance
axes[1].barh(
    permutation_top["Feature"],
    permutation_top["Importance"]
)

axes[1].set_title("Permutation Feature Importance")
axes[1].set_xlabel("Decrease in Average Precision")

plt.tight_layout()
plt.show()
