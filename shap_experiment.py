
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load the trained model
bundle = joblib.load("fraud_model_bundle.pkl")
model = bundle["model"]
features = bundle["features"]

# 2. Load the original dataset
df = pd.read_csv("creditcard.csv")

# 3. Select the actual fraud transaction
transaction = df.loc[[541]]
X = transaction[features]

# 4. Generate SHAP explanation
explainer = shap.TreeExplainer(model)
explanation = explainer(X)

# 5. Inspect the explanation
print("Transaction index:", transaction.index[0])
print("Actual class:", int(transaction["Class"].iloc[0]))
print("Model classes:", model.classes_)

print("\nSHAP explanation type:")
print(type(explanation))

print("\nSHAP values shape:")
print(explanation.values.shape)

print("\nSHAP base values:")
print(explanation.base_values)

print("\nInput feature shape:")
print(X.shape)

print("\nModel probabilities:")
print(model.predict_proba(X))


import numpy as np

# 6. Extract SHAP values for fraud class (class 1)
fraud_shap_values = explanation.values[0, :, 1]

# 7. Get the base value for fraud class
fraud_base_value = float(explanation.base_values[0, 1])

# 8. Calculate the sum of SHAP contributions
explained_fraud_score = fraud_base_value + fraud_shap_values.sum()

# 9. Get actual fraud probability
actual_fraud_score = float(model.predict_proba(X)[0, 1])

print("\n========== FRAUD EXPLANATION ==========")

print("Base value:", fraud_base_value)
print("Sum of SHAP contributions:", fraud_shap_values.sum())
print("SHAP reconstructed score:", explained_fraud_score)
print("Actual model fraud score:", actual_fraud_score)

print("\n========== TOP 10 INFLUENTIAL FEATURES ==========")

# Sort features by absolute SHAP contribution
top_indices = np.argsort(np.abs(fraud_shap_values))[::-1][:10]

for index in top_indices:
    feature = features[index]
    contribution = fraud_shap_values[index]

    direction = "INCREASED fraud score" if contribution > 0 else "DECREASED fraud score"

    print(
        f"{feature:8s} | "
        f"Contribution: {contribution:+.6f} | "
        f"{direction}"
    )

# 10. Generate a waterfall plot for fraud class
fraud_explanation = explanation[0, :, 1]

shap.plots.waterfall(
    fraud_explanation,
    max_display=10,
    show=False
)

plt.title("Fraud Explanation - Transaction 541")
plt.tight_layout()

plt.savefig(
    "shap_waterfall_541.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nWaterfall plot saved successfully!")
