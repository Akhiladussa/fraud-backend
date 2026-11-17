
import joblib
import shap
import numpy as np

# Load the trained model bundle once
bundle = joblib.load("fraud_model_bundle.pkl")

model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]

# Find the class indices using their actual labels
fraud_index = list(model.classes_).index(1)

# Create the SHAP explainer once
explainer = shap.TreeExplainer(model)


def explain_transaction(transaction, top_n=5):
    """
    Explain one transaction using SHAP.

    transaction: DataFrame containing the model features.
    top_n: Number of most influential features to return.
    """

    # 1. Validate input
    if len(transaction) != 1:
        raise ValueError("Exactly one transaction must be provided")

    missing_features = [
        feature for feature in FEATURES
        if feature not in transaction.columns
    ]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # 2. Arrange features in the trained model's order
    X = transaction[FEATURES]

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite numbers")

    # 3. Generate SHAP explanation
    explanation = explainer(X)

    # 4. Extract fraud-class contributions
    shap_values = explanation.values[0, :, fraud_index]

    base_value = float(
        explanation.base_values[0, fraud_index]
    )

    # 5. Get model fraud score
    probabilities = model.predict_proba(X)[0]
    fraud_probability = float(probabilities[fraud_index])

    # 6. Rank features by absolute contribution
    top_indices = np.argsort(
        np.abs(shap_values)
    )[::-1][:top_n]

    top_features = []

    for index in top_indices:
        contribution = float(shap_values[index])

        if contribution > 0:
            direction = "Increases fraud score"
        elif contribution < 0:
            direction = "Decreases fraud score"
        else:
            direction = "No effect"

        top_features.append({
            "feature": FEATURES[index],
            "value": float(X.iloc[0, index]),
            "contribution": contribution,
            "direction": direction
        })

    # 7. Return explanation
    return {
        "fraud_probability": fraud_probability,
        "threshold": THRESHOLD,
        "prediction": int(fraud_probability >= THRESHOLD),
        "base_value": base_value,
        "reconstructed_score": float(
            base_value + shap_values.sum()
        ),
        "top_features": top_features
    }
