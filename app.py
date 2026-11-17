
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

from shap_explainer import explain_transaction

app = Flask(__name__)

CORS(app)

# Load the final model bundle once when the API starts
model_bundle = joblib.load("fraud_model_bundle.pkl")

model = model_bundle["model"]
THRESHOLD = model_bundle["threshold"]
FEATURE_NAMES = model_bundle["features"]


@app.route("/")
def home():
    return jsonify({
        "message": "Fraud Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON request"}), 400

        # Check for missing features
        missing_features = [
            name for name in FEATURE_NAMES if name not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features",
                "missing": missing_features
            }), 400

        # Convert input values into numeric values
        try:
            features = [float(data[name]) for name in FEATURE_NAMES]
        except (ValueError, TypeError):
            return jsonify({
                "error": "All feature values must be numeric"
            }), 400

        input_array = np.array(features).reshape(1, -1)

        # Reject NaN and infinite values
        if not np.isfinite(input_array).all():
            return jsonify({
                "error": "Feature values must be finite numbers"
            }), 400

        # Get prediction probabilities
        probabilities = model.predict_proba(input_array)[0]

        fraud_index = list(model.classes_).index(1)
        legit_index = list(model.classes_).index(0)

        fraud_probability = float(probabilities[fraud_index])
        legit_probability = float(probabilities[legit_index])

        # Apply the selected threshold
        prediction = int(fraud_probability >= THRESHOLD)

        # Prepare the transaction for SHAP explanation
        transaction_df = pd.DataFrame(
            [features],
            columns=FEATURE_NAMES
        )

        # Generate the top 5 SHAP feature contributions
        shap_result = explain_transaction(
            transaction_df,
            top_n=5
        )

        result = {
            "prediction": prediction,
            "label": (
                "🚨 FRAUDULENT"
                if prediction == 1
                else "✅ LEGITIMATE"
            ),
            "fraud_probability": round(fraud_probability * 100, 2),
            "legit_probability": round(legit_probability * 100, 2),
            "threshold": THRESHOLD,
            "explanation": {
                "base_value": shap_result["base_value"],
                "reconstructed_score": shap_result["reconstructed_score"],
                "top_features": shap_result["top_features"]
            }
        }

        return jsonify(result)

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "An internal error occurred during prediction"
        }), 500


@app.route("/sample", methods=["GET"])
def sample_transaction():
    """Returns a synthetic transaction for testing the UI."""
    import random

    sample = {
        "Time": random.uniform(0, 172792),
        "Amount": round(random.uniform(1, 2500), 2)
    }

    for i in range(1, 29):
        sample[f"V{i}"] = round(random.uniform(-3, 3), 6)

    return jsonify(sample)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
