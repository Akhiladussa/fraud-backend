from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import traceback

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# Load the saved model and scaler once at startup
model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# The 30 features in the exact order the model expects
FEATURE_NAMES = (
    ['Time'] +
    [f'V{i}' for i in range(1, 29)] +
    ['Amount']
)

@app.route('/')
def home():
    return jsonify({"message": "Fraud Detection API is running ✅"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build feature array in correct order
        features = []
        for name in FEATURE_NAMES:
            value = data.get(name)
            if value is None:
                return jsonify({"error": f"Missing field: {name}"}), 400
            features.append(float(value))

        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        result = {
            "prediction": int(prediction),
            "label": "🚨 FRAUDULENT" if prediction == 1 else "✅ LEGITIMATE",
            "fraud_probability":    round(float(probability[1]) * 100, 2),
            "legit_probability":    round(float(probability[0]) * 100, 2)
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/sample', methods=['GET'])
def sample_transaction():
    """Returns a sample transaction for quick testing in the UI."""
    import random
    # Realistic-looking PCA-transformed values
    sample = {
        "Time": random.uniform(0, 172792),
        "Amount": round(random.uniform(1, 2500), 2),
    }
    for i in range(1, 29):
        sample[f"V{i}"] = round(random.uniform(-3, 3), 6)
    return jsonify(sample)


if __name__ == '__main__':
    app.run(debug=True, port=5000)