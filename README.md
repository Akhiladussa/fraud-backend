# FraudGuard – Credit Card Fraud Detection API

FraudGuard is a machine-learning-based credit card fraud detection prototype. It uses a Weighted Random Forest classifier to identify potentially fraudulent transactions and SHAP (SHapley Additive exPlanations) to explain the features influencing individual predictions.

The backend is built using Python and Flask and exposes REST API endpoints for transaction prediction, synthetic sample generation, and API health checking.

## Project Overview

Credit card fraud datasets are highly imbalanced: fraudulent transactions represent only a small fraction of all transactions. FraudGuard addresses this challenge using class-weighted model training and a configurable decision threshold.

The application is intended as a decision-support prototype for fraud-risk analysts and fraud-operations teams. It can help prioritize transactions for further investigation.

**Key features:**
- Fraud prediction using a trained Random Forest model.
- Class-imbalance handling using `balanced_subsample`.
- Configurable fraud classification threshold.
- Fraud and legitimate transaction scores.
- Individual transaction explanations using SHAP.
- Top five influential features for each prediction.
- Input validation and error handling through Flask REST APIs.
- Synthetic sample transaction generation for frontend testing.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Backend framework | Flask |
| Machine learning | Scikit-learn |
| Model | Random Forest Classifier |
| Explainable AI | SHAP |
| Data processing | Pandas, NumPy |
| Model serialization | Joblib |
| API testing | Postman |
| Deployment | Render |

## Machine Learning Approach

### Dataset

The project uses the Credit Card Fraud Detection dataset (`creditcard.csv`).

Dataset characteristics:

- Total transactions: 284,807
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Total input features: 30
- Target column: `Class`

The dataset contains:
- `Time`: Time elapsed between transactions in seconds.
- `Amount`: Transaction amount.
- `V1` to `V28`: Anonymized numerical features obtained through PCA transformation.
- `Class`: Target variable (`0` = Legitimate, `1` = Fraudulent).

The `V1`–`V28` features are anonymized. Their individual real-world meanings are not available.

### Data Splitting

The dataset is divided using stratified sampling:

- 60% training data
- 20% validation data
- 20% test data

The training and validation datasets are combined to train the final model. The remaining test dataset is used for evaluation.

### Model Configuration

The final model is a Weighted Random Forest Classifier.

```python
RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

Class weighting helps the model account for the severe imbalance between legitimate and fraudulent transactions.

### Decision Threshold

The final model uses a fraud decision threshold of `0.40`.

```python
prediction = int(fraud_probability >= 0.40)
```

A transaction is classified as fraudulent when its model fraud score is greater than or equal to the threshold.

The threshold is configurable through the saved model bundle. It is a prototype operating choice and would require further validation before real banking use.

## Model Evaluation

The recorded evaluation results at the selected threshold are:

| Metric | Result |
|---|---:|
| Accuracy | 99.9614% |
| Precision | 96.3855% |
| Recall | 80.8081% |
| F1-score | 87.9121% |
| ROC-AUC | 95.2898% |
| Average Precision | 87.8062% |

Confusion matrix:

```text
                 Predicted
                 Legit  Fraud
Actual Legit     56860     3
Actual Fraud        19    80
```

**Evaluation note:** The held-out test set was also used during earlier model comparisons. Therefore, these results should be treated as project evaluation results rather than a completely untouched independent benchmark.

## Explainable AI with SHAP

FraudGuard uses `shap.TreeExplainer` to explain individual Random Forest predictions.

For each transaction, the SHAP module:

1. Validates the transaction input.
2. Arranges features in the model's expected order.
3. Calculates SHAP contributions for the fraud class.
4. Ranks features by the absolute magnitude of their contributions.
5. Returns the top five influential features.

Each feature explanation includes:
- Feature name
- Feature value
- SHAP contribution
- Direction of contribution

A positive contribution indicates that the feature increases the model's fraud score, while a negative contribution indicates that it decreases the score.

SHAP explanations describe model behavior. They do not establish causation or prove that a transaction is fraudulent.

## Backend API Endpoints

The Flask backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks API availability |
| POST | `/predict` | Predicts whether a transaction is fraudulent and returns SHAP explanations |
| GET | `/sample` | Generates synthetic transaction features for frontend testing |

### 1. API Health Check

**Request:**

```http
GET /
```

**Response:**

```json
{
  "message": "Fraud Detection API is running"
}
```

### 2. Predict Transaction

**Request:**

```http
POST /predict
Content-Type: application/json
```

The API expects all 30 input features: `Time`, `V1` through `V28`, and `Amount`.

Example using Python:

```python
import requests

payload = {
    "Time": 0,
    "Amount": 100,
    **{f"V{i}": 0 for i in range(1, 29)}
}

response = requests.post(
    "http://localhost:5000/predict",
    json=payload
)

print(response.json())
```

The example values are synthetic and are intended only to demonstrate the API request format.

**Successful response structure:**

```json
{
  "prediction": 0,
  "label": "LEGITIMATE",
  "fraud_probability": 3.99,
  "legit_probability": 96.01,
  "threshold": 0.4,
  "explanation": {
    "base_value": 0.0,
    "reconstructed_score": 0.0,
    "top_features": [
      {
        "feature": "V14",
        "value": -1.2,
        "contribution": -0.02,
        "direction": "Decreases fraud score"
      }
    ]
  }
}
```

The numerical values above illustrate the response structure; actual predictions and SHAP values depend on the submitted transaction.

**Input validation includes:**
- Invalid JSON requests.
- Missing required features.
- Non-numeric feature values.
- NaN and infinite values.
- Unexpected internal prediction errors.

### 3. Generate Synthetic Sample

**Request:**

```http
GET /sample
```

This endpoint returns randomly generated values for all 30 features. It is intended for frontend testing and demonstration.

**Important:** The generated sample is not a verified transaction from the original dataset and does not represent a known legitimate or fraudulent example.

## Project Structure

```text
backend/
│
├── app.py
├── final_model.py
├── train_model.py
├── fraud_model_bundle.pkl
├── shap_explainer.py
│
├── compare_models.py
├── class_weight_experiment.py
├── feature_importance.py
├── threshold_analysis.py
├── threshold_validation_final.py
├── shap_experiment.py
├── test_shap_explainer.py
│
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

The dataset and generated test files are maintained separately and are not required to be committed to the public repository.

## Installation and Local Setup

### Prerequisites

- Python 3.11
- pip
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/Akhiladussa/fraud-backend.git
cd fraud-backend
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask application

Ensure `fraud_model_bundle.pkl` is present in the backend directory.

```bash
python app.py
```

The API will be available at:

```text
http://localhost:5000
```

## Dependencies

The backend uses the following pinned dependencies:

```text
flask==3.0.3
flask-cors==4.0.1
scikit-learn==1.9.1
pandas==2.2.2
numpy==2.4.6
joblib==1.4.2
shap==0.51.0
scipy==1.17.1
gunicorn==21.2.0
```

## Deployment

The backend is deployed on Render.

**Live API:**

https://fraud-backend-app.onrender.com

The frontend communicates with this backend through its REST API.

The deployment uses Gunicorn as the production application server.

## Limitations and Future Scope

FraudGuard is an educational and demonstration prototype, not a production-ready banking fraud detection system.

Current limitations:

- Uses a publicly available historical dataset rather than live banking transactions.
- Does not connect to actual bank payment gateways or customer accounts.
- Does not automatically block or decline transactions.
- Uses anonymized PCA features whose original meanings are unavailable.
- Model scores are not calibrated real-world fraud probabilities.
- The selected threshold requires further business validation.
- SHAP explanations describe model contributions, not causal reasons for fraud.
- Production use would require security hardening, monitoring, model drift detection, privacy controls, and independent validation.

Potential future improvements include:
- Integration with secure real-time transaction streams.
- Continuous model monitoring and retraining.
- Business-specific threshold optimization.
- Analyst feedback and investigation workflows.
- Authentication, authorization, and stronger API security.

## Author

**Akhila Dussa**

GitHub: https://github.com/Akhiladussa

LinkedIn: https://linkedin.com/in/akhiladussa

---

**FraudGuard – Explainable Credit Card Fraud Detection**