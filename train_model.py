import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading dataset...")
credit_card_data = pd.read_csv('creditcard.csv')

# Separate legit and fraud
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(f"Legit transactions: {len(legit)}")
print(f"Fraud transactions: {len(fraud)}")

# Under-sampling to balance classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Split features and target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
print("Training model...")
model = LogisticRegression(solver='saga', max_iter=5000)
model.fit(X_train_scaled, Y_train)

# Evaluate
train_acc = accuracy_score(Y_train, model.predict(X_train_scaled))
test_acc  = accuracy_score(Y_test,  model.predict(X_test_scaled))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, model.predict(X_test_scaled)))

# Save model and scaler
joblib.dump(model,  'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ model.pkl and scaler.pkl saved successfully!")