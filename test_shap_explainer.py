
import pandas as pd
import numpy as np

from shap_explainer import explain_transaction

# Load two known transactions
df = pd.read_csv("creditcard.csv", nrows=542)

fraud_transaction = df.loc[[541]].copy()
legitimate_transaction = df.loc[[0]].copy()

def run_test(name, transaction):
    try:
        explain_transaction(transaction)
        print(f"❌ FAILED: {name} — invalid input was accepted")
    except (ValueError, TypeError, KeyError) as error:
        print(f"✅ PASSED: {name} — {error}")

print("\n========== SHAP INPUT VALIDATION ==========\n")

# 1. Missing feature
missing = legitimate_transaction.drop(columns=["V1"])
run_test("Missing feature", missing)

# 2. Non-numeric feature
non_numeric = legitimate_transaction.copy()
non_numeric["V1"] = non_numeric["V1"].astype(object)
non_numeric.loc[non_numeric.index[0], "V1"] = "invalid"
run_test("Non-numeric value", non_numeric)

# 3. NaN value
nan_input = legitimate_transaction.copy()
nan_input.loc[nan_input.index[0], "V1"] = np.nan
run_test("NaN value", nan_input)

# 4. Infinite value
infinite_input = legitimate_transaction.copy()
infinite_input.loc[infinite_input.index[0], "V1"] = np.inf
run_test("Infinite value", infinite_input)

# 5. Multiple transactions
multiple = pd.concat(
    [fraud_transaction, legitimate_transaction]
)
run_test("Multiple transactions", multiple)

print("\n========== VALID INPUT CHECK ==========\n")

result = explain_transaction(fraud_transaction)

if result["prediction"] == 1:
    print("✅ PASSED: Valid fraud transaction accepted")
else:
    print("❌ FAILED: Unexpected prediction")

print("\nValidation tests completed.")
