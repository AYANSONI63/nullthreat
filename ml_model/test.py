import pandas as pd 
import numpy as np 

from pathlib import Path

from ml_model.training.trainer import train_xgboost
from ml_model.training.evaluator import evaluate_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ==========================================
# Load datasets 
# ==========================================


X_train = pd.read_csv(
    ARTIFACTS_DIR / "X_train_processed_test.csv"
)

X_test = pd.read_csv(
    ARTIFACTS_DIR / "X_test_processed_test.csv"
)

y_train = pd.read_csv(
    ARTIFACTS_DIR / "y_train.csv"
).squeeze("columns")

y_test = pd.read_csv(
    ARTIFACTS_DIR / "y_test.csv"
).squeeze("columns")


print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

print("\nTraining Dataset")
print(X_train.shape)
print(y_train.shape)

print("\nTesting Dataset")
print(X_test.shape)
print(y_test.shape)



# ============================================================
# Train final candidate
# ============================================================

print("\n" + "=" * 60)
print("Training Final Baseline Model")
print("=" * 60)

model = train_xgboost(
    X_train,
    y_train
)

print("\nTraining completed successfully.")



# ============================================================
# Evaluate on test set
# ============================================================

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

test_metrics = evaluate_model(
    model,
    X_test,
    y_test
)


for metric, value in test_metrics.items():
    print(f"{metric}: {value}")






# ============================================================
# Threshold Analysis
# ============================================================

print("\n" + "=" * 60)
print("TEST THRESHOLD ANALYSIS")
print("=" * 60)

TEST_THRESHOLD = 0.34


test_probabilities = model.predict_proba(
    X_test
)[:, 1]


test_predictions = (
    test_probabilities >= TEST_THRESHOLD
).astype(int)



accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

print(
    f"Threshold: {TEST_THRESHOLD:2f} | "
    f"Accuracy: {accuracy:.6f} | "
    f"Precision: {precision:.6f} | "
    f"Recall: {recall:.6f} | "
    f"F1: {f1:.6f}"
)