import numpy as np
import pandas as pd 

from pathlib import Path
from ml_model.training.trainer import train_xgboost
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

X_train = pd.read_csv(
    ARTIFACTS_DIR / "X_train_processed_test.csv"
)

X_val = pd.read_csv(
    ARTIFACTS_DIR / "X_val_processed_test.csv"
)

y_train = pd.read_csv(
    ARTIFACTS_DIR / "y_train.csv"
).squeeze("columns")

y_val = pd.read_csv(
    ARTIFACTS_DIR / "y_val.csv"
).squeeze("columns")



# Train baseline model
model = train_xgboost(
    X_train,
    y_train
)


# Validation probabilities
val_probabilities = model.predict_proba(
    X_val
)[:, 1]


print("=" * 60)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 60)


thresholds = np.arange(0.10, 0.91, 0.01)

results = []


for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# Find best validation F1
best_result = max(
    results,
    key=lambda x: x["f1"]
)


print("\nBEST VALIDATION THRESHOLD")
print("=" * 60)

for key, value in best_result.items():
    print(f"{key}: {value:.6f}")