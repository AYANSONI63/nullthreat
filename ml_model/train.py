import pandas as pd
import json

from pathlib import Path

from ml_model.training.trainer import train_xgboost
from ml_model.training.evaluator import evaluate_model


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Load Dataset
# ============================================================

X_train = pd.read_csv(
    ARTIFACTS_DIR / "X_train_processed.csv"
)

X_val = pd.read_csv(
    ARTIFACTS_DIR / "X_val_processed.csv"
)

X_test = pd.read_csv(
    ARTIFACTS_DIR / "X_test_processed.csv"
)


y_train = pd.read_csv(
    ARTIFACTS_DIR / "y_train.csv"
).squeeze("columns")

y_val = pd.read_csv(
    ARTIFACTS_DIR / "y_val.csv"
).squeeze("columns")

y_test = pd.read_csv(
    ARTIFACTS_DIR / "y_test.csv"
).squeeze("columns")


# ============================================================
# Dataset Information
# ============================================================

print("=" * 60)
print("Loading Processed Dataset...")
print("=" * 60)

print("\nTraining Dataset")
print(X_train.shape)
print(y_train.shape)

print("\nValidation Dataset")
print(X_val.shape)
print(y_val.shape)

print("\nTesting Dataset")
print(X_test.shape)
print(y_test.shape)


# ============================================================
# MODEL 1
# 14-Feature Benchmark Model
# Includes URLSimilarityIndex
# ============================================================

print("\n")
print("=" * 60)
print("MODEL 1 — 14 FEATURE BENCHMARK")
print("=" * 60)

print("\nFeatures:")
print(list(X_train.columns))

print("\n# Training XGBoost Model...")

model_14 = train_xgboost(
    X_train,
    y_train
)

print("\n# Training completed successfully.")


# ------------------------------------------------------------
# Training Metrics
# ------------------------------------------------------------

train_metrics_14 = evaluate_model(
    model_14,
    X_train,
    y_train
)

print("\n" + "=" * 60)
print("MODEL 1 — Training Metrics")
print("=" * 60)

for metric, value in train_metrics_14.items():
    print(f"{metric}: {value}")


# ------------------------------------------------------------
# Validation Metrics
# ------------------------------------------------------------

val_metrics_14 = evaluate_model(
    model_14,
    X_val,
    y_val
)

print("\n" + "=" * 60)
print("MODEL 1 — Validation Metrics")
print("=" * 60)

for metric, value in val_metrics_14.items():
    print(f"{metric}: {value}")


# ------------------------------------------------------------
# Save Model 1
# ------------------------------------------------------------

model_14_path = (
    MODEL_DIR /
    "xgboost_baseline_14_features.json"
)

model_14.save_model(
    model_14_path
)


# ------------------------------------------------------------
# Save Model 1 Metrics
# ------------------------------------------------------------

metrics_14 = {
    "model": "xgboost_baseline_14_features",
    "feature_count": len(X_train.columns),
    "features": list(X_train.columns),
    "train": train_metrics_14,
    "validation": val_metrics_14
}

metrics_14_path = (
    ARTIFACTS_DIR /
    "xgboost_baseline_14_features_metrics.json"
)

with open(
    metrics_14_path,
    "w"
) as f:
    json.dump(
        metrics_14,
        f,
        indent=4
    )

print("\nModel 1 saved to:")
print(model_14_path)

print("\nModel 1 metrics saved to:")
print(metrics_14_path)


# ============================================================
# MODEL 2
# 13-Feature Production Candidate
# Removes URLSimilarityIndex
# ============================================================

print("\n")
print("=" * 60)
print("MODEL 2 — 13 FEATURE PRODUCTION CANDIDATE")
print("=" * 60)


# Remove URLSimilarityIndex
X_train_13 = X_train.drop(
    columns=["URLSimilarityIndex"]
)

X_val_13 = X_val.drop(
    columns=["URLSimilarityIndex"]
)

X_test_13 = X_test.drop(
    columns=["URLSimilarityIndex"]
)


print("\nFeatures:")
print(list(X_train_13.columns))

print("\nTraining Dataset")
print(X_train_13.shape)

print("\nValidation Dataset")
print(X_val_13.shape)

print("\nTesting Dataset")
print(X_test_13.shape)


print("\n# Training XGBoost Model...")


model_13 = train_xgboost(
    X_train_13,
    y_train
)

print("\n# Training completed successfully.")


# ------------------------------------------------------------
# Training Metrics
# ------------------------------------------------------------

train_metrics_13 = evaluate_model(
    model_13,
    X_train_13,
    y_train
)

print("\n" + "=" * 60)
print("MODEL 2 — Training Metrics")
print("=" * 60)

for metric, value in train_metrics_13.items():
    print(f"{metric}: {value}")


# ------------------------------------------------------------
# Validation Metrics
# ------------------------------------------------------------

val_metrics_13 = evaluate_model(
    model_13,
    X_val_13,
    y_val
)

print("\n" + "=" * 60)
print("MODEL 2 — Validation Metrics")
print("=" * 60)

for metric, value in val_metrics_13.items():
    print(f"{metric}: {value}")


# ------------------------------------------------------------
# Save Model 2
# ------------------------------------------------------------

model_13_path = (
    ARTIFACTS_DIR / "production" /
    "xgboost_baseline_13_features.json"
)

model_13.save_model(
    model_13_path
)


# ------------------------------------------------------------
# Save Model 2 Metrics
# ------------------------------------------------------------

metrics_13 = {
    "model": "xgboost_baseline_13_features",
    "removed_features": [
        "URLSimilarityIndex"
    ],
    "feature_count": len(X_train_13.columns),
    "features": list(X_train_13.columns),
    "train": train_metrics_13,
    "validation": val_metrics_13
}

metrics_13_path = (
    ARTIFACTS_DIR /
    "xgboost_baseline_13_features_metrics.json"
)

with open(
    metrics_13_path,
    "w"
) as f:
    json.dump(
        metrics_13,
        f,
        indent=4
    )


print("\nModel 2 saved to:")
print(model_13_path)

print("\nModel 2 metrics saved to:")
print(metrics_13_path)


# ============================================================
# Final Summary
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE TRAINING COMPLETE")
print("=" * 60)

print("\nModel 1:")
print("14 features + URLSimilarityIndex")
print(model_14_path)

print("\nModel 2:")
print("13 features - URLSimilarityIndex")
print(model_13_path)

print("\nMetrics saved for both models.")

print("\nTest set has NOT been evaluated.")
print("It remains reserved for the next evaluation phase.")

print("=" * 60)