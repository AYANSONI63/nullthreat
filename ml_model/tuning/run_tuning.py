import pandas as pd 

from pathlib import Path 
from ml_model.tuning.tuner import run_tuning



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


study = run_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=50
)


print("=" * 60)
print("BEST TRIAL")
print("=" * 60)

print(f"Best F1 : {study.best_value:.6f}")
print("\nBest Parameters:")


for parameter, value in study.best_params.items():
    print(f"{parameter}: {value}")