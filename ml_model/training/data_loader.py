from pathlib import Path
import pandas as pd


def load_processed_dataset(artifacts_dir: Path):
    """
    Load processed train, validation and test datasets.
    """

    X_train = pd.read_csv(artifacts_dir / "X_train_processed.csv")
    X_val = pd.read_csv(artifacts_dir / "X_val_processed.csv")
    X_test = pd.read_csv(artifacts_dir / "X_test_processed.csv")

    y_train = pd.read_csv(
        artifacts_dir / "y_train.csv"
    ).squeeze("columns")

    y_val = pd.read_csv(
        artifacts_dir / "y_val.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        artifacts_dir / "y_test.csv"
    ).squeeze("columns")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )