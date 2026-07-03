from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "xgb_model.pkl"
SELECTED_FEATURES_PATH = ARTIFACT_DIR / "selected_features.json"
OUTLIER_CAPS_PATH = ARTIFACT_DIR / "outlier_caps.json"
X_TRAIN_PATH = ARTIFACT_DIR / "X_train_selected.csv"
Y_TRAIN_PATH = ARTIFACT_DIR / "y_train.csv"
X_TEST_PATH = ARTIFACT_DIR / "X_test_selected.csv"
Y_TEST_PATH = ARTIFACT_DIR / "y_test.csv"

OVERFIT_GAP_THRESHOLD = 0.05


def load_selected_features():
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_outlier_caps():
    if not OUTLIER_CAPS_PATH.exists():
        return {}

    with open(OUTLIER_CAPS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_features(data, selected_features, data_name, ignored_extra_columns=None):
    ignored_extra_columns = set(ignored_extra_columns or [])
    missing_features = [feature for feature in selected_features if feature not in data.columns]
    extra_features = [
        feature
        for feature in data.columns
        if feature not in selected_features and feature not in ignored_extra_columns
    ]

    if missing_features:
        raise ValueError(
            f"{data_name} is missing required model features: {missing_features}\n"
            "Fresh data must contain the same selected feature columns used during training."
        )

    if extra_features:
        print(f"Warning: {data_name} has unused extra columns: {extra_features}")


def apply_outlier_caps(X, outlier_caps):
    X = X.copy()
    for feature, cap_value in outlier_caps.items():
        if feature in X.columns:
            X[feature] = X[feature].clip(upper=cap_value)
    return X


def load_split_dataset(x_path, y_path, selected_features, outlier_caps):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)["label"]
    validate_features(X, selected_features, x_path.name)
    X = apply_outlier_caps(X[selected_features], outlier_caps)
    return X, y


def load_fresh_dataset(csv_path, selected_features, outlier_caps, label_column):
    data = pd.read_csv(csv_path)
    validate_features(
        data,
        selected_features,
        Path(csv_path).name,
        ignored_extra_columns=[label_column],
    )

    X = data[selected_features]
    X = apply_outlier_caps(X, outlier_caps)
    y = data[label_column] if label_column in data.columns else None

    return X, y, data


def score_model(model, X, y, split_name):
    predictions = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

    print(f"\n--- {split_name} results ---")
    print(f"Rows: {len(X):,}")
    print("Label distribution:")
    print(y.value_counts().sort_index().to_string())
    print("Prediction distribution:")
    print(pd.Series(predictions).value_counts().sort_index().to_string())
    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions))
    print("\nClassification report:")
    print(classification_report(y, predictions, digits=4, zero_division=0))

    return metrics


def predict_without_labels(model, X, split_name):
    predictions = model.predict(X)

    print(f"\n--- {split_name} predictions ---")
    print(f"Rows: {len(X):,}")
    print("Prediction distribution:")
    print(pd.Series(predictions).value_counts().sort_index().to_string())

    return predictions


def print_metric_summary(train_metrics, test_metrics):
    print("\n--- Train vs unseen test summary ---")
    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if metric_name not in train_metrics or metric_name not in test_metrics:
            continue

        train_score = train_metrics[metric_name]
        test_score = test_metrics[metric_name]
        gap = train_score - test_score
        print(
            f"{metric_name:>9}: train={train_score:.4f} "
            f"test={test_score:.4f} gap={gap:.4f}"
        )


def print_generalization_verdict(train_metrics, test_metrics):
    accuracy_gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    f1_gap = train_metrics["f1"] - test_metrics["f1"]

    print("\n--- Verdict ---")
    if test_metrics["accuracy"] < 0.70 or test_metrics["f1"] < 0.70:
        print(
            "The model is not performing well on the unseen test split. "
            "Check the data split, feature processing, and class balance before using it."
        )
    elif accuracy_gap > OVERFIT_GAP_THRESHOLD or f1_gap > OVERFIT_GAP_THRESHOLD:
        print(
            "The model may be overfitting: train performance is meaningfully higher "
            "than unseen test performance."
        )
    else:
        print(
            "The saved model appears to generalize well on the unseen test split. "
            "There is no large train-test performance gap."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved XGBoost model on holdout or fresh unseen data."
    )
    parser.add_argument(
        "--fresh-data",
        type=Path,
        help=(
            "Path to a fresh CSV that was not used in training/preprocessing. "
            "It must contain the selected feature columns. Include a label column "
            "to calculate accuracy and other metrics."
        ),
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Ground-truth label column in the fresh CSV. Default: label.",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        help="Optional path to save fresh-data predictions as a CSV.",
    )
    parser.add_argument(
        "--skip-train-comparison",
        action="store_true",
        help="Only evaluate the requested data, without scoring training data.",
    )
    return parser.parse_args()


def warn_if_not_fresh_data(csv_path):
    known_training_source = (BASE_DIR.parent / "dataset" / "raw_phiusiil.csv").resolve()
    supplied_path = Path(csv_path).resolve()

    if supplied_path == known_training_source:
        print(
            "\nWarning: dataset/raw_phiusiil.csv looks like the original source dataset, "
            "not fresh unseen data. Use a completely separate CSV for a genuine test."
        )


def save_predictions(output_path, source_data, predictions):
    output = source_data.copy()
    output["predicted_label"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")


def evaluate_saved_holdout(model, selected_features, outlier_caps, skip_train_comparison):
    X_test, y_test = load_split_dataset(X_TEST_PATH, Y_TEST_PATH, selected_features, outlier_caps)
    test_metrics = score_model(model, X_test, y_test, "Saved holdout test data")

    if skip_train_comparison:
        return

    X_train, y_train = load_split_dataset(X_TRAIN_PATH, Y_TRAIN_PATH, selected_features, outlier_caps)
    train_metrics = score_model(model, X_train, y_train, "Training data")
    print_metric_summary(train_metrics, test_metrics)
    print_generalization_verdict(train_metrics, test_metrics)


def evaluate_fresh_data(args, model, selected_features, outlier_caps):
    warn_if_not_fresh_data(args.fresh_data)

    X_fresh, y_fresh, source_data = load_fresh_dataset(
        args.fresh_data,
        selected_features,
        outlier_caps,
        args.label_column,
    )

    if y_fresh is None:
        predictions = predict_without_labels(model, X_fresh, "Fresh unseen data")
        print(
            f"\nNo '{args.label_column}' column was found, so accuracy cannot be calculated. "
            "Add ground-truth labels to confirm whether the model is genuinely correct."
        )
    else:
        predictions = model.predict(X_fresh)
        fresh_metrics = score_model(model, X_fresh, y_fresh, "Fresh unseen data")

        if not args.skip_train_comparison:
            X_train, y_train = load_split_dataset(
                X_TRAIN_PATH,
                Y_TRAIN_PATH,
                selected_features,
                outlier_caps,
            )
            train_metrics = score_model(model, X_train, y_train, "Training data")
            print_metric_summary(train_metrics, fresh_metrics)
            print_generalization_verdict(train_metrics, fresh_metrics)

    if args.save_predictions:
        save_predictions(args.save_predictions, source_data, predictions)


def main():
    args = parse_args()
    selected_features = load_selected_features()
    outlier_caps = load_outlier_caps()
    model = joblib.load(MODEL_PATH)

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Outlier caps loaded: {len(outlier_caps)}")

    if args.fresh_data:
        evaluate_fresh_data(args, model, selected_features, outlier_caps)
    else:
        evaluate_saved_holdout(
            model,
            selected_features,
            outlier_caps,
            args.skip_train_comparison,
        )


if __name__ == "__main__":
    main()
