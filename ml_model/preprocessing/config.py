from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"

ARTIFACTS_DIR = PROJECT_ROOT / "ml_model" / "artifacts"

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)