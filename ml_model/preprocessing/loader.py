from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"


def load_phiusiil() -> pd.DataFrame:

    dataset_path = DATASET_DIR / "raw_phiusiil.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )
    
    return pd.read_csv(dataset_path)