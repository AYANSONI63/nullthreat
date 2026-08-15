import json
from pathlib import Path


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary as a JSON file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: Path) -> dict:
    """
    Load a JSON file and return it as a dictionary.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)