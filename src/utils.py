from pathlib import Path
import json


def parse_json(file_path: str | Path) -> dict:
    """Load and return JSON data from a file path."""
    if not isinstance(file_path, (str, Path)):
        raise TypeError("file_path must be either str or Path object")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(f"there is an issue about the path provided: {e}")
        exit()
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {file_path}: {e}")
        exit()
    return data
