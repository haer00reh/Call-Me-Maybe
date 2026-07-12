from pathlib import Path
import json
from enum import Enum, auto
from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


def _normalize_numeric_values(value: dict[str, Any]) -> dict[str, Any]:
    """Convert integer values in nested structures to floats."""
    if isinstance(value, dict):
        return {
            key: _normalize_numeric_values(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, list):
        return [_normalize_numeric_values(item) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


class Phase(Enum):
    """State machine phases for function-call generation."""

    PREFIX = auto()
    FUNCTION_NAME = auto()
    PARAMETERS = auto()
    DONE = auto()


class FSMState(BaseModel):
    """Track progress while building a function-call response."""

    model_config = ConfigDict(validate_assignment=True)

    phase: Phase = Phase.PREFIX
    chosen_function: Optional[str] = None
    current_param_index: int = 0
    function_name_tokens: List[int] = Field(default_factory=list)
    current_value_tokens: List[int] = Field(default_factory=list)


def parse_json(file_path: str | Path) -> Any:
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
