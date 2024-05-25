from collections.abc import Container
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from temporal.utils.image import load_image, save_image
from temporal.utils.numpy import load_array, save_array

def load_dict(d: dict[str, Any], data: dict[str, Any], data_dir: Path, existing_only: bool = True) -> None:
    for key, value in data.items():
        if not existing_only or key in d:
            d[key] = _load_value(value, data_dir)

def save_dict(d: dict[str, Any], data_dir: Path, filter: Optional[Container[str]] = None) -> dict[str, Any]:
    return {k: _save_value(v, data_dir) for k, v in d.items() if not filter or k in filter}

def load_object(obj: Any, data: dict[str, Any], data_dir: Path, existing_only: bool = True) -> None:
    for key, value in data.items():
        if not existing_only or hasattr(obj, key):
            setattr(obj, key, _load_value(value, data_dir))

def save_object(obj: Any, data_dir: Path, filter: Optional[Container[str]] = None) -> dict[str, Any]:
    return {k: _save_value(v, data_dir) for k, v in vars(obj).items() if not filter or k in filter}

def _load_value(value: Any, data_dir: Path) -> Any:
    if isinstance(value, bool | int | float | str | None):
        return value

    elif isinstance(value, dict):
        type = value.get("type", "")

        if type == "tuple":
            return tuple(_load_value(x, data_dir) for x in value.get("data", []))

        elif type == "list":
            return [_load_value(x, data_dir) for x in value.get("data", [])]

        elif type == "dict":
            return {k: _load_value(v, data_dir) for k, v in value.get("data", {}).items()}

        elif type == "pil":
            return load_image(data_dir / value.get("filename", ""))

        elif type == "np":
            return load_array(data_dir / value.get("filename", ""))

        else:
            print(f"WARNING: Cannot load value of type {type}")

def _save_value(value: Any, data_dir: Path) -> Any:
    if isinstance(value, bool | int | float | str | None):
        return value

    elif isinstance(value, tuple):
        return {"type": "tuple", "data": [_save_value(x, data_dir) for x in value]}

    elif isinstance(value, list):
        return {"type": "list", "data": [_save_value(x, data_dir) for x in value]}

    elif isinstance(value, dict):
        return {"type": "dict", "data": {k: _save_value(v, data_dir) for k, v in value.items()}}

    elif isinstance(value, Image.Image):
        filename = f"{id(value)}.png"
        save_image(value, data_dir / filename)
        return {"type": "pil", "filename": filename}

    elif isinstance(value, np.ndarray):
        filename = f"{id(value)}.npz"
        save_array(value, data_dir / filename)
        return {"type": "np", "filename": filename}

    else:
        print(f"WARNING: Cannot save value of type {value.__class__.__name__}")
