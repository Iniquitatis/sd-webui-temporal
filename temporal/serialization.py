import numpy as np
from PIL import Image

from temporal.image_utils import load_image, save_image

def load_dict(d, data, data_dir, existing_only = True):
    for key, value in data.items():
        if not existing_only or key in d:
            d[key] = _load_value(value, data_dir)

def save_dict(d, data_dir, filter = None):
    return {k: _save_value(v, data_dir) for k, v in d.items() if not filter or k in filter}

def load_object(obj, data, data_dir, existing_only = True):
    for key, value in data.items():
        if not existing_only or hasattr(obj, key):
            setattr(obj, key, _load_value(value, data_dir))

def save_object(obj, data_dir, filter = None):
    return {k: _save_value(v, data_dir) for k, v in vars(obj).items() if not filter or k in filter}

def _load_value(value, data_dir):
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
            return np.array(load_image(data_dir / value.get("filename", "")))

        else:
            print(f"WARNING: Cannot load value of type {type}")

def _save_value(value, data_dir):
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
        filename = f"{id(value)}.png"
        save_image(Image.fromarray(value), data_dir / filename)
        return {"type": "np", "filename": filename}

    else:
        print(f"WARNING: Cannot save value of type {value.__class__.__name__}")
