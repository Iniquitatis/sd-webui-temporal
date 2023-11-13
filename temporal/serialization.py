import numpy as np
from PIL import Image

from temporal.image_utils import load_image

def load_dict(d, data, data_dir):
    for key, value in data.items():
        if key in d:
            d[key] = _load_value(value, data_dir)

def save_dict(d, data_dir, filter = None):
    return {k: _save_value(v, data_dir) for k, v in d.items() if not filter or k in filter}

def load_object(obj, data, data_dir):
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, _load_value(value, data_dir))

def save_object(obj, data_dir, filter = None):
    return {k: _save_value(v, data_dir) for k, v in vars(obj).items() if not filter or k in filter}

def _load_value(value, data_dir):
    if isinstance(value, bool | int | float | str | None):
        return value

    elif isinstance(value, list):
        return [_load_value(x, data_dir) for x in value]

    elif isinstance(value, dict):
        im_type = value.get("im_type", "")
        im_path = data_dir / value.get("filename", "")

        if im_type == "pil":
            return load_image(im_path)
        elif im_type == "np":
            return np.array(load_image(im_path))
        else:
            return {k: _load_value(v, data_dir) for k, v in value.items()}

def _save_value(value, data_dir):
    if isinstance(value, bool | int | float | str | None):
        return value

    elif isinstance(value, tuple):
        return tuple(_save_value(x, data_dir) for x in value)

    elif isinstance(value, list):
        return [_save_value(x, data_dir) for x in value]

    elif isinstance(value, dict):
        return {k: _save_value(v, data_dir) for k, v in value.items()}

    elif isinstance(value, Image.Image):
        filename = f"{id(value)}.png"
        value.save(data_dir / filename)
        return {"im_type": "pil", "filename": filename}

    elif isinstance(value, np.ndarray):
        filename = f"{id(value)}.png"
        Image.fromarray(value).save(data_dir / filename)
        return {"im_type": "np", "filename": filename}
