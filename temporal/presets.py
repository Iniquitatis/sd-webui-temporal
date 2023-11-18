import json
from shutil import rmtree

from temporal.fs import safe_get_directory
from temporal.interop import EXTENSION_DIR
from temporal.serialization import load_dict, save_object

PRESETS_DIR = EXTENSION_DIR / "presets"

presets = {}

def refresh_presets():
    presets.clear()

    if not PRESETS_DIR.is_dir():
        return

    for preset_dir in PRESETS_DIR.iterdir():
        if not preset_dir.is_dir():
            continue

        with open(preset_dir / "parameters.json", "r", encoding = "utf-8") as file:
            presets[preset_dir.name] = {}
            load_dict(presets[preset_dir.name], json.load(file), preset_dir, False)

def save_preset(name, ext_params):
    rmtree(PRESETS_DIR / name, ignore_errors = True)

    preset_dir = safe_get_directory(PRESETS_DIR / name)

    with open(preset_dir / "parameters.json", "w", encoding = "utf-8") as file:
        presets[name] = save_object(ext_params, preset_dir)
        json.dump(presets[name], file, indent = 4)

def delete_preset(name):
    rmtree(PRESETS_DIR / name, ignore_errors = True)
    presets.pop(name)
