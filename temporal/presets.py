from copy import deepcopy

from temporal.fs import iterate_subdirectories, load_json, recreate_directory, remove_directory, save_json
from temporal.interop import EXTENSION_DIR
from temporal.serialization import load_dict, save_object

PRESETS_DIR = EXTENSION_DIR / "presets"

presets = {}

def refresh_presets():
    presets.clear()

    for preset_dir in iterate_subdirectories(PRESETS_DIR):
        if data := load_json(preset_dir / "parameters.json"):
            presets[preset_dir.name] = {}
            load_dict(presets[preset_dir.name], data, preset_dir, False)

def save_preset(name, ext_params):
    preset_dir = recreate_directory(PRESETS_DIR / name)
    presets[name] = deepcopy(vars(ext_params))
    save_json(preset_dir / "parameters.json", save_object(ext_params, preset_dir))

def delete_preset(name):
    remove_directory(PRESETS_DIR / name)
    presets.pop(name)
