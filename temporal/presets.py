from temporal.fs import iterate_subdirectories, load_json, recreate_directory, remove_directory, save_json
from temporal.interop import EXTENSION_DIR
from temporal.serialization import load_dict, save_object

PRESETS_DIR = EXTENSION_DIR / "presets"

preset_names = []

def refresh_presets():
    preset_names.clear()
    preset_names.extend(x.name for x in iterate_subdirectories(PRESETS_DIR))

def load_preset(name, ext_params):
    if not (preset_dir := (PRESETS_DIR / name)).exists():
        return

    preset = {}
    load_dict(preset, load_json(preset_dir / "parameters.json", {}), preset_dir, False)

    for k, v in preset.items():
        setattr(ext_params, k, v)

def save_preset(name, ext_params):
    preset_dir = recreate_directory(PRESETS_DIR / name)
    save_json(preset_dir / "parameters.json", save_object(ext_params, preset_dir))

    if name not in preset_names:
        preset_names.append(name)

def delete_preset(name):
    remove_directory(PRESETS_DIR / name)
    preset_names.remove(name)
