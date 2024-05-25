from temporal.fs import iterate_subdirectories, remove_entry
from temporal.preset import Preset

class PresetStore:
    def __init__(self, path):
        self.path = path
        self.preset_names = []

    def refresh_presets(self):
        self.preset_names.clear()
        self.preset_names.extend(x.name for x in iterate_subdirectories(self.path))

    def open_preset(self, name):
        return Preset(self.path / name)

    def save_preset(self, name, ext_params):
        preset = Preset(self.path / name)
        preset.write_ext_params(ext_params)

        if name not in self.preset_names:
            self.preset_names.append(name)

    def delete_preset(self, name):
        remove_entry(self.path / name)
        self.preset_names.remove(name)
