from pathlib import Path
from typing import Any

from temporal.preset import Preset
from temporal.utils.collection import natural_sort
from temporal.utils.fs import iterate_subdirectories, remove_entry


class PresetStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.preset_names = []

    def refresh_presets(self) -> None:
        self.preset_names.clear()
        self.preset_names.extend(x.name for x in iterate_subdirectories(self.path))
        self.preset_names[:] = natural_sort(self.preset_names)

    def open_preset(self, name: str) -> Preset:
        result = Preset()
        result.load(self.path / name)
        return result

    def save_preset(self, name: str, data: dict[str, Any]) -> None:
        preset = Preset(data = data)
        preset.save(self.path / name)

        if name not in self.preset_names:
            self.preset_names.append(name)
            self.preset_names[:] = natural_sort(self.preset_names)

    def delete_preset(self, name: str) -> None:
        remove_entry(self.path / name)
        self.preset_names.remove(name)
