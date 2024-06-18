from pathlib import Path
from typing import Any

from temporal.preset import Preset
from temporal.utils.collection import natural_sort
from temporal.utils.fs import iterate_subdirectories, remove_entry


class PresetStore:
    def __init__(self, path: Path, sorting_order: str) -> None:
        self.path = path
        self.sorting_order = sorting_order
        self.preset_names: list[str] = []

    def refresh_presets(self) -> None:
        self.preset_names.clear()
        self.preset_names.extend(x.name for x in iterate_subdirectories(self.path))
        self._sort()

    def open_preset(self, name: str) -> Preset:
        result = Preset()
        result.load(self.path / name)
        return result

    def save_preset(self, name: str, data: dict[str, Any]) -> None:
        preset = Preset(data = data)
        preset.save(self.path / name)

        if name not in self.preset_names:
            self.preset_names.append(name)
            self._sort()

    def delete_preset(self, name: str) -> None:
        remove_entry(self.path / name)
        self.preset_names.remove(name)

    def _sort(self) -> None:
        if self.sorting_order == "alphabetical":
            self.preset_names[:] = natural_sort(self.preset_names)
        elif self.sorting_order == "date":
            self.preset_names[:] = sorted(self.preset_names, key = lambda x: (self.path / x).stat().st_ctime_ns)
