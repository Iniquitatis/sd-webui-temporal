from pathlib import Path
from types import SimpleNamespace

from temporal.serialization import load_dict, save_object
from temporal.utils.fs import load_json, recreate_directory, save_json

class Preset:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_ext_params(self, ext_params: SimpleNamespace) -> None:
        data = {}
        load_dict(data, load_json(self.path / "parameters.json", {}), self.path, False)

        for k, v in data.items():
            setattr(ext_params, k, v)

    def write_ext_params(self, ext_params: SimpleNamespace) -> None:
        recreate_directory(self.path)
        save_json(self.path / "parameters.json", save_object(ext_params, self.path))
