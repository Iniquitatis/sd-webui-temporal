from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 8

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]

        for key in ["multisampling_algorithm", "frame_merging_algorithm"]:
            if ext_params[key] == "mean":
                ext_params[key] = "arithmetic_mean"

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
