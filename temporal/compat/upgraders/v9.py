from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 9

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]

        for feature in ["multisampling", "frame_merging"]:
            if (algo := ext_params.pop(f"{feature}_algorithm")) != "median":
                ext_params[f"{feature}_preference"] = {
                    "harmonic_mean": -2.0,
                    "geometric_mean": -1.0,
                    "arithmetic_mean": 0.0,
                    "root_mean_square": 1.0,
                }[algo]
            else:
                ext_params[f"{feature}_trimming"] = 0.5
                ext_params[f"{feature}_preference"] = 1.0

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
