from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 7

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params["preprocessing_order"] = {
            "type": "list",
            "data": [
                "noise_compression",
                "color_correction",
                "color_balancing",
                "noise",
                "modulation",
                "tinting",
                "sharpening",
                "transformation",
                "symmetry",
                "blurring",
                "custom_code",
            ],
        }

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
