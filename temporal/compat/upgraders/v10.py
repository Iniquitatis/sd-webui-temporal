from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 10

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "initial_noise_factor": float(ext_params.pop("noise_for_first_frame")),
            "initial_noise_scale": 1,
            "initial_noise_octaves": 1,
            "initial_noise_lacunarity": 2.0,
            "initial_noise_persistence": 0.5,
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
