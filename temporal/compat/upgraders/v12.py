from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 12

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "noise_overlay_scale": 1,
            "noise_overlay_octaves": 1,
            "noise_overlay_lacunarity": 2.0,
            "noise_overlay_persistence": 0.5,
            "noise_overlay_seed": 0,
            "noise_overlay_use_dynamic_seed": True,
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
