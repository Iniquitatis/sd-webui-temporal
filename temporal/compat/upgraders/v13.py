from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 13

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "symmetry_horizontal": True,
            "symmetry_vertical": False,
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
