from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 16

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        params_path = path / "project" / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "image_filtering_order": ext_params.pop("preprocessing_order"),
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
