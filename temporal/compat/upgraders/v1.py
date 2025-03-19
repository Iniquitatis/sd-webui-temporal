from pathlib import Path
from typing import Any

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 1

    def upgrade(self, path: Path) -> bool:
        def upgrade_value(value: Any) -> Any:
            if isinstance(value, list):
                return {"type": "list", "data": [upgrade_value(x) for x in value]}
            elif isinstance(value, dict):
                if "im_type" in value:
                    return {"type": value["im_type"], "filename": value["filename"]}
                else:
                    return {"type": "dict", "data": {k: upgrade_value(v) for k, v in value.items()}}
            else:
                return value

        def upgrade_values(d: dict[str, Any]) -> dict[str, Any]:
            return {k: upgrade_value(v) for k, v in d.items()}

        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if "im_type" not in load_text(params_path, ""):
            return False

        data = load_json(params_path, {})

        data["shared_params"] = upgrade_values(data.get("shared_params", {}))
        data["generation_params"] = upgrade_values(data.get("generation_params", {}))

        for i, unit_data in enumerate(data.get("controlnet_params", [])):
            data["controlnet_params"][i] = upgrade_values(unit_data)

        data["extension_params"] = upgrade_values(data.get("extension_params", {}))

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
