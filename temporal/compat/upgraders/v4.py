from pathlib import Path
from typing import Any

import numpy as np

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_json, save_text
from temporal.utils.image import load_image


class _(Upgrader):
    version = 4

    def upgrade(self, path: Path) -> bool:
        def upgrade_value(value: Any) -> Any:
            if isinstance(value, dict):
                type = value.get("type", None)

                if type == "list":
                    return {"type": "list", "data": [upgrade_value(x) for x in value["data"]]}
                elif type == "dict":
                    return {"type": "dict", "data": {k: upgrade_value(v) for k, v in value["data"].items()}}
                elif type == "np":
                    im_path = path / "session" / value["filename"]
                    arr_path = im_path.with_suffix(".npy")
                    np.save(arr_path, np.array(load_image(im_path)))
                    im_path.unlink()
                    return {"type": "np", "filename": arr_path.name}
                else:
                    return value
            else:
                return value

        def upgrade_values(d: dict[str, Any]) -> dict[str, Any]:
            return {k: upgrade_value(v) for k, v in d.items()}

        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
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
