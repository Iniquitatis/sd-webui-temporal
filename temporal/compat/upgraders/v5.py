from pathlib import Path
from typing import Any

import numpy as np

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_json, save_text
from temporal.utils.image import load_image, pil_to_np


class _(Upgrader):
    version = 5

    def upgrade(self, path: Path) -> bool:
        def upgrade_value(value: Any) -> Any:
            if isinstance(value, dict):
                type = value.get("type", None)

                if type == "list":
                    return {"type": "list", "data": [upgrade_value(x) for x in value["data"]]}
                elif type == "dict":
                    return {"type": "dict", "data": {k: upgrade_value(v) for k, v in value["data"].items()}}
                elif type == "np":
                    arr_path = path / "session" / value["filename"]
                    arrz_path = arr_path.with_suffix(".npz")
                    np.savez_compressed(arrz_path, np.load(arr_path))
                    arr_path.unlink()
                    return {"type": "np", "filename": arrz_path.name}
                else:
                    return value
            else:
                return value

        def upgrade_values(d: dict[str, Any]) -> dict[str, Any]:
            return {k: upgrade_value(v) for k, v in d.items()}

        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"
        buffer_dir = path / "session" / "buffer"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        data["shared_params"] = upgrade_values(data.get("shared_params", {}))
        data["generation_params"] = upgrade_values(data.get("generation_params", {}))

        for i, unit_data in enumerate(data.get("controlnet_params", [])):
            data["controlnet_params"][i] = upgrade_values(unit_data)

        data["extension_params"] = upgrade_values(data.get("extension_params", {}))

        save_json(params_path, data)

        image_paths = sorted(buffer_dir.glob("*.png"), key = lambda x: int(x.stem))

        np.savez_compressed(buffer_dir / "buffer.npz", np.stack([
            pil_to_np(load_image(x))
            for x in image_paths
        ], axis = 0))

        for path in image_paths:
            path.unlink()

        save_json(buffer_dir / "data.json", {
            "array": {
                "type": "np",
                "filename": "buffer.npz",
            },
            "last_index": 0,
        })

        save_text(version_path, str(self.version))

        return True
