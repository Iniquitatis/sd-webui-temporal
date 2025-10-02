from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 6

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "multisampling_samples": ext_params.pop("image_samples", 1),
            "multisampling_batch_size": ext_params.pop("batch_size", 1),
            "multisampling_algorithm": "mean",
            "multisampling_easing": 0.0,
            "frame_merging_frames": ext_params.pop("merged_frames", 1),
            "frame_merging_algorithm": "mean",
            "frame_merging_easing": ext_params.pop("merged_frames_easing", 0.0),
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
