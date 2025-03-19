from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 2

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]

        for before, after in [
            ("normalize_contrast", "color_correction_normalize_contrast"),
            ("brightness", "color_balancing_brightness"),
            ("contrast", "color_balancing_contrast"),
            ("saturation", "color_balancing_saturation"),
            ("noise_relative", "noise_amount_relative"),
            ("modulation_relative", "modulation_amount_relative"),
            ("tinting_relative", "tinting_amount_relative"),
            ("sharpening_amount", "sharpening_strength"),
            ("sharpening_relative", "sharpening_amount_relative"),
            ("translation_x", "transformation_translation_x"),
            ("translation_y", "transformation_translation_y"),
            ("rotation", "transformation_rotation"),
            ("scaling", "transformation_scaling"),
            ("symmetrize", "symmetry_enabled"),
            ("custom_code", "custom_code_code"),
        ]:
            ext_params[after] = ext_params.pop(before)

        for key in [
            "noise_compression_amount",
            "color_correction_amount",
            "color_balancing_amount",
            "sharpening_amount",
            "transformation_amount",
            "symmetry_amount",
            "blurring_amount",
            "custom_code_amount",
        ]:
            ext_params[key] = 1.0

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
