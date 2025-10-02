from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_json, load_text, save_json, save_text


class _(Upgrader):
    version = 11

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "blurring_blend_mode": "normal",
            "color_balancing_blend_mode": "normal",
            "color_correction_blend_mode": "normal",
            "color_overlay_amount": ext_params.pop("tinting_amount"),
            "color_overlay_amount_relative": ext_params.pop("tinting_amount_relative"),
            "color_overlay_blend_mode": ext_params.pop("tinting_mode"),
            "color_overlay_color": ext_params.pop("tinting_color"),
            "color_overlay_mask": ext_params.pop("tinting_mask"),
            "color_overlay_mask_normalized": ext_params.pop("tinting_mask_normalized"),
            "color_overlay_mask_inverted": ext_params.pop("tinting_mask_inverted"),
            "color_overlay_mask_blurring": ext_params.pop("tinting_mask_blurring"),
            "custom_code_blend_mode": "normal",
            "image_overlay_amount": ext_params.pop("modulation_amount"),
            "image_overlay_amount_relative": ext_params.pop("modulation_amount_relative"),
            "image_overlay_blend_mode": ext_params.pop("modulation_mode"),
            "image_overlay_image": ext_params.pop("modulation_image"),
            "image_overlay_blurring": ext_params.pop("modulation_blurring"),
            "image_overlay_mask": ext_params.pop("modulation_mask"),
            "image_overlay_mask_normalized": ext_params.pop("modulation_mask_normalized"),
            "image_overlay_mask_inverted": ext_params.pop("modulation_mask_inverted"),
            "image_overlay_mask_blurring": ext_params.pop("modulation_mask_blurring"),
            "median_blend_mode": "normal",
            "morphology_blend_mode": "normal",
            "noise_compression_blend_mode": "normal",
            "noise_overlay_amount": ext_params.pop("noise_amount"),
            "noise_overlay_amount_relative": ext_params.pop("noise_amount_relative"),
            "noise_overlay_blend_mode": ext_params.pop("noise_mode"),
            "noise_overlay_mask": ext_params.pop("noise_mask"),
            "noise_overlay_mask_normalized": ext_params.pop("noise_mask_normalized"),
            "noise_overlay_mask_inverted": ext_params.pop("noise_mask_inverted"),
            "noise_overlay_mask_blurring": ext_params.pop("noise_mask_blurring"),
            "palettization_blend_mode": "normal",
            "sharpening_blend_mode": "normal",
            "symmetry_blend_mode": "normal",
            "transformation_blend_mode": "normal",
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True
