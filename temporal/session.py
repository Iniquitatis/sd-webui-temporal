from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from modules.options import Options
from modules.processing import StableDiffusionProcessingImg2Img

from temporal.serialization import load_dict, load_object, save_dict, save_object
from temporal.utils.fs import load_json, recreate_directory, save_json


saved_ext_param_ids: list[str] = []


class Session:
    def __init__(self, opts: Optional[Options] = None, p: Optional[StableDiffusionProcessingImg2Img] = None, cn_units: Optional[list[Any]] = None, ext_params: Optional[SimpleNamespace] = None) -> None:
        self.opts = opts
        self.p = p
        self.cn_units = cn_units
        self.ext_params = ext_params

    def load(self, path: Path) -> None:
        if not (data := load_json(path / "parameters.json")):
            return

        # NOTE: `p.override_settings` juggles VAEs back-and-forth, slowing down the process considerably
        if self.opts is not None:
            load_dict(self.opts.data, data.get("shared_params", {}), path)

        if self.p is not None:
            load_object(self.p, data.get("generation_params", {}), path)

        if self.cn_units is not None:
            for unit_data, cn_unit in zip(data.get("controlnet_params", []), self.cn_units):
                load_object(cn_unit, unit_data, path)

        if self.ext_params is not None:
            load_object(self.ext_params, data.get("extension_params", {}), path)

    def save(self, path: Path) -> None:
        recreate_directory(path)
        save_json(path / "parameters.json", dict(
            shared_params = save_dict(self.opts.data, path, [
                "sd_model_checkpoint",
                "sd_vae",
                "CLIP_stop_at_last_layers",
                "always_discard_next_to_last_sigma",
            ]) if self.opts is not None else {},
            generation_params = save_object(self.p, path, [
                "prompt",
                "negative_prompt",
                "init_images",
                "image_mask",
                "resize_mode",
                "mask_blur_x",
                "mask_blur_y",
                "inpainting_mask_invert",
                "inpainting_fill",
                "inpaint_full_res",
                "inpaint_full_res_padding",
                "sampler_name",
                "steps",
                "refiner_checkpoint",
                "refiner_switch_at",
                "width",
                "height",
                "cfg_scale",
                "denoising_strength",
                "seed",
                "seed_enable_extras",
                "subseed",
                "subseed_strength",
                "seed_resize_from_w",
                "seed_resize_from_h",
            ]) if self.p is not None else {},
            controlnet_params = list(
                save_object(cn_unit, path, [
                    "image",
                    "enabled",
                    "low_vram",
                    "pixel_perfect",
                    "module",
                    "model",
                    "weight",
                    "guidance_start",
                    "guidance_end",
                    "processor_res",
                    "threshold_a",
                    "threshold_b",
                    "control_mode",
                    "resize_mode",
                ])
                for cn_unit in self.cn_units
            ) if self.cn_units is not None else [],
            extension_params = save_object(self.ext_params, path, saved_ext_param_ids) if self.ext_params is not None else {},
        ))
