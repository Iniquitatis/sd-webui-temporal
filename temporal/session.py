from typing import Optional

from modules.options import Options
from modules.processing import StableDiffusionProcessingImg2Img

from temporal.data import ExtensionData
from temporal.interop import ControlNetUnitWrapper
from temporal.meta.serializable import Serializable, field
from temporal.serialization import BasicObjectSerializer


class Session(Serializable):
    options: Optional[Options] = field(None)
    processing: Optional[StableDiffusionProcessingImg2Img] = field(None)
    controlnet_units: Optional[list[ControlNetUnitWrapper]] = field(None)
    ext_data: Optional[ExtensionData] = field(None)


class _(BasicObjectSerializer[Options], create = False):
    keys = [
        "sd_model_checkpoint",
        "sd_vae",
        "CLIP_stop_at_last_layers",
        "always_discard_next_to_last_sigma",
    ]


class _(BasicObjectSerializer[StableDiffusionProcessingImg2Img], create = False):
    keys = [
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
    ]


class _(BasicObjectSerializer[ControlNetUnitWrapper], create = False):
    keys = [
        "instance.image",
        "instance.enabled",
        "instance.low_vram",
        "instance.pixel_perfect",
        "instance.module",
        "instance.model",
        "instance.weight",
        "instance.guidance_start",
        "instance.guidance_end",
        "instance.processor_res",
        "instance.threshold_a",
        "instance.threshold_b",
        "instance.control_mode",
        "instance.resize_mode",
    ]
