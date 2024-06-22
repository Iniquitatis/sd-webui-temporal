from typing import TYPE_CHECKING, Any, Optional

from modules import shared as webui_shared
from modules.options import Options
from modules.processing import StableDiffusionProcessingImg2Img

from temporal.interop import ControlNetUnitWrapper
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise
if TYPE_CHECKING:
    from temporal.pipeline import Pipeline
    from temporal.project import Project
from temporal.serialization import BasicObjectSerializer
from temporal.utils.image import NumpyImage
from temporal.utils.object import copy_with_overrides


# FIXME: To shut up the type checker
opts: Options = getattr(webui_shared, "opts")


class InitialNoiseParams(Serializable):
    factor: float = Field(0.0)
    noise: Noise = Field(factory = Noise)
    use_initial_seed: bool = Field(False)


class IterationData(Serializable):
    images: list[NumpyImage] = Field(factory = list)
    index: int = Field(1)
    module_id: Optional[str] = Field(None)


class Session(Serializable):
    # NOTE: The next four fields should be assigned manually
    options: Options = Field(factory = lambda: copy_with_overrides(opts, data = opts.data.copy()))
    processing: StableDiffusionProcessingImg2Img = Field(factory = StableDiffusionProcessingImg2Img)
    controlnet_units: Optional[list[ControlNetUnitWrapper]] = Field(factory = list)
    project: "Project" = Field(None, saved = False)
    initial_noise: InitialNoiseParams = Field(factory = InitialNoiseParams)
    pipeline: "Pipeline" = Field()
    iteration: IterationData = Field(factory = IterationData)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from temporal.pipeline import Pipeline

        super().__init__(*args, **kwargs)
        self.pipeline = Pipeline()


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
