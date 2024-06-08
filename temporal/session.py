from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from modules.options import Options
from modules.processing import StableDiffusionProcessingImg2Img

from temporal.image_filterer import ImageFilterer
from temporal.interop import ControlNetUnitWrapper
from temporal.meta.serializable import Serializable, field
if TYPE_CHECKING:
    from temporal.pipeline import Pipeline
from temporal.serialization import BasicObjectSerializer
from temporal.video_renderer import VideoRenderer


class OutputParams(Serializable):
    output_dir: Path = field(Path("outputs/temporal"))
    project_subdir: str = field("untitled")


class InitialNoiseParams(Serializable):
    factor: float = field(0.0)
    scale: int = field(1)
    octaves: int = field(1)
    lacunarity: float = field(2.0)
    persistence: float = field(0.5)


class Session(Serializable):
    options: Options = field()
    processing: StableDiffusionProcessingImg2Img = field()
    controlnet_units: Optional[list[ControlNetUnitWrapper]] = field()
    output: OutputParams = field(factory = OutputParams, saved = False)
    initial_noise: InitialNoiseParams = field(factory = InitialNoiseParams)
    pipeline: "Pipeline" = field()
    image_filterer: ImageFilterer = field(factory = ImageFilterer)
    video_renderer: VideoRenderer = field(factory = VideoRenderer, saved = False)

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
