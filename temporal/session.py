from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from modules import shared as webui_shared
from modules.options import Options
from modules.processing import StableDiffusionProcessingImg2Img

from temporal.interop import ControlNetUnitList, ControlNetUnitWrapper
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise
if TYPE_CHECKING:
    from temporal.pipeline import Pipeline
    from temporal.project import Project
from temporal.serialization import BasicObjectSerializer, Serializer
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
    controlnet_units: Optional[ControlNetUnitList] = Field(factory = ControlNetUnitList)
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


class _(Serializer[ControlNetUnitWrapper]):
    keys = [
        "image",
        "enabled",
        "low_vram",
        "pixel_perfect",
        "effective_region_mask",
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
    ]

    @classmethod
    def read(cls, obj, ar):
        for key in cls.keys:
            value = ar[key].create()

            if isinstance(object_value := getattr(obj.instance, key), Enum):
                value = type(object_value)(value)

            setattr(obj.instance, key, value)

        return obj

    @classmethod
    def write(cls, obj, ar):
        for key in cls.keys:
            value = getattr(obj.instance, key)

            if isinstance(value, Enum):
                value = value.value

            ar[key].write(value)


class _(Serializer[ControlNetUnitList]):
    @classmethod
    def read(cls, obj, ar):
        for unit, child in zip(obj.units, ar):
            child.read(unit)

        return obj

    @classmethod
    def write(cls, obj, ar):
        for unit in obj.units:
            ar.make_child().write(unit)
