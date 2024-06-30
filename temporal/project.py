from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from modules import shared as webui_shared
from modules.options import Options
from modules.processing import StableDiffusionProcessingImg2Img

from temporal.compat import VERSION, upgrade_project
from temporal.interop import ControlNetUnitList, ControlNetUnitWrapper
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise
if TYPE_CHECKING:
    from temporal.pipeline import Pipeline
from temporal.serialization import BasicObjectSerializer, Serializer
from temporal.utils import logging
from temporal.utils.fs import clear_directory, ensure_directory_exists, remove_entry
from temporal.utils.image import NumpyImage
from temporal.utils.object import copy_with_overrides
from temporal.video_renderer import VideoRenderer
from temporal.web_ui import has_schedulers


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


class Project(Serializable):
    path: Path = Field(Path("outputs/temporal/untitled"), saved = False)
    version: int = Field(VERSION)
    # NOTE: The next three fields should be assigned manually
    options: Options = Field(factory = lambda: copy_with_overrides(opts, data = opts.data.copy()))
    processing: StableDiffusionProcessingImg2Img = Field(factory = StableDiffusionProcessingImg2Img)
    controlnet_units: Optional[ControlNetUnitList] = Field(factory = ControlNetUnitList)
    initial_noise: InitialNoiseParams = Field(factory = InitialNoiseParams)
    pipeline: "Pipeline" = Field(factory = lambda: _make_pipeline())
    iteration: IterationData = Field(factory = IterationData)

    def load(self, dir: Path) -> None:
        upgrade_project(dir)
        super().load(dir / "project")

    def save(self, dir: Path) -> None:
        super().save(dir / "project")

    def get_description(self) -> str:
        return "\n\n".join(f"{k}: {v}" for k, v in {
            "Name": self.path.name,
            "Prompt": self.processing.prompt,
            "Negative prompt": self.processing.negative_prompt,
            "Checkpoint": self.options.sd_model_checkpoint,
            "Last frame": self.get_last_frame_index(),
            "Saved frames": self.get_actual_frame_count(),
        }.items())

    def get_first_frame_index(self) -> int:
        return min((_parse_frame_index(x)[0] for x in self._iterate_frame_paths()), default = 0)

    def get_last_frame_index(self) -> int:
        return max((_parse_frame_index(x)[0] for x in self._iterate_frame_paths()), default = 0)

    def get_actual_frame_count(self, parallel_index: int = 1) -> int:
        return sum(_parse_frame_index(x)[1] == parallel_index for x in self._iterate_frame_paths())

    def list_all_frame_paths(self, parallel_index: int = 1) -> list[Path]:
        return sorted((x for x in self._iterate_frame_paths() if _parse_frame_index(x)[1] == parallel_index), key = lambda x: x.name)

    def delete_all_frames(self) -> None:
        clear_directory(self.path, "*.png")

    def delete_intermediate_frames(self) -> None:
        kept_indices = self.get_first_frame_index(), self.get_last_frame_index()

        for image_path in self._iterate_frame_paths():
            frame_index, _ = _parse_frame_index(image_path)

            if frame_index not in kept_indices:
                remove_entry(image_path)

    def delete_session_data(self) -> None:
        for module in self.pipeline.modules.values():
            module.reset()

        self.iteration = IterationData()

    def render_video(self, renderer: VideoRenderer, is_final: bool, parallel_index: int = 1) -> Path:
        video_path = ensure_directory_exists(self.path / "videos") / f"{parallel_index:02d}-{'final' if is_final else 'draft'}.mp4"
        renderer.enqueue_video_render(video_path, self.list_all_frame_paths(parallel_index), is_final)
        return video_path

    def _iterate_frame_paths(self) -> Iterator[Path]:
        return self.path.glob("*.png")


def _make_pipeline() -> "Pipeline":
    from temporal.pipeline import Pipeline
    return Pipeline()


def _parse_frame_index(image_path: Path) -> tuple[int, int]:
    if image_path.is_file():
        try:
            return int(image_path.stem), 1
        except:
            pass

        try:
            frame_index, parallel_index = image_path.stem.split("-")
            return int(frame_index), int(parallel_index)
        except:
            pass

        logging.warning(f"{image_path.stem} doesn't match the frame name format")

    return 0, 0


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
    ] + (["scheduler"] if has_schedulers() else [])


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
