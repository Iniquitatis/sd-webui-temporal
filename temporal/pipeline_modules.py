from pathlib import Path
from typing import Optional, Type

import gradio as gr
import numpy as np
from numpy.typing import NDArray

from modules.processing import Processed
from modules.sd_samplers import visible_sampler_names

from temporal.global_options import global_options
from temporal.meta.configurable import Configurable, ui_param
from temporal.meta.serializable import field
from temporal.metrics import Metrics
from temporal.project import render_project_video
from temporal.session import Session
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, ensure_image_dims, match_image, np_to_pil, pil_to_np
from temporal.utils.math import lerp, quantize
from temporal.utils.numpy import average_array, make_eased_weight_array, saturate_array
from temporal.utils.object import copy_with_overrides
from temporal.utils.time import wait_until
from temporal.video_renderer import video_render_queue
from temporal.web_ui import image_save_queue, process_images, save_processed_image


PIPELINE_MODULES: dict[str, Type["PipelineModule"]] = {}


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    icon: str = "\U00002699"

    enabled: bool = field(False)
    preview: bool = field(True)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return images

    def finalize(self, images: list[NumpyImage], session: Session) -> None:
        pass


class AveragingModule(PipelineModule):
    id = "averaging"
    name = "Averaging"
    icon = "\U0001f553"

    frames: int = ui_param("Frame count", gr.Number, precision = 0, minimum = 1, step = 1, value = 1)
    trimming: float = ui_param("Trimming", gr.Slider, minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0)
    easing: float = ui_param("Easing", gr.Slider, minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)
    preference: float = ui_param("Preference", gr.Slider, minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0)

    buffer: NDArray[np.float_] = field(factory = lambda: np.empty((0,)))
    last_index: int = field(0)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer.shape[0] == 0:
            self.buffer = np.stack([np.repeat(
                ensure_image_dims(image, "RGB", (session.processing.width, session.processing.height))[np.newaxis, ...],
                self.frames,
                axis = 0,
            ) for image in images], 0)

        for sub, image in zip(self.buffer, images):
            sub[self.last_index] = match_image(image, sub[0])

        self.last_index += 1
        self.last_index %= self.frames

        return [sub[0] if self.frames == 1 else saturate_array(average_array(
            sub,
            axis = 0,
            trim = self.trimming,
            power = self.preference + 1.0,
            weights = np.roll(make_eased_weight_array(self.frames, self.easing), self.last_index),
        )) for sub in self.buffer]


class DetailingModule(PipelineModule):
    id = "detailing"
    name = "Detailing"
    icon = "\U0001f9ec"

    scale: float = ui_param("Scale", gr.Slider, minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0)
    sampler: str = ui_param("Sampling method", gr.Dropdown, choices = visible_sampler_names(), value = "Euler a")
    steps: int = ui_param("Steps", gr.Slider, precision = 0, minimum = 1, maximum = 150, step = 1, value = 15)
    denoising_strength: float = ui_param("Denoising strength", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := process_images(
            copy_with_overrides(session.processing,
                sampler_name = self.sampler,
                steps = self.steps,
                width = quantize(session.processing.width * self.scale, 8),
                height = quantize(session.processing.height * self.scale, 8),
                denoising_strength = self.denoising_strength,
                seed_enable_extras = True,
                seed_resize_from_w = session.processing.seed_resize_from_w or session.processing.width,
                seed_resize_from_h = session.processing.seed_resize_from_h or session.processing.height,
                do_not_save_samples = True,
                do_not_save_grid = True,
            ),
            [(np_to_pil(x), seed + i, 1) for i, x in enumerate(images)],
            global_options.processing.pixels_per_batch,
            self.preview and not global_options.live_preview.show_only_finished_images,
        )):
            return None

        return [
            pil_to_np(ensure_image_dims(image_array[0], "RGB", (session.processing.width, session.processing.height)))
            for image_array in processed_images
        ]


class InterpolationModule(PipelineModule):
    id = "interpolation"
    name = "Interpolation"
    icon = "\U0001f553"

    rate: float = ui_param("Rate", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0)

    buffer: NDArray[np.float_] = field(factory = lambda: np.empty((0,)))

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer.shape[0] == 0:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (session.processing.width, session.processing.height))
                for image in images
            ], 0)

        for sub, image in zip(self.buffer, images):
            sub[:] = lerp(sub, match_image(image, sub), self.rate)

        return [sub for sub in self.buffer]


class LimitingModule(PipelineModule):
    id = "limiting"
    name = "Limiting"
    icon = "\U0001f553"

    mode: str = ui_param("Mode", gr.Dropdown, choices = ["clamp", "compress"], value = "clamp")
    max_difference: float = ui_param("Maximum difference", gr.Slider, minimum = 0.001, maximum = 1.0, step = 0.001, value = 1.0)

    buffer: NDArray[np.float_] = field(factory = lambda: np.empty((0,)))

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer.shape[0] == 0:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (session.processing.width, session.processing.height))
                for image in images
            ], 0)

        for sub, image in zip(self.buffer, images):
            a = sub
            b = match_image(image, sub)
            diff = b - a

            if self.mode == "clamp":
                np.clip(diff, -self.max_difference, self.max_difference, out = diff)
            elif self.mode == "compress":
                diff_range = np.abs(diff.max() - diff.min())
                max_diff_range = self.max_difference * 2.0

                if diff_range > max_diff_range:
                    diff *= max_diff_range / diff_range

            sub[:] = saturate_array(a + diff)

        return [sub for sub in self.buffer]


class MeasuringModule(PipelineModule):
    id = "measuring"
    name = "Measuring"
    icon = "\U0001f6e0"

    plot_every_nth_frame: int = ui_param("Plot every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 10)

    metrics: Metrics = field(factory = Metrics)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        self.metrics.measure(images[0])

        if frame_index % self.plot_every_nth_frame == 0:
            self.metrics.plot_to_directory(ensure_directory_exists(Path(global_options.output.output_dir) / session.output.project_subdir / "metrics"))

        return images


class ProcessingModule(PipelineModule):
    id = "processing"
    name = "Processing"
    icon = "\U0001f9ec"

    samples: int = ui_param("Sample count", gr.Number, precision = 0, minimum = 1, value = 1)
    trimming: float = ui_param("Trimming", gr.Slider, minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0)
    easing: float = ui_param("Easing", gr.Slider, minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)
    preference: float = ui_param("Preference", gr.Slider, minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := process_images(
            copy_with_overrides(session.processing, do_not_save_samples = True, do_not_save_grid = True),
            [(np_to_pil(x), seed + i, self.samples) for i, x in enumerate(images)],
            global_options.processing.pixels_per_batch,
            self.preview and not global_options.live_preview.show_only_finished_images,
        )):
            return None

        return [
            pil_to_np(image_array[0]) if len(image_array) == 1 else saturate_array(average_array(
                np.stack([pil_to_np(x) for x in image_array]),
                axis = 0,
                trim = self.trimming,
                power = self.preference + 1.0,
                weights = np.flip(make_eased_weight_array(len(image_array), self.easing)),
            ))
            for image_array in processed_images
        ]


class SavingModule(PipelineModule):
    id = "saving"
    name = "Saving"
    icon = "\U0001f6e0"

    scale: float = ui_param("Scale", gr.Slider, minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0)
    save_every_nth_frame: int = ui_param("Save every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 1)
    save_final: bool = ui_param("Save final", gr.Checkbox, value = False)
    archive_mode: bool = ui_param("Archive mode", gr.Checkbox, value = False)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if frame_index % self.save_every_nth_frame != 0:
            return images

        for i, image in enumerate(self._get_scaled_images(images, session)):
            file_name = f"{frame_index:05d}"

            if len(images) > 1:
                file_name += f"-{i + 1:02d}"

            save_processed_image(
                image = np_to_pil(image),
                p = session.processing,
                processed = Processed(session.processing, []),
                output_dir = ensure_directory_exists(Path(global_options.output.output_dir) / session.output.project_subdir),
                file_name = file_name,
                archive_mode = self.archive_mode,
            )

        return images

    def finalize(self, images: list[NumpyImage], session: Session) -> None:
        if self.save_final:
            for image in self._get_scaled_images(images, session):
                save_processed_image(
                    image = np_to_pil(image),
                    p = session.processing,
                    processed = Processed(session.processing, []),
                    output_dir = ensure_directory_exists(Path(global_options.output.output_dir)),
                    file_name = None,
                    archive_mode = self.archive_mode,
                )

        wait_until(lambda: not image_save_queue.busy)

    def _get_scaled_images(self, images: list[NumpyImage], session: Session) -> list[NumpyImage]:
        return [ensure_image_dims(x, size = (
            int(quantize(session.processing.width * self.scale, 8)),
            int(quantize(session.processing.height * self.scale, 8)),
        )) for x in images] if self.scale != 1.0 else images


class VideoRenderingModule(PipelineModule):
    id = "video_rendering"
    name = "Video rendering"
    icon = "\U0001f6e0"

    render_draft_every_nth_frame: int = ui_param("Render draft every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 100)
    render_final_every_nth_frame: int = ui_param("Render final every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 1000)
    render_draft_on_finish: bool = ui_param("Render draft on finish", gr.Checkbox, value = False)
    render_final_on_finish: bool = ui_param("Render final on finish", gr.Checkbox, value = False)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        for i, _ in enumerate(images, 1):
            if frame_index % self.render_draft_every_nth_frame == 0:
                render_project_video(Path(global_options.output.output_dir) / session.output.project_subdir, session.video_renderer, False, i)

            if frame_index % self.render_final_every_nth_frame == 0:
                render_project_video(Path(global_options.output.output_dir) / session.output.project_subdir, session.video_renderer, True, i)

        return images

    def finalize(self, images: list[NumpyImage], session: Session) -> None:
        for i, _ in enumerate(images, 1):
            if self.render_draft_on_finish:
                render_project_video(Path(global_options.output.output_dir) / session.output.project_subdir, session.video_renderer, False, i)

            if self.render_final_on_finish:
                render_project_video(Path(global_options.output.output_dir) / session.output.project_subdir, session.video_renderer, True, i)

        wait_until(lambda: not video_render_queue.busy)
