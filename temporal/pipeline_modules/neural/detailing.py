from typing import Optional

from modules.sd_samplers import visible_sampler_names

from temporal.meta.configurable import EnumParam, FloatParam, IntParam
from temporal.pipeline_modules.neural import NeuralModule
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.image import NumpyImage, ensure_image_dims, np_to_pil, pil_to_np
from temporal.utils.math import quantize
from temporal.utils.object import copy_with_overrides
from temporal.web_ui import get_schedulers, has_schedulers, process_images


class DetailingModule(NeuralModule):
    id = "detailing"
    name = "Detailing"

    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")
    sampler: str = EnumParam("Sampling method", choices = visible_sampler_names(), value = "Euler a", ui_type = "menu")
    if has_schedulers():
        scheduler: str = EnumParam("Schedule type", choices = get_schedulers(), value = "Automatic", ui_type = "menu")
    steps: int = IntParam("Steps", minimum = 1, maximum = 150, step = 1, value = 15, ui_type = "slider")
    denoising_strength: float = FloatParam("Denoising strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2, ui_type = "slider")

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := process_images(
            copy_with_overrides(project.processing,
                sampler_name = self.sampler,
                steps = self.steps,
                width = quantize(project.processing.width * self.scale, 8),
                height = quantize(project.processing.height * self.scale, 8),
                denoising_strength = self.denoising_strength,
                seed_enable_extras = True,
                seed_resize_from_w = project.processing.seed_resize_from_w or project.processing.width,
                seed_resize_from_h = project.processing.seed_resize_from_h or project.processing.height,
                do_not_save_samples = True,
                do_not_save_grid = True,
                **({"scheduler" : self.scheduler} if hasattr(self, "scheduler") else {})
            ),
            [(np_to_pil(x), seed + i, 1) for i, x in enumerate(images)],
            shared.options.processing.pixels_per_batch,
            shared.previewed_modules[self.id] and not shared.options.live_preview.show_only_finished_images,
        )):
            return None

        return [
            pil_to_np(ensure_image_dims(image_array[0], "RGB", (project.processing.width, project.processing.height)))
            for image_array in processed_images
        ]
