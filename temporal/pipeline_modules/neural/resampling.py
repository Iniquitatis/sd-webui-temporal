from math import floor
from typing import Optional

from temporal.meta.configurable import EnumParam, FloatParam
from temporal.pipeline_modules.neural import NeuralModule
from temporal.project import Project
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, PILImage, ensure_image_dims, np_to_pil, pil_to_np
from temporal.web_ui import get_upscalers, upscale_image


class ResamplingModule(NeuralModule):
    name = "Resampling"

    upscaler: str = EnumParam("Upscaler", choices = get_upscalers(), value = get_first_element(get_upscalers()), ui_type = "menu")
    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        def resample(im: PILImage):
            scale = self.scale

            if scale < 1.0:
                scale = 1.0 / scale
                im = ensure_image_dims(im, size = (floor(project.processing.width / scale), floor(project.processing.height / scale)))

            return upscale_image(im, self.upscaler, scale)

        return [
            pil_to_np(ensure_image_dims(resample(np_to_pil(im)), "RGB", (project.processing.width, project.processing.height)))
            for im in images
        ]
