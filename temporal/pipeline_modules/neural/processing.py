from math import floor
from typing import Optional

from temporal.general_data import GeneralData
from temporal.meta.configurable import FloatParam, ProcessingParamsParam
from temporal.pipeline_modules.neural import NeuralModule
from temporal.processing_params import ProcessingParams
from temporal.shared import shared
from temporal.utils.image import NumpyImage
from temporal.utils.math import quantize
from temporal.utils.object import copy_with_overrides
from temporal.utils.prompt import evaluate_prompt


class ProcessingModule(NeuralModule):
    name = "Processing"

    parameters: ProcessingParams = ProcessingParamsParam("Processing parameters")
    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def forward(self, images: list[NumpyImage], general: GeneralData, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if processed_images := shared.backend.image_to_image_batched(
            images,
            copy_with_overrides(self.parameters,
                positive_prompt = evaluate_prompt(self.parameters.positive_prompt, frame_index - 1, seed),
                negative_prompt = evaluate_prompt(self.parameters.negative_prompt, frame_index - 1, seed),
                seed = seed,
            ),
            int(quantize(floor(general.image_size.x * self.scale), 8)),
            int(quantize(floor(general.image_size.y * self.scale), 8)),
            shared.options.processing.pixels_per_batch,
            shared.previewed_modules[self.id] and not shared.options.live_preview.show_only_finished_images,
        ):
            return processed_images
