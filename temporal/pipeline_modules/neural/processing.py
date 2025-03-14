from math import floor

from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.neural import NeuralModule
from temporal.processing_params import ProcessingParams
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.math import quantize
from temporal.utils.object import copy_with_overrides
from temporal.utils.prompt import evaluate_prompt


class ProcessingModule(NeuralModule):
    name = "Processing"

    parameters: ProcessingParams = Param("Processing parameters", value = ProcessingParams)
    scale: float = Param("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        if (result := shared.backend.image_to_image(
            image,
            copy_with_overrides(self.parameters,
                positive_prompt = evaluate_prompt(self.parameters.positive_prompt, frame_index - 1, seed),
                negative_prompt = evaluate_prompt(self.parameters.negative_prompt, frame_index - 1, seed),
                seed = seed,
            ),
            int(quantize(floor(general.image_size.x * self.scale), 8)),
            int(quantize(floor(general.image_size.y * self.scale), 8)),
        )) is not None:
            return ensure_image_dims(result, (general.image_size.x, general.image_size.y), 3)
        else:
            logging.warning("Couldn't process an image for some reason")
            return image
