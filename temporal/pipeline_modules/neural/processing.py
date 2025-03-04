from math import floor

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.neural import NeuralModule
from temporal.processing_params import ProcessingParams
from temporal.shared import shared
from temporal.utils.image import NumpyImage
from temporal.utils.logging import warning
from temporal.utils.math import quantize
from temporal.utils.object import copy_with_overrides
from temporal.utils.prompt import evaluate_prompt


class ProcessingModule(NeuralModule):
    name = "Processing"

    parameters: ProcessingParams = Param("Processing parameters", factory = ProcessingParams)
    scale: float = Param("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def process(self, npim: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        if (result := shared.backend.image_to_image(
            npim,
            copy_with_overrides(self.parameters,
                positive_prompt = evaluate_prompt(self.parameters.positive_prompt, frame_index - 1, seed),
                negative_prompt = evaluate_prompt(self.parameters.negative_prompt, frame_index - 1, seed),
                seed = seed,
            ),
            int(quantize(floor(general.image_size.x * self.scale), 8)),
            int(quantize(floor(general.image_size.y * self.scale), 8)),
            shared.previewed_modules[self.uuid] and not shared.options.live_preview.show_only_finished_images,
        )) is not None:
            return result
        else:
            warning("Couldn't process an image for some reason")
            return npim
