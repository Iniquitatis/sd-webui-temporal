from math import floor
from random import randint
from typing import Any

from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.neural import NeuralModule
from temporal.processing_params import ProcessingParams
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.math import quantize
from temporal.utils.prompt import evaluate_prompt


class ProcessingModule(NeuralModule):
    name = "Processing"

    model: str = Param("Model", choices = lambda: list(shared.backend.list_models()), value = lambda: get_first_element(shared.backend.list_models(), ""), ui_type = "menu")
    vae: str = Param("VAE", choices = lambda: list(shared.backend.list_vaes()), value = lambda: get_first_element(shared.backend.list_vaes(), ""), ui_type = "menu")
    clip_skip: int = Param("CLIP skip", minimum = 1, maximum = 12, step = 1, value = 1, ui_type = "slider")
    positive_prompt: str = Param("Positive prompt", value = "", ui_type = "area")
    negative_prompt: str = Param("Negative prompt", value = "", ui_type = "area")
    sampler: str = Param("Sampler", choices = lambda: list(shared.backend.list_samplers()), value = lambda: get_first_element(shared.backend.list_samplers(), ""), ui_type = "menu")
    scheduler: str = Param("Scheduler", choices = lambda: list(shared.backend.list_schedulers()), value = lambda: get_first_element(shared.backend.list_schedulers(), ""), ui_type = "menu")
    steps: int = Param("Steps", minimum = 1, maximum = 150, step = 1, value = 20, ui_type = "slider")
    cfg: float = Param("CFG", minimum = 1.0, maximum = 30.0, step = 0.5, value = 5.0, ui_type = "slider")
    strength: float = Param("Strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5, ui_type = "slider")
    use_global_seed: bool = Param("Use global seed", value = True)
    seed: int = Param("Seed", value = -1, dependencies = {"use_global_seed": False}, ui_type = "seed")
    scale: float = Param("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.seed == -1:
            self.seed = randint(0, 0x7fffffff)

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        if (result := shared.backend.image_to_image(
            image,
            ProcessingParams(
                model = self.model,
                vae = self.vae,
                clip_skip = self.clip_skip,
                positive_prompt = evaluate_prompt(self.positive_prompt, iter_index - 1, seed),
                negative_prompt = evaluate_prompt(self.negative_prompt, iter_index - 1, seed),
                sampler = self.sampler,
                scheduler = self.scheduler,
                steps = self.steps,
                cfg = self.cfg,
                strength = self.strength,
                seed = seed if self.use_global_seed else self.seed,
            ),
            int(quantize(floor(general.image_size.x * self.scale), 8)),
            int(quantize(floor(general.image_size.y * self.scale), 8)),
        )) is not None:
            return ensure_image_dims(result, (general.image_size.x, general.image_size.y), 3)
        else:
            logging.warning("Couldn't process an image for some reason")
            return image
