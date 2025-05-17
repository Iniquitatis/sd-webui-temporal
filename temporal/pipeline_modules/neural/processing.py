from math import floor

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.neural import NeuralModule
from temporal.processing_params import ProcessingParams
from temporal.seed import Seed
from temporal.shared import shared
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.logging import log
from temporal.utils.math import quantize
from temporal.utils.prompt import evaluate_prompt


class ProcessingModule(NeuralModule):
    name = "Processing"

    model: str = Field(lambda: get_first_element(shared.backend.list_models(), ("", ""))[0], name = "Model", choices = lambda: dict(shared.backend.list_models()), display = "menu")
    vae: str = Field(lambda: get_first_element(shared.backend.list_vaes(), ("", ""))[0], name = "VAE", choices = lambda: dict(shared.backend.list_vaes()), display = "menu")
    clip_skip: int = Field(1, name = "CLIP skip", minimum = 1, maximum = 12, step = 1, display = "slider")
    positive_prompt: str = Field("", name = "Positive prompt", display = "area")
    negative_prompt: str = Field("", name = "Negative prompt", display = "area")
    sampler: str = Field(lambda: get_first_element(shared.backend.list_samplers(), ("", ""))[0], name = "Sampler", choices = lambda: dict(shared.backend.list_samplers()), display = "menu")
    scheduler: str = Field(lambda: get_first_element(shared.backend.list_schedulers(), ("", ""))[0], name = "Scheduler", choices = lambda: dict(shared.backend.list_schedulers()), display = "menu")
    steps: int = Field(20, name = "Steps", minimum = 1, maximum = 150, step = 1, display = "slider")
    cfg: float = Field(5.0, name = "CFG", minimum = 1.0, maximum = 30.0, step = 0.5, display = "slider")
    strength: float = Field(0.5, name = "Strength", minimum = 0.0, maximum = 1.0, step = 0.01, display = "slider")
    use_global_seed: bool = Field(True, name = "Use global seed")
    seed: Seed = Field(Seed, name = "Seed", dependencies = {"use_global_seed": False})
    scale: float = Field(1.0, name = "Scale", minimum = 0.25, maximum = 4.0, step = 0.25, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        seed = seed if self.use_global_seed else self.seed.fixed_value

        # FIXME: Should throw something like `BackendInterrupted`... maybe? But
        # using exceptions for control flow is far from the most obvious/
        # convenient thing to do.
        # Anyway, as of now, it _does_ return an image (an unchanged one),
        # making the engine "think" that the step was completed successfully.
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
                seed = seed,
            ),
            int(quantize(floor(general.image_size.x * self.scale), 8)),
            int(quantize(floor(general.image_size.y * self.scale), 8)),
        )) is not None:
            return ensure_image_dims(result, (general.image_size.x, general.image_size.y), 3)
        else:
            log.warning("Couldn't process an image for some reason")
            return image
