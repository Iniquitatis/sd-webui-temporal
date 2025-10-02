from math import floor

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.neural import NeuralModule
from modules.pipeline_state import PipelineResult, PipelineState
from modules.processing_params import ProcessingParams
from modules.seed import Seed
from modules.shared import shared
from modules.utils.collection import get_first_element
from modules.utils.image import NumpyImage, ensure_image_dims
from modules.utils.math import quantize
from modules.utils.prompt import evaluate_prompt


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
    advance_seed: bool = Field(False, name = "Advance seed")
    scale: float = Field(1.0, name = "Scale", minimum = 0.25, maximum = 4.0, step = 0.25, display = "slider")
    iteration: int = Field(0, flags = {"runtime"})

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        seed = (general.seed if self.use_global_seed else self.seed).fixed_value

        if self.advance_seed:
            seed += self.iteration

        if (result := shared.backend.image_to_image(
            image,
            ProcessingParams(
                model = self.model,
                vae = self.vae,
                clip_skip = self.clip_skip,
                positive_prompt = evaluate_prompt(self.positive_prompt, self.iteration, seed),
                negative_prompt = evaluate_prompt(self.negative_prompt, self.iteration, seed),
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
            self.iteration += 1

            yield PipelineState.finish(image = self._blend(image, ensure_image_dims(result, (general.image_size.x, general.image_size.y), 3)), preview = self.preview)

        else:
            yield PipelineState.fail()

    def interrupt(self, general: GeneralData) -> None:
        shared.backend.interrupt()
