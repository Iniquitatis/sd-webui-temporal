from dataclasses import dataclass, field
from functools import lru_cache
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Optional, Type, TypeVar

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from temporal.backend import Backend
from temporal.processing_params import ProcessingParams
from temporal.utils.image import NumpyImage, ensure_image_dims


class StandaloneBackend(Backend):
    def __init__(self, model_dir: Path, vae_dir: Path) -> None:
        self._model_dir = model_dir
        self._vae_dir = vae_dir
        self._cached_states: dict[tuple[str, Optional[str]], _PipelineState] = {}
        self._interrupted = False

    def list_models(self) -> Iterable[str]:
        return (x.name for x in self._model_dir.glob("*.safetensors"))

    def list_vaes(self) -> Iterable[str]:
        return chain(iter(["Automatic"]), (x.name for x in self._vae_dir.glob("*.safetensors")))

    def list_upscalers(self) -> Iterable[str]:
        return ["None"]

    def list_samplers(self) -> Iterable[str]:
        return _SCHEDULERS.keys()

    def list_schedulers(self) -> Iterable[str]:
        return _SCHEDULES.keys()

    def image_to_image(self, image: NumpyImage, params: ProcessingParams, width: int, height: int) -> Optional[NumpyImage]:
        self._interrupted = False

        state = self._get_pipeline_state(params.model, params.vae)

        positive_embeds, positive_pooled, negative_embeds, negative_pooled = state.make_embeds(params.positive_prompt, params.negative_prompt)

        state.pipeline.scheduler = state.make_scheduler(params.sampler, params.scheduler)

        def interrupt_callback(pipeline: Any, i: Any, t: Any, kwargs: Any) -> Any:
            if self._interrupted:
                pipeline._interrupt = True

            self._interrupted = False

            return kwargs

        return state.pipeline(
            image = ensure_image_dims(image, (width, height)),
            prompt_embeds = positive_embeds,
            pooled_prompt_embeds = positive_pooled,
            negative_prompt_embeds = negative_embeds,
            negative_pooled_prompt_embeds = negative_pooled,
            num_inference_steps = params.steps,
            guidance_scale = params.cfg,
            generator = torch.manual_seed(params.seed),
            strength = params.strength,
            num_images_per_prompt = 1,
            output_type = "np",
            callback_on_step_end = interrupt_callback,
        ).images[0]

    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
        return image

    def interrupt(self) -> None:
        self._interrupted = True

    def _get_pipeline_state(self, model_name: str, vae_name: Optional[str]) -> "_PipelineState":
        cache_key = model_name, vae_name

        if state := self._cached_states.get(cache_key, None):
            return state

        self._cached_states[cache_key] = state = _PipelineState(
            self._model_dir / model_name,
            self._vae_dir / vae_name if vae_name and vae_name != "Automatic" else None,
        )

        return state


_T = TypeVar("_T", bound = Any)


@dataclass
class _PipelineDef:
    cls: Type[Any]
    size_min: float
    size_max: float


@dataclass
class _SchedulerDef:
    name: str
    cls: Type[Any]
    args: dict[str, Any] = field(default_factory = dict)


@dataclass
class _ScheduleDef:
    name: str
    args: dict[str, Any] = field(default_factory = dict)


_GB = 2 ** 30


_PIPELINES: dict[str, _PipelineDef] = {
    "sd15": _PipelineDef(StableDiffusionImg2ImgPipeline, 1.9 * _GB, 2.5 * _GB),
    "sdxl": _PipelineDef(StableDiffusionXLImg2ImgPipeline, 6.0 * _GB, 8.0 * _GB),
}


_SCHEDULERS: dict[str, _SchedulerDef] = {
    "ddim": _SchedulerDef("DDIM", DDIMScheduler),
    "ddpm": _SchedulerDef("DDPM", DDPMScheduler),
    "deis": _SchedulerDef("DEIS", DEISMultistepScheduler),
    "dpm++": _SchedulerDef("DPM++", DPMSolverSinglestepScheduler),
    "dpm++_sde": _SchedulerDef("DPM++ SDE", DPMSolverSinglestepScheduler, dict(algorithm_type = "sde-dpmsolver++")),
    "dpm++_2m": _SchedulerDef("DPM++ 2M", DPMSolverMultistepScheduler),
    "dpm++_2m_sde": _SchedulerDef("DPM++ 2M SDE", DPMSolverMultistepScheduler, dict(algorithm_type = "sde-dpmsolver++")),
    "dpm2": _SchedulerDef("DPM2", KDPM2DiscreteScheduler),
    "dpm2_ancestral": _SchedulerDef("DPM2 Ancestral", KDPM2AncestralDiscreteScheduler),
    "euler": _SchedulerDef("Euler", EulerDiscreteScheduler),
    "euler_ancestral": _SchedulerDef("Euler Ancestral", EulerAncestralDiscreteScheduler),
    "heun": _SchedulerDef("Heun", HeunDiscreteScheduler),
    "lms": _SchedulerDef("LMS", LMSDiscreteScheduler),
    "unipc": _SchedulerDef("UniPC", UniPCMultistepScheduler),
}


_SCHEDULES: dict[str, _ScheduleDef] = {
    "uniform": _ScheduleDef("Uniform"),
    "beta": _ScheduleDef("Beta", dict(timestep_spacing = "linspace", use_beta_sigmas = True)),
    "exponential": _ScheduleDef("Exponential", dict(timestep_spacing = "linspace", use_exponential_sigmas = True)),
    "karras": _ScheduleDef("Karras", dict(use_karras_sigmas = True)),
    "simple": _ScheduleDef("Simple", dict(timestep_spacing = "trailing")),
}


class _PipelineState:
    def __init__(self, model_path: Path, vae_path: Optional[Path] = None) -> None:
        model_size = model_path.stat().st_size

        for model_type, pipeline_def in _PIPELINES.items():
            if pipeline_def.size_min <= model_size <= pipeline_def.size_max:
                break
        else:
            raise Exception(f"Couldn't detect type for {model_path} of size {model_size}")

        self.model_type = model_type

        self.pipeline = _load_model(
            pipeline_def.cls,
            model_path,
            **({"vae": _load_model(
                AutoencoderKL,
                vae_path,
            )} if vae_path is not None else {}),
        )
        # FIXME: For now, it's either-or: Diffusers has some bug that messes
        # with a compiled model and triggers an exception
        # NOTE: Requires triton
        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode = "reduce-overhead", fullgraph = True)
        # NOTE: Requires accelerate
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_xformers_memory_efficient_attention()

    @lru_cache(maxsize = 64)
    def make_embeds(self, positive_prompt: str, negative_prompt: str) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if self.pipeline is None:
            raise Exception

        if self.model_type == "sd15":
            compel = Compel(
                tokenizer = [
                    self.pipeline.tokenizer,
                ],
                text_encoder = [
                    self.pipeline.text_encoder,
                ],
                truncate_long_prompts = False,
            )

            positive_embeds = compel([positive_prompt])
            negative_embeds = compel([negative_prompt])

            [positive_embeds, negative_embeds] = compel.pad_conditioning_tensors_to_same_length([positive_embeds, negative_embeds])

            return positive_embeds, None, negative_embeds, None

        elif self.model_type == "sdxl":
            compel = Compel(
                tokenizer = [
                    self.pipeline.tokenizer,
                    self.pipeline.tokenizer_2,
                ],
                text_encoder = [
                    self.pipeline.text_encoder,
                    self.pipeline.text_encoder_2,
                ],
                truncate_long_prompts = False,
                returned_embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled = [False, True],
            )

            positive_embeds, positive_pooled = compel([positive_prompt])
            negative_embeds, negative_pooled = compel([negative_prompt])

            [positive_embeds, negative_embeds] = compel.pad_conditioning_tensors_to_same_length([positive_embeds, negative_embeds])

            return positive_embeds, positive_pooled, negative_embeds, negative_pooled

        else:
            raise NotImplementedError

    @lru_cache(maxsize = 64)
    def make_scheduler(self, scheduler_name: str, schedule_name: str) -> Any:
        if self.pipeline is None:
            raise Exception

        scheduler_def = _SCHEDULERS[scheduler_name]
        schedule_def = _SCHEDULES[schedule_name]

        params = signature(scheduler_def.cls.__init__).parameters

        return scheduler_def.cls.from_config(self.pipeline.scheduler.config, **{
            k: v
            for k, v in chain(scheduler_def.args.items(), schedule_def.args.items())
            if k in params
        })


def _load_model(cls: Type[_T], path: Path, **kwargs: Any) -> _T:
    return cls.from_single_file(
        path.as_posix(),
        torch_dtype = torch.float16,
        variant = "fp16",
        use_safetensors = True,
        # FIXME: Not sure if it should be enabled by default
        local_files_only = False,
        **kwargs,
    ).to("cuda")
