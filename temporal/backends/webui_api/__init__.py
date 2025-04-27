from math import floor
from typing import Any, Literal, Optional

import requests

from temporal.backend import Backend
from temporal.processing_params import ProcessingParams
from temporal.utils.image import NumpyImage, base64_to_image, image_to_base64


class WebUIAPIBackend(Backend):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    @property
    def url(self):
        return f"{self.host}:{self.port}"

    def list_models(self) -> list[str]:
        _safe_request("POST", f"{self.url}/sdapi/v1/refresh-checkpoints")

        return [x["model_name"] for x in _safe_request("GET", f"{self.url}/sdapi/v1/sd-models").json()]

    def list_vaes(self) -> list[str]:
        _safe_request("POST", f"{self.url}/sdapi/v1/refresh-vae")

        return ["Automatic", "None"] + [x["model_name"] for x in _safe_request("GET", f"{self.url}/sdapi/v1/sd-vae").json()]

    def list_upscalers(self) -> list[str]:
        return [x["name"] for x in _safe_request("GET", f"{self.url}/sdapi/v1/upscalers").json()]

    def list_samplers(self) -> list[str]:
        return [x["name"] for x in _safe_request("GET", f"{self.url}/sdapi/v1/samplers").json()]

    def list_schedulers(self) -> list[str]:
        return [x["label"] for x in _safe_request("GET", f"{self.url}/sdapi/v1/schedulers").json()]

    def image_to_image(self, image: NumpyImage, params: ProcessingParams, width: int, height: int) -> Optional[NumpyImage]:
        if (result := _safe_request("POST", f"{self.url}/sdapi/v1/img2img", json = {
            "init_images": [image_to_base64(image, False, "fast")],
            "prompt": params.positive_prompt,
            "negative_prompt": params.negative_prompt,
            "width": width,
            "height": height,
            "sampler_name": params.sampler,
            "scheduler": params.scheduler,
            "steps": params.steps,
            "cfg_scale": params.cfg,
            "denoising_strength": params.strength,
            "seed": params.seed,
            "n_iter": 1,
            "batch_size": 1,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "override_settings": {
                "samples_format": "png",
                "save_to_dirs": False,
                "sd_model_checkpoint": params.model,
                **({"sd_vae": params.vae} if params.vae else {}),
                "CLIP_stop_at_last_layers": params.clip_skip,
                "show_progress_every_n_steps": -1,
            },
            "override_settings_restore_afterwards": False,
        }).json()) and not self._is_interrupted():
            return base64_to_image(result["images"][0], False)

    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
        if (result := _safe_request("POST", f"{self.url}/sdapi/v1/extra-single-image", json = {
            "image": image_to_base64(image, False, "fast"),
            "resize_mode": 0,
            "upscaling_resize_w": floor(image.shape[1] * scale),
            "upscaling_resize_h": floor(image.shape[0] * scale),
            "upscaler_1": upscaler,
        }).json()) and not self._is_interrupted():
            return base64_to_image(result["image"], False)

    def interrupt(self) -> None:
        _safe_request("POST", f"{self.url}/sdapi/v1/interrupt")

    def _is_interrupted(self) -> bool:
        return _safe_request("GET", f"{self.url}/sdapi/v1/progress").json()["state"]["interrupted"]


def _safe_request(method: Literal["GET", "POST"], *args: Any, **kwargs: Any) -> requests.Response:
    if (r := getattr(requests, method.lower())(*args, **kwargs)).ok:
        return r
    else:
        raise requests.RequestException(response = r)
