from math import floor
from pathlib import Path
from typing import Any, Optional

import requests

from temporal.backend import Backend
from temporal.general_data import GeneralData
from temporal.processing_params import ProcessingParams
from temporal.thread_queue import ThreadQueue
from temporal.utils.image import NumpyImage, base64_to_image, image_to_base64, np_to_pil, save_image


class WebUIAPIBackend(Backend):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.image_save_queue = ThreadQueue()
        self._preview_image = None

    @property
    def url(self):
        return f"{self.host}:{self.port}"

    def list_models(self) -> list[str]:
        if (r := requests.get(f"{self.url}/sdapi/v1/sd-models")).ok:
            return [x["model_name"] for x in r.json()]
        else:
            raise requests.RequestException(response = r)

    def list_vaes(self) -> list[str]:
        if (r := requests.get(f"{self.url}/sdapi/v1/sd-vae")).ok:
            return ["Automatic", "None"] + [x["model_name"] for x in r.json()]
        else:
            raise requests.RequestException(response = r)

    def list_upscalers(self) -> list[str]:
        if (r := requests.get(f"{self.url}/sdapi/v1/upscalers")).ok:
            return [x["name"] for x in r.json()]
        else:
            raise requests.RequestException(response = r)

    def list_samplers(self) -> list[str]:
        if (r := requests.get(f"{self.url}/sdapi/v1/samplers")).ok:
            return [x["name"] for x in r.json()]
        else:
            raise requests.RequestException(response = r)

    def list_schedulers(self) -> list[str]:
        if (r := requests.get(f"{self.url}/sdapi/v1/schedulers")).ok:
            return [x["label"] for x in r.json()]
        else:
            raise requests.RequestException(response = r)

    def image_to_image(self, images: list[NumpyImage], params: ProcessingParams, width: int, height: int, preview: bool = False) -> Optional[list[NumpyImage]]:
        settings: dict[str, Any] = {
            "samples_format": "png",
            "save_to_dirs": False,
            "sd_model_checkpoint": params.model,
            "CLIP_stop_at_last_layers": params.clip_skip,
        }

        if params.vae:
            settings["sd_vae"] = params.vae

        if not preview:
            settings["show_progress_every_n_steps"] = -1

        if (r := requests.post(f"{self.url}/sdapi/v1/img2img", json = {
            "init_images": [image_to_base64(x, "fast") for x in images],
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
            "batch_size": len(images),
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "override_settings": settings,
            "override_settings_restore_afterwards": False,
        })).ok:
            return [base64_to_image(x) for x in r.json()["images"][:len(images)]]
        else:
            raise requests.RequestException(response = r)

    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
        if (r := requests.post(f"{self.url}/sdapi/v1/extra-single-image", json = {
            "image": image_to_base64(image, "fast"),
            "resize_mode": 0,
            "upscaling_resize_w": floor(image.shape[1] * scale),
            "upscaling_resize_h": floor(image.shape[0] * scale),
            "upscaler_1": upscaler,
        })).ok:
            return base64_to_image(r.json()["image"])
        else:
            raise requests.RequestException(response = r)

    def get_preview(self) -> Optional[NumpyImage]:
        return self._preview_image

    def set_preview(self, image: Optional[NumpyImage] = None) -> None:
        self._preview_image = image

    def save_image(self, image: NumpyImage, general: GeneralData, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
        if not file_name:
            return

        self.image_save_queue.enqueue(
            save_image,
            np_to_pil(image),
            (output_dir / file_name).with_suffix(".png"),
            archive_mode = archive_mode,
        )

    def are_images_saved(self) -> bool:
        return not self.image_save_queue.busy

    def interrupt(self) -> None:
        if not (r := requests.post(f"{self.url}/sdapi/v1/interrupt")).ok:
            raise requests.RequestException(response = r)

    def is_interrupted(self) -> bool:
        if (r := requests.get(f"{self.url}/sdapi/v1/progress")).ok:
            data = r.json()
            return data["state"]["interrupted"] or data["state"]["skipped"]
        else:
            raise requests.RequestException(response = r)
