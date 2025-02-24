from io import BytesIO
from math import floor
from pathlib import Path
from time import sleep
from typing import Any, Literal, Optional

import requests
from PIL import Image

from temporal.backend import Backend
from temporal.general_data import GeneralData
from temporal.processing_params import ProcessingParams
from temporal.thread_queue import ThreadQueue
from temporal.utils.image import NumpyImage, np_to_pil, pil_to_np, save_image


class ComfyUIAPIBackend(Backend):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.image_save_queue = ThreadQueue()
        self._preview_image = None
        self._interrupted = True

    @property
    def url(self):
        return f"{self.host}:{self.port}"

    def list_models(self) -> list[str]:
        return _safe_request("GET", f"{self.url}/models/checkpoints").json()

    def list_vaes(self) -> list[str]:
        return ["Automatic"] + _safe_request("GET", f"{self.url}/models/vae").json()

    def list_upscalers(self) -> list[str]:
        return _safe_request("GET", f"{self.url}/models/upscale_models").json()

    def list_samplers(self) -> list[str]:
        return _safe_request("GET", f"{self.url}/object_info").json()["KSampler"]["input"]["required"]["sampler_name"][0]

    def list_schedulers(self) -> list[str]:
        return _safe_request("GET", f"{self.url}/object_info").json()["KSampler"]["input"]["required"]["scheduler"][0]

    def image_to_image(self, images: list[NumpyImage], params: ProcessingParams, width: int, height: int, preview: bool = False) -> Optional[list[NumpyImage]]:
        self._interrupted = False

        self._clear_queue()

        result = []

        is_vae_defined = params.vae and params.vae != "Automatic"

        vae_loader = {
            "vae_loader": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": params.vae,
                },
            },
        } if is_vae_defined else {}

        for i, image in enumerate(images):
            processed = self._get_image(self._prompt(vae_loader | {
                "checkpoint_loader": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {
                        "ckpt_name": params.model,
                    },
                },
                "sampler": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["checkpoint_loader", 0],
                        "positive": ["positive_encoder", 0],
                        "negative": ["negative_encoder", 0],
                        "latent_image": ["latent_image", 0],
                        "seed": params.seed,
                        "steps": params.steps,
                        "cfg": params.cfg,
                        "sampler_name": params.sampler,
                        "scheduler": params.scheduler,
                        "denoise": params.strength,
                    },
                },
                "image": {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": self._upload_image(f"_temporal_{i}.png", image),
                    },
                },
                "latent_image": {
                    "class_type": "VAEEncode",
                    "inputs": {
                        "pixels": ["image", 0],
                        "vae": ["vae_loader", 0] if is_vae_defined else ["checkpoint_loader", 2],
                    },
                },
                "clip_skipper": {
                    "class_type": "CLIPSetLastLayer",
                    "inputs": {
                        "clip": ["checkpoint_loader", 1],
                        "stop_at_clip_layer": -params.clip_skip,
                    },
                },
                "positive_encoder": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["clip_skipper", 0] if params.clip_skip > 1 else ["checkpoint_loader", 1],
                        "text": params.positive_prompt,
                    },
                },
                "negative_encoder": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["clip_skipper", 0] if params.clip_skip > 1 else ["checkpoint_loader", 1],
                        "text": params.negative_prompt,
                    },
                },
                "vae_decoder": {
                    "class_type": "VAEDecode",
                    "inputs": {
                        "samples": ["sampler", 0],
                        "vae": ["vae_loader", 0] if is_vae_defined else ["checkpoint_loader", 2],
                    },
                },
                "image_saver": {
                    "class_type": "PreviewImage",
                    "inputs": {
                        "filename_prefix": "_temporal",
                        "images": ["vae_decoder", 0],
                    },
                },
            }))

            if processed is None:
                return

            result.append(processed)

        return result

    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
        self._interrupted = False

        self._clear_queue()

        return self._get_image(self._prompt({
            "model_loader": {
                "class_type": "UpscaleModelLoader",
                "inputs": {
                    "model_name": upscaler,
                },
            },
            "image_loader": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": self._upload_image(f"_temporal_upscale.png", image),
                },
            },
            "upscaler": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["model_loader", 0],
                    "image": ["image_loader", 0],
                },
            },
            "final_scaler": {
                "class_type": "ImageScale",
                "inputs": {
                    "upscale_method": "lanczos",
                    "width": floor(image.shape[1] * scale),
                    "height": floor(image.shape[0] * scale),
                    "crop": "disabled",
                    "image": ["upscaler", 0],
                },
            },
            "image_saver": {
                "class_type": "PreviewImage",
                "inputs": {
                    "filename_prefix": "_temporal",
                    "images": ["final_scaler", 0],
                },
            },
        }))

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
        _safe_request("POST", f"{self.url}/interrupt")

        self._interrupted = True

    def is_interrupted(self) -> bool:
        return self._interrupted

    def _clear_queue(self) -> None:
        _safe_request("POST", f"{self.url}/queue", json = {"clear": "true"})

    def _get_image(self, prompt_id: str) -> Optional[NumpyImage]:
        while True:
            if history_data := _safe_request("GET", f"{self.url}/history/{prompt_id}").json():
                break

            sleep(1.0)

        if history_data[prompt_id]["status"]["status_str"] != "success":
            return None

        with BytesIO(_safe_request("GET", f"{self.url}/view", params = history_data[prompt_id]["outputs"]["image_saver"]["images"][0]).content) as stream:
            return pil_to_np(Image.open(stream))

    def _prompt(self, nodes: dict[str, Any]) -> str:
        return _safe_request("POST", f"{self.url}/prompt", json = {"prompt": nodes}).json()["prompt_id"]

    def _upload_image(self, file_name: str, image: NumpyImage) -> str:
        with BytesIO() as stream:
            np_to_pil(image).save(stream, "png")

            return _safe_request("POST", f"{self.url}/upload/image", files = {
                "image": (file_name, stream.getvalue(), "image/png"),
                "overwrite": "true",
            }).json()["name"]


def _safe_request(method: Literal["GET", "POST"], *args: Any, **kwargs: Any) -> requests.Response:
    if (r := getattr(requests, method.lower())(*args, **kwargs)).ok:
        return r
    else:
        raise requests.RequestException(response = r)
