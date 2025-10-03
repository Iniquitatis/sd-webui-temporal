import json
from io import BytesIO
from math import floor
from typing import Any, Iterator, Literal, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import requests
from PIL import Image
from websocket import WebSocket

from modules.backend import Backend
from modules.processing_params import ProcessingParams
from modules.utils.image import NumpyImage, ensure_image_dims, np_to_pil, pil_to_np


class ComfyUIAPIBackend(Backend):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.client_id = str(uuid4())

    @property
    def url(self):
        return f"{self.host}:{self.port}"

    def list_models(self) -> Iterator[tuple[str, str]]:
        for name in _safe_request("GET", f"{self.url}/models/checkpoints").json():
            yield name, name

    def list_vaes(self) -> Iterator[tuple[str, str]]:
        yield "auto", "Automatic"

        for name in _safe_request("GET", f"{self.url}/models/vae").json():
            yield name, name

    def list_upscalers(self) -> Iterator[tuple[str, str]]:
        for name in _safe_request("GET", f"{self.url}/models/upscale_models").json():
            yield name, name

    def list_samplers(self) -> Iterator[tuple[str, str]]:
        for name in _safe_request("GET", f"{self.url}/object_info").json()["KSampler"]["input"]["required"]["sampler_name"][0]:
            yield name, name

    def list_schedulers(self) -> Iterator[tuple[str, str]]:
        for name in _safe_request("GET", f"{self.url}/object_info").json()["KSampler"]["input"]["required"]["scheduler"][0]:
            yield name, name

    def image_to_image(self, image: NumpyImage, params: ProcessingParams, width: int, height: int) -> Optional[NumpyImage]:
        self._clear_queue()

        is_vae_defined = params.vae and params.vae != "auto"

        return self._get_image(self._prompt({
            "checkpoint_loader": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": params.model,
                },
            },
            **({"vae_loader": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": params.vae,
                },
            }} if is_vae_defined else {}),
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
                    "image": self._upload_image("_temporal_image_to_image.png", ensure_image_dims(image, (width, height))),
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
                "class_type": "SaveImageWebsocket",
                "inputs": {
                    "images": ["vae_decoder", 0],
                },
            },
        }), "image_saver")

    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
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
                "class_type": "SaveImageWebsocket",
                "inputs": {
                    "images": ["final_scaler", 0],
                },
            },
        }), "image_saver")

    def interrupt(self) -> None:
        _safe_request("POST", f"{self.url}/interrupt")

    def _clear_queue(self) -> None:
        _safe_request("POST", f"{self.url}/queue", json = {"clear": "true"})

    def _get_image(self, prompt_id: str, target_node_id: str) -> Optional[NumpyImage]:
        result = None

        url = urlunsplit(urlsplit(self.url)._replace(scheme = "ws"))

        ws = WebSocket()
        ws.connect(f"{url}/ws?clientId={self.client_id}")

        current_node = ""

        while True:
            if isinstance(chunk := ws.recv(), str):
                message = json.loads(chunk)

                if message["type"] != "executing":
                    continue

                data = message["data"]

                if data["prompt_id"] != prompt_id:
                    continue

                if data["node"] is None:
                    break

                current_node = data["node"]

            elif current_node == target_node_id:
                with BytesIO(chunk[8:]) as stream:
                    result = pil_to_np(Image.open(stream))

        ws.close()

        return result

    def _prompt(self, nodes: dict[str, Any]) -> str:
        return _safe_request("POST", f"{self.url}/prompt", json = {"client_id": self.client_id, "prompt": nodes}).json()["prompt_id"]

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
