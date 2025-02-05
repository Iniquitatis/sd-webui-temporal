import mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/css", ".css")

from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from temporal.api import register_api
from temporal.backend import Backend
from temporal.engine import Engine


class StandaloneWebUIBackend(Backend):
    host = "http://127.0.0.1"
    port = "7862"
    url = f"{host}:{port}"

    def list_models(self):
        if (r := requests.get(f"{self.url}/sdapi/v1/sd-models")).ok:
            return [x["model_name"] for x in r.json()]

        return []

    def list_vaes(self):
        if (r := requests.get(f"{self.url}/sdapi/v1/sd-vae")).ok:
            return [x["model_name"] for x in r.json()]

        return []

    def list_upscalers(self):
        if (r := requests.get(f"{self.url}/sdapi/v1/upscalers")).ok:
            return [x["name"] for x in r.json()]

        return []

    def list_samplers(self):
        if (r := requests.get(f"{self.url}/sdapi/v1/samplers")).ok:
            return [x["name"] for x in r.json()]

        return []

    def list_schedulers(self):
        if (r := requests.get(f"{self.url}/sdapi/v1/schedulers")).ok:
            return [x["label"] for x in r.json()]

        return []

    def text_to_image(self, params, preview = False):
        return None

    def image_to_image(self, params, preview = False):
        return None

    def upscale_image(self, image, upscaler, scale):
        return None

    def set_preview(self, image = None):
        pass

    def save_image(self, image, project, output_dir, file_name = None, archive_mode = False):
        pass

    def are_images_saved(self):
        return True

    def is_interrupted(self):
        return False


engine = Engine(StandaloneWebUIBackend(), Path("settings"), Path("presets"))

app = FastAPI(title = "Temporal API")

register_api(app, engine)

app.mount("/", StaticFiles(directory = "ui", html = True), name = "static")


uvicorn.run(app, host = "127.0.0.1", port = 8087)
