import mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/css", ".css")

from argparse import ArgumentParser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from temporal.api import register_api
from temporal.engine import Engine


parser = ArgumentParser()
parser.add_argument("--host", type = str, default = "0.0.0.0")
parser.add_argument("--port", type = int, default = 7870)
parser.add_argument("--backend", choices = ["comfyui", "sdwebui"], default = "sdwebui")
parser.add_argument("--backend-host", type = str, default = "")
parser.add_argument("--backend-port", type = int, default = 0)

args = parser.parse_args()

if args.backend == "comfyui":
    from temporal.backends.comfyui_api import ComfyUIAPIBackend

    backend = ComfyUIAPIBackend(args.backend_host or "http://127.0.0.1", args.backend_port or 8188)

elif args.backend == "sdwebui":
    from temporal.backends.webui_api import WebUIAPIBackend

    backend = WebUIAPIBackend(args.backend_host or "http://127.0.0.1", args.backend_port or 7860)

else:
    raise ValueError(f"Unknown backend {args.backend}")

app = FastAPI(title = "Temporal API")
register_api(app, Engine(backend, Path("settings"), Path("presets")))
app.mount("/", StaticFiles(directory = "ui", html = True), name = "static")

uvicorn.run(app, host = args.host, port = args.port)
