import mimetypes
from argparse import ArgumentParser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from temporal.api import register_api
from temporal.shared import shared
from temporal.utils.logging import LogLevel, log


mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/css", ".css")

parser = ArgumentParser()
parser.add_argument("--host", type = str, default = "0.0.0.0")
parser.add_argument("--port", type = int, default = 7870)
parser.add_argument("--settings-dir", type = Path, default = "settings")
parser.add_argument("--backend", choices = ["comfyui", "sdwebui", "standalone"], default = "standalone")
parser.add_argument("--backend-host", type = str, default = "")
parser.add_argument("--backend-port", type = int, default = 0)
parser.add_argument("--model-dir", type = Path, default = ".")
parser.add_argument("--vae-dir", type = Path, default = ".")
parser.add_argument("--log-level", type = str, choices = list(x.name.lower() for x in LogLevel), default = log.level.name.lower())
parser.add_argument("--log-path", type = Path)

args = parser.parse_args()

log.level = getattr(LogLevel, args.log_level.upper())
log.path = args.log_path

if args.backend == "comfyui":
    from temporal.backends.comfyui_api import ComfyUIAPIBackend

    backend = ComfyUIAPIBackend(args.backend_host or "http://127.0.0.1", args.backend_port or 8188)

elif args.backend == "sdwebui":
    from temporal.backends.webui_api import WebUIAPIBackend

    backend = WebUIAPIBackend(args.backend_host or "http://127.0.0.1", args.backend_port or 7860)

elif args.backend == "standalone":
    from temporal.backends.standalone import StandaloneBackend

    backend = StandaloneBackend(args.model_dir, args.vae_dir)

else:
    raise ValueError(f"Unknown backend {args.backend}")

shared.init(backend, args.settings_dir)

app = FastAPI(title = "Temporal API")
register_api(app)
app.mount("/", StaticFiles(directory = "ui", html = True), name = "static")

uvicorn.run(app, host = args.host, port = args.port, log_level = "warning")
