import mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/css", ".css")

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from temporal.api import register_api
from temporal.backends.webui_api import WebUIAPIBackend
from temporal.engine import Engine


engine = Engine(WebUIAPIBackend("http://127.0.0.1", 7862), Path("settings"), Path("presets"))

app = FastAPI(title = "Temporal API")

register_api(app, engine)

app.mount("/", StaticFiles(directory = "ui", html = True), name = "static")


uvicorn.run(app, host = "0.0.0.0", port = 8087)
