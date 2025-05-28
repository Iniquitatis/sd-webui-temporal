import html
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from modules import script_callbacks, scripts
from modules.initialize_util import gradio_server_name
from modules.shared_cmd_options import cmd_opts

from temporal.api import register_api
from temporal.backends.webui_api import WebUIAPIBackend
from temporal.shared import shared


EXTENSION_DIR = Path(scripts.basedir())


def on_app_started(_: gr.Blocks, app: FastAPI) -> None:
    shared.init(WebUIAPIBackend(
        host if (host := gradio_server_name()) and host != "0.0.0.0" else "127.0.0.1",
        cmd_opts.port or 7860,
    ), EXTENSION_DIR / "settings")

    register_api(app)


def on_ui_tabs() -> list[tuple[gr.Blocks, str, str]]:
    with gr.Blocks() as block:
        gr.HTML(f'<iframe id="temporal-iframe" src="/file={html.escape((EXTENSION_DIR / "ui" / "index.html").as_posix())}"></iframe>')

        return [(block, "Temporal", "temporal")]


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
