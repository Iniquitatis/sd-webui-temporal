import html
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from modules import script_callbacks, scripts, shared as webui_shared
from modules.shared_state import State

from temporal.api import register_api
from temporal.backends.webui import WebUIBackend
from temporal.engine import Engine


# FIXME: To shut up the type checker
state: State = getattr(webui_shared, "state")


EXTENSION_DIR = Path(scripts.basedir())


class WebUIEngine(Engine):
    def on_iteration(self, iteration: int) -> None:
        state.job = "Temporal main loop"
        state.job_no = iteration


def on_app_started(_: gr.Blocks, app: FastAPI) -> None:
    engine = WebUIEngine(WebUIBackend(), EXTENSION_DIR / "settings", EXTENSION_DIR / "presets")

    register_api(app, engine)


def on_ui_tabs() -> list[tuple[gr.Blocks, str, str]]:
    with gr.Blocks() as block:
        gr.HTML(f'<iframe id="temporal-iframe" src="/file={html.escape((EXTENSION_DIR / "ui" / "index.html").as_posix())}"></iframe>')

        return [(block, "Temporal", "temporal")]


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
