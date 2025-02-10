from pathlib import Path
from threading import Thread
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel

from temporal.animation.parsing import parse_animation
from temporal.blend_modes import BLEND_MODES
from temporal.engine import Engine
from temporal.noise import Noise
from temporal.pipeline import Pipeline
from temporal.pipeline_module import PIPELINE_MODULES, PipelineModule
from temporal.processing_params import ImageToImageParams
from temporal.project import InitialNoiseParams, Project
from temporal.shared import shared
from temporal.utils.image import image_to_base64, load_image, pil_to_np
from temporal.video_filters import VIDEO_FILTERS


class FSOperationRequest(BaseModel):
    store: Literal["presets", "projects"]
    operation: Literal["refresh", "load", "save", "rename", "delete"]
    args: dict[str, Any] = {}


class GenerateRequest(BaseModel):
    parameters: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    pipeline: dict[str, Any] = {}
    animation: str = ""
    iter_count: int = 10


def register_api(app: FastAPI, engine: Engine) -> None:
    @app.get("/temporal/blend_modes")
    async def _() -> Any:
        return {
            x.id: x.name
            for x in BLEND_MODES
        }

    @app.post("/temporal/fs_operation")
    async def _(request: FSOperationRequest) -> Any:
        if request.store == "presets":
            store = shared.preset_store
        elif request.store == "projects":
            store = shared.project_store
        else:
            raise ValueError

        if request.operation == "refresh":
            store.refresh()
        elif request.operation == "load":  # TODO
            raise NotImplementedError
        elif request.operation == "save":  # TODO
            raise NotImplementedError
        elif request.operation == "rename":
            store.rename_entry(request.args["old_name"], request.args["new_name"])
        elif request.operation == "delete":
            store.delete_entry(request.args["name"])
        else:
            raise ValueError

    @app.post("/temporal/generate")
    async def _(request: GenerateRequest) -> Any:
        project = Project(
            path = Path("_standalone_project"),
            parameters = ImageToImageParams(
                **request.parameters,
                images = [pil_to_np(load_image("ui/_example_image.png"))],
            ),
            initial_noise = InitialNoiseParams(
                factor = request.initial_noise.get("factor", 0.0),
                noise = Noise(**request.initial_noise.get("noise", {})),
            ),
            pipeline = Pipeline(
                parallel = request.pipeline.get("parallel", 1),
                modules = [PipelineModule.from_json(x) for x in request.pipeline.get("modules", [])],
            ),
            animation = parse_animation(request.animation),
        )

        thread = Thread(target = engine.start, args = (project, request.iter_count))
        thread.start()

    @app.post("/temporal/interrupt")
    async def _() -> Any:
        shared.backend.interrupt()

    @app.get("/temporal/models")
    async def _() -> Any:
        return [x for x in shared.backend.list_models()]

    @app.get("/temporal/pipeline_modules")
    async def _() -> Any:
        return {module.id: module.schema() for module in PIPELINE_MODULES}

    @app.get("/temporal/presets")
    async def _() -> Any:
        return [x for x in shared.preset_store.entry_names]

    last_preview = None

    @app.get("/temporal/preview")
    async def _() -> Any:
        nonlocal last_preview

        if (image := shared.backend.get_preview()) is not None and image is not last_preview:
            last_preview = image
            return image_to_base64(image)

    @app.get("/temporal/projects")
    async def _() -> Any:
        return [x for x in shared.project_store.entry_names]

    @app.get("/temporal/samplers")
    async def _() -> Any:
        return [x for x in shared.backend.list_samplers()]

    @app.get("/temporal/schedulers")
    async def _() -> Any:
        return [x for x in shared.backend.list_schedulers()]

    @app.get("/temporal/state")
    async def _() -> Any:
        return {
            "state": engine.state,
            "current_iteration": engine.current_iteration,
            "total_iterations": engine.total_iterations,
        }

    @app.get("/temporal/upscalers")
    async def _() -> Any:
        return [x for x in shared.backend.list_upscalers()]

    @app.get("/temporal/vaes")
    async def _() -> Any:
        return [x for x in shared.backend.list_vaes()]

    @app.get("/temporal/video_filters")
    async def _() -> Any:
        return {filter.id: filter.schema() for filter in VIDEO_FILTERS}
