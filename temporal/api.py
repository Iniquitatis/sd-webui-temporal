from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel

from temporal.blend_modes import BLEND_MODES
from temporal.engine import Engine
from temporal.gradient import Gradient
from temporal.noise import Noise
from temporal.pattern import Pattern
from temporal.pipeline_module import PIPELINE_MODULES
from temporal.project import Project
from temporal.shared import shared
from temporal.thread_queue import ThreadQueue
from temporal.utils.image import image_to_base64
from temporal.video_filters import VIDEO_FILTERS


class FSOperationRequest(BaseModel):
    store: Literal["presets", "projects"]
    operation: Literal["refresh", "load", "save", "rename", "delete"]
    args: dict[str, Any] = {}


class GenerateRequest(BaseModel):
    name: str
    project: dict[str, Any]
    load_parameters: bool = True
    continue_from_last_frame: bool = True
    iter_count: int = 10


class RenderTextureRequest(BaseModel):
    type: Literal["gradient", "noise", "pattern"]
    data: dict[str, Any] = {}
    size: tuple[int, int] = (256, 256)
    channels: int = 3


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
        elif request.operation == "load":
            return store.load_entry(request.args["name"]).to_json()
        elif request.operation == "save":
            store.save_entry(request.args["name"], store.type.from_json(request.args["data"]))
        elif request.operation == "rename":
            store.rename_entry(request.args["old_name"], request.args["new_name"])
        elif request.operation == "delete":
            store.delete_entry(request.args["name"])
        else:
            raise ValueError

    generation_queue = ThreadQueue()

    @app.post("/temporal/generate")
    async def _(request: GenerateRequest) -> Any:
        if generation_queue.busy:
            return

        project = Project.from_json(request.project)
        project.general.path = shared.options.output.output_dir / request.name

        if request.load_parameters:
            project.load(project.general.path)

        if not request.continue_from_last_frame:
            project.general.delete_all_frames()
            project.delete_session_data()

        generation_queue.enqueue(engine.start, project, request.iter_count)

    @app.post("/temporal/interrupt")
    async def _() -> Any:
        shared.backend.interrupt()
        engine.running = False

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
            return image_to_base64(image, "fast")

    @app.get("/temporal/projects")
    async def _() -> Any:
        return [x for x in shared.project_store.entry_names]

    @app.post("/temporal/render_texture")
    async def _(request: RenderTextureRequest) -> Any:
        if request.type == "gradient":
            cls = Gradient
        elif request.type == "noise":
            cls = Noise
        elif request.type == "pattern":
            cls = Pattern
        else:
            raise ValueError

        obj = cls.from_json(request.data)

        return image_to_base64(obj.generate((request.size[1], request.size[0], request.channels)), "fast")

    generation_queue = ThreadQueue()

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
