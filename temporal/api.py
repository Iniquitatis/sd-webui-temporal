from asyncio import get_event_loop
from typing import Any, Literal, Optional, Type, get_type_hints
from uuid import uuid4

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from temporal.blend_modes import BLEND_MODES
from temporal.engine import Engine
from temporal.global_options import GlobalOptions
from temporal.gradient import Gradient
from temporal.noise import Noise
from temporal.pattern import Pattern
from temporal.pipeline_module import PIPELINE_MODULES
from temporal.project import Project
from temporal.shared import shared
from temporal.thread_queue import ThreadQueue
from temporal.utils.bytes import bytes_to_base64
from temporal.utils.image import base64_to_image, image_to_base64
from temporal.video_filters import VIDEO_FILTERS
from temporal.video_renderer import VideoRenderer


ENDPOINTS: list[Type["Endpoint"]] = []


class Endpoint:
    method: Literal["GET", "POST"]
    path: str

    def __init_subclass__(cls) -> None:
        ENDPOINTS.append(cls)

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.router = APIRouter()
        self.router.add_api_route(
            self.path,
            self.do,
            name = "",
            methods = [self.method],
            response_model = get_type_hints(self.do)["return"],
        )

    async def do(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _(Endpoint):
    method = "POST"
    path = "/temporal/apply_settings"

    class Request(BaseModel):
        data: dict[str, Any] = {}

    async def do(self, request: Request) -> None:
        shared.options = GlobalOptions.from_json(request.data)
        shared.options.save(shared.options_path)


class _(Endpoint):
    method = "GET"
    path = "/temporal/blend_modes"

    async def do(self) -> dict[str, str]:
        return {x.__type_name__: x.name for x in BLEND_MODES}


class _(Endpoint):
    method = "POST"
    path = "/temporal/fs_operation"

    class Request(BaseModel):
        store: Literal["presets", "projects"]
        operation: Literal["refresh", "load", "save", "rename", "delete"]
        args: dict[str, Any] = {}

    async def do(self, request: Request) -> Optional[dict[str, Any]]:
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


class _(Endpoint):
    method = "POST"
    path = "/temporal/generate"

    class Request(BaseModel):
        class Session(BaseModel):
            load_parameters: bool = True
            continue_from_last_frame: bool = True
            iter_count: int = 10

        image: Optional[str] = None
        project: dict[str, Any] = {}
        session: Session = Session()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.queue = ThreadQueue()

    async def do(self, request: Request) -> None:
        if self.queue.busy:
            return

        path = shared.options.output.output_dir / request.project.get("general", {}).get("name", "untitled")

        if request.session.load_parameters:
            project = Project.load(path)
        else:
            project = Project.from_json(request.project)
            project.general.path = path
            project.general.initial_image = base64_to_image(request.image) if request.image else None

        if not request.session.continue_from_last_frame:
            project.general.delete_all_frames()
            project.delete_session_data()

        self.queue.enqueue(self.engine.start, project, request.session.iter_count)


class _(Endpoint):
    method = "POST"
    path = "/temporal/interrupt"

    async def do(self) -> None:
        self.engine.stop()


class _(Endpoint):
    method = "GET"
    path = "/temporal/models"

    async def do(self) -> list[str]:
        return [x for x in shared.backend.list_models()]


class _(Endpoint):
    method = "GET"
    path = "/temporal/option_categories"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {key: field.type.schema() for key, field in shared.options.__fields__.items()}


class _(Endpoint):
    method = "GET"
    path = "/temporal/pipeline_modules"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {module.__type_name__: module.schema() for module in PIPELINE_MODULES}


class _(Endpoint):
    method = "GET"
    path = "/temporal/presets"

    async def do(self) -> list[str]:
        return shared.preset_store.entry_names


class _(Endpoint):
    method = "GET"
    path = "/temporal/preview"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_preview = None

    async def do(self) -> Optional[str]:
        if (image := shared.backend.get_preview()) is not None and image is not self.last_preview:
            self.last_preview = image

            return image_to_base64(image, "fast")


class _(Endpoint):
    method = "POST"
    path = "/temporal/preview_state"

    class Request(BaseModel):
        uuid: str
        state: bool

    async def do(self, request: Request) -> None:
        shared.previewed_modules[request.uuid] = request.state


class _(Endpoint):
    method = "POST"
    path = "/temporal/project_metadata"

    class Request(BaseModel):
        name: str
        include_last_image: bool = False

    class Response(BaseModel):
        frame_count: int
        first_frame_index: int
        last_frame_index: int
        last_image: Optional[str]

    async def do(self, request: Request) -> Response:
        project = shared.project_store.load_entry(request.name)

        return self.Response(
            frame_count = project.general.get_actual_frame_count(),
            first_frame_index = project.general.get_first_frame_index(),
            last_frame_index = project.general.get_last_frame_index(),
            last_image = image_to_base64(image)
                if request.include_last_image and (image := project.general.get_last_frame()) is not None
                else None,
        )


class _(Endpoint):
    method = "GET"
    path = "/temporal/projects"

    async def do(self) -> list[str]:
        return shared.project_store.entry_names


class _(Endpoint):
    method = "POST"
    path = "/temporal/render_texture"

    class Request(BaseModel):
        type: Literal["gradient", "noise", "pattern"]
        data: dict[str, Any] = {}
        size: tuple[int, int] = (256, 256)
        channels: int = 3

    async def do(self, request: Request) -> str:
        if request.type == "gradient":
            cls = Gradient
        elif request.type == "noise":
            cls = Noise
        elif request.type == "pattern":
            cls = Pattern
        else:
            raise ValueError

        def render() -> str:
            return image_to_base64(cls.from_json(request.data).generate((request.size[1], request.size[0], request.channels)), "fast")

        return await get_event_loop().run_in_executor(None, render)


class _(Endpoint):
    method = "POST"
    path = "/temporal/render_video"

    class Request(BaseModel):
        type: Literal["draft", "final"]
        data: dict[str, Any] = {}

    async def do(self, request: Request) -> Optional[str]:
        def render() -> Optional[str]:
            shared.video_renderer = VideoRenderer.from_json(request.data)

            with self.engine._state_lock:
                project = self.engine.active_project

            if not project:
                return

            return bytes_to_base64(project.general.render_video(shared.video_renderer, request.type == "final", False).read_bytes())

        return await get_event_loop().run_in_executor(None, render)


class _(Endpoint):
    method = "GET"
    path = "/temporal/samplers"

    async def do(self) -> list[str]:
        return [x for x in shared.backend.list_samplers()]


class _(Endpoint):
    method = "GET"
    path = "/temporal/schedulers"

    async def do(self) -> list[str]:
        return [x for x in shared.backend.list_schedulers()]


class _(Endpoint):
    method = "GET"
    path = "/temporal/state"

    class Response(BaseModel):
        state: str
        current_iteration: int
        total_iterations: int

    async def do(self) -> Response:
        with self.engine._state_lock:
            return self.Response(
                state = self.engine.state,
                current_iteration = self.engine.current_iteration,
                total_iterations = self.engine.total_iterations,
            )


class _(Endpoint):
    method = "GET"
    path = "/temporal/uuid"

    async def do(self) -> str:
        return str(uuid4())


class _(Endpoint):
    method = "GET"
    path = "/temporal/upscalers"

    async def do(self) -> list[str]:
        return [x for x in shared.backend.list_upscalers()]


class _(Endpoint):
    method = "GET"
    path = "/temporal/vaes"

    async def do(self) -> list[str]:
        return [x for x in shared.backend.list_vaes()]


class _(Endpoint):
    method = "GET"
    path = "/temporal/video_filters"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {filter.__type_name__: filter.schema() for filter in VIDEO_FILTERS}


def register_api(app: FastAPI, engine: Engine) -> None:
    endpoints: list[Endpoint] = []

    for cls in ENDPOINTS:
        endpoint = cls(engine)
        app.include_router(endpoint.router)
        endpoints.append(endpoint)
