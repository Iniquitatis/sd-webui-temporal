import asyncio
from typing import Any, Literal, Optional

from pydantic import BaseModel

from modules.api.endpoint import Endpoint
from modules.api.session import global_session
from modules.project import Project
from modules.shared import shared
from modules.utils.fs import remove_directory
from modules.utils.image import image_to_base64
from modules.utils.logging import log


class _(Endpoint):
    method = "POST"
    path = "/api/execution/generate"

    class Request(BaseModel):
        iterations: int = 10
        project: dict[str, Any] = {}

    async def do(self, request: Request) -> None:
        if global_session.is_task_active:
            log.warning("Generation is already started")
            return

        project = Project.from_json(request.project)
        project.general.path = shared.settings.fs.project_dir / project.general.name
        remove_directory(project.general.path)
        project.save(project.general.path)

        global_session.task = asyncio.create_task(global_session.engine.start(project, request.iterations))


class _(Endpoint):
    method = "POST"
    path = "/api/execution/interrupt"

    async def do(self) -> None:
        global_session.engine.stop()

        if global_session.task is not None:
            await global_session.task


class _(Endpoint):
    method = "GET"
    path = "/api/execution/state"

    class Response(BaseModel):
        state: Literal["active", "stopped"]
        current_iteration: int
        total_iterations: int
        eta: float
        preview: Optional[str] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_preview = None

    async def do(self) -> Response:
        engine = global_session.engine

        if (preview := engine.preview) is not None and preview is not self.last_preview:
            self.last_preview = preview
            sent_preview = preview
        else:
            sent_preview = None

        return self.Response(
            state = "active" if global_session.is_task_active else "stopped",
            current_iteration = engine.current_iteration,
            total_iterations = engine.total_iterations,
            eta = engine.stopwatch.eta(engine.current_iteration, engine.total_iterations),
            preview = image_to_base64(sent_preview, True, "fast") if sent_preview is not None else None,
        )
