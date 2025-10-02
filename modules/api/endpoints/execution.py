from asyncio import get_event_loop
from typing import Any, Optional

from pydantic import BaseModel

from modules.api.endpoint import Endpoint
from modules.api.session import global_session
from modules.project import Project
from modules.shared import shared
from modules.utils.fs import remove_directory
from modules.utils.image import image_to_base64


class _(Endpoint):
    method = "POST"
    path = "/temporal/execution/generate"

    class Request(BaseModel):
        iterations: int = 10
        project: dict[str, Any] = {}

    async def do(self, request: Request) -> None:
        project = Project.from_json(request.project)
        project.general.path = shared.settings.fs.project_dir / project.general.name
        remove_directory(project.general.path)
        project.save(project.general.path)
        get_event_loop().run_in_executor(None, global_session.engine.start, project, request.iterations)


class _(Endpoint):
    method = "POST"
    path = "/temporal/execution/interrupt"

    async def do(self) -> None:
        global_session.engine.stop()


class _(Endpoint):
    method = "GET"
    path = "/temporal/execution/state"

    class Response(BaseModel):
        state: str
        current_iteration: int
        total_iterations: int
        preview: Optional[str] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_preview = None

    async def do(self) -> Response:
        with global_session.engine.state_lock:
            # FIXME: After starting the Engine in a _different_ thread, this
            # thing might not even be "initialized", providing false information
            # to the frontend
            state = global_session.engine.state

            if (preview := state.preview) is not None and preview is not self.last_preview:
                self.last_preview = preview
                sent_preview = preview
            else:
                sent_preview = None

            return self.Response(
                state = state.state,
                current_iteration = state.current_iteration,
                total_iterations = state.total_iterations,
                preview = image_to_base64(sent_preview, True, "fast") if sent_preview is not None else None,
            )
