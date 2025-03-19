from typing import Any, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.project import Project
from temporal.shared import shared
from temporal.thread_queue import ThreadQueue
from temporal.utils.image import base64_to_image, image_to_base64


class _(Endpoint):
    method = "POST"
    path = "/temporal/execution/generate"

    class Request(BaseModel):
        image: Optional[str] = None
        load_parameters: bool = True
        continue_from_last_iteration: bool = True
        iter_count: int = 10
        project: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.queue = ThreadQueue()

    async def do(self, request: Request) -> None:
        if self.queue.busy:
            return

        path = shared.settings.fs.project_dir / request.project.get("general", {}).get("name", "untitled")

        if request.load_parameters:
            project = Project.load(path)

            if not request.continue_from_last_iteration:
                project.delete_session_data()
                project.general.initial_image = base64_to_image(request.image, True) if request.image else None

        else:
            if path.is_dir():
                existing = Project.load(path)
                existing.delete_session_data()

            project = Project.from_json(request.project)
            project.general.path = path
            project.general.initial_image = base64_to_image(request.image, True) if request.image else None

        self.queue.enqueue(self.engine.start, project, request.iter_count)


class _(Endpoint):
    method = "POST"
    path = "/temporal/execution/interrupt"

    async def do(self) -> None:
        self.engine.stop()


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
        with shared.state_lock:
            if (preview := shared.state.preview) is not None and preview is not self.last_preview:
                self.last_preview = preview
                sent_preview = preview
            else:
                sent_preview = None

            return self.Response(
                state = shared.state.state,
                current_iteration = shared.state.current_iteration,
                total_iterations = shared.state.total_iterations,
                preview = image_to_base64(sent_preview, True, "fast") if sent_preview is not None else None,
            )
