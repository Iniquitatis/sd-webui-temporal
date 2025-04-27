from asyncio import sleep
from typing import Any, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.api.session import global_session
from temporal.project import Project
from temporal.shared import shared
from temporal.thread_queue import ThreadQueue
from temporal.utils.image import image_to_base64


class _(Endpoint):
    method = "POST"
    path = "/temporal/execution/generate"

    class Request(BaseModel):
        load_parameters: bool = True
        continue_from_last_iteration: bool = True
        iter_count: int = 10
        project: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.queue = ThreadQueue()

    async def do(self, request: Request) -> bool:
        if self.queue.busy:
            return False

        path = shared.settings.fs.project_dir / request.project.get("general", {}).get("name", "untitled")

        # TODO: Extract this logic into frontend. This endpoint should accept
        # either name of a project to load, or data of a new project.
        if request.load_parameters:
            project = Project.load(path)

            if not request.continue_from_last_iteration:
                project.delete_session_data()

        else:
            if path.is_dir():
                existing = Project.load(path)
                existing.delete_session_data()

            project = Project.from_json(request.project)

        def execute(project: Project, iter_count: int) -> None:
            global_session.active_project = project
            global_session.engine.start(project, iter_count)
            global_session.active_project = None

        self.queue.enqueue(execute, project, request.iter_count)

        # FIXME: Kind of a hack and might still fail, for example, on 0
        # iterations or very fast ones. The state should be probably somehow
        # controllable _outside_ of the engine. For example, right here.
        for _ in range(100):  # NOTE: 10 seconds
            with global_session.engine.state_lock:
                if global_session.engine.state.state == "active":
                    break

            await sleep(0.1)
        else:
            return False

        return True


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
