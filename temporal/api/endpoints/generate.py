from typing import Any, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.project import Project
from temporal.shared import shared
from temporal.thread_queue import ThreadQueue
from temporal.utils.image import base64_to_image


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

        path = shared.settings.output.output_dir / request.project.get("general", {}).get("name", "untitled")

        if request.session.load_parameters:
            project = Project.load(path)

            if not request.session.continue_from_last_frame:
                project.delete_session_data()

        else:
            if path.is_dir():
                existing = Project.load(path)
                existing.delete_session_data()

            project = Project.from_json(request.project)
            project.general.path = path
            project.general.initial_image = base64_to_image(request.image, True) if request.image else None

        self.queue.enqueue(self.engine.start, project, request.session.iter_count)
