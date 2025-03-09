from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.shared import shared


class _(Endpoint):
    method = "POST"
    path = "/temporal/preview_state"

    class Request(BaseModel):
        uuid: str
        state: bool

    async def do(self, request: Request) -> None:
        with shared.state_lock:
            project = shared.state.active_project

        if (project is not None and (module := project.pipeline.find_module(request.uuid)) is not None):
            module.preview = request.state
