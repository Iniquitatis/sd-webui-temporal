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
        shared.previewed_modules[request.uuid] = request.state
