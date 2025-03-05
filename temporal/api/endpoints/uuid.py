from uuid import uuid4

from temporal.api.endpoint import Endpoint


class _(Endpoint):
    method = "GET"
    path = "/temporal/uuid"

    async def do(self) -> str:
        return str(uuid4())
