from temporal.api.endpoint import Endpoint
from temporal.shared import shared


class _(Endpoint):
    method = "GET"
    path = "/temporal/schedulers"

    async def do(self) -> list[str]:
        return [x for x in shared.backend.list_schedulers()]
