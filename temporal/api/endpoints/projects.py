from temporal.api.endpoint import Endpoint
from temporal.shared import shared


class _(Endpoint):
    method = "GET"
    path = "/temporal/projects"

    async def do(self) -> list[str]:
        return shared.project_store.entry_names
