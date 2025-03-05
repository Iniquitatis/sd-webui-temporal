from temporal.api.endpoint import Endpoint
from temporal.shared import shared


class _(Endpoint):
    method = "GET"
    path = "/temporal/presets"

    async def do(self) -> list[str]:
        return shared.preset_store.entry_names
