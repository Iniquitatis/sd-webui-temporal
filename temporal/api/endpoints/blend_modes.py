from temporal.api.endpoint import Endpoint
from temporal.blend_modes import BLEND_MODES


class _(Endpoint):
    method = "GET"
    path = "/temporal/blend_modes"

    async def do(self) -> dict[str, str]:
        return {x.__type_name__: x.name for x in BLEND_MODES}
