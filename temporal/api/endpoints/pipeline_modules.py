from typing import Any

from temporal.api.endpoint import Endpoint
from temporal.pipeline_module import PIPELINE_MODULES


class _(Endpoint):
    method = "GET"
    path = "/temporal/pipeline_modules"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {module.__type_name__: module.schema() for module in PIPELINE_MODULES}
