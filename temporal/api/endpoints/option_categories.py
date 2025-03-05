from typing import Any

from temporal.api.endpoint import Endpoint
from temporal.shared import shared


class _(Endpoint):
    method = "GET"
    path = "/temporal/option_categories"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {key: field.type.schema() for key, field in shared.settings.__fields__.items()}
