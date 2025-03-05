from typing import Any

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.settings import Settings
from temporal.shared import shared


class _(Endpoint):
    method = "POST"
    path = "/temporal/apply_settings"

    class Request(BaseModel):
        data: dict[str, Any] = {}

    async def do(self, request: Request) -> None:
        shared.settings = Settings.from_json(request.data)
        shared.settings.save(shared.settings_path)
