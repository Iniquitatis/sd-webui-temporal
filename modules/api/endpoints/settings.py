from typing import Any

from pydantic import BaseModel

from modules.api.endpoint import Endpoint
from modules.settings import Settings
from modules.shared import shared


class _(Endpoint):
    method = "POST"
    path = "/api/settings/apply"

    class Request(BaseModel):
        data: dict[str, Any] = {}

    async def do(self, request: Request) -> None:
        shared.settings = Settings.from_json(request.data)
        shared.settings.save(shared.settings_path)
