from asyncio import get_event_loop
from typing import Any, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.pipeline_module import PipelineModule
from temporal.utils.image import image_to_base64


class _(Endpoint):
    method = "POST"
    path = "/temporal/render_sample"

    class Request(BaseModel):
        data: dict[str, Any]
        size: tuple[int, int] = (256, 256)

    async def do(self, request: Request) -> Optional[str]:
        def render() -> Optional[str]:
            return image_to_base64(PipelineModule.from_json(request.data).sample(request.size), "fast")

        return await get_event_loop().run_in_executor(None, render)
