from asyncio import get_event_loop
from typing import Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.shared import shared
from temporal.utils.image import image_to_base64, pil_to_np


class _(Endpoint):
    method = "POST"
    path = "/temporal/render_graph"

    class Request(BaseModel):
        uuid: str

    async def do(self, request: Request) -> Optional[str]:
        def render() -> Optional[str]:
            with shared.state_lock:
                project = shared.state.active_project

            if (project is not None and (module := project.pipeline.find_module(request.uuid, MeasuringModule)) is not None):
                return image_to_base64(pil_to_np(module.plot()), True, "fast")

        return await get_event_loop().run_in_executor(None, render)
