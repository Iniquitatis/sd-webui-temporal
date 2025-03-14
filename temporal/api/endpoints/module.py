from asyncio import get_event_loop
from typing import Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.object import object_registry
from temporal.pipeline_module import PipelineModule
from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.base64 import encode_with_mime_type
from temporal.utils.image import image_to_base64, pil_to_np


# FIXME: Temporary
logging.log_level = logging.LogLevel.DEBUG


class _(Endpoint):
    method = "POST"
    path = "/temporal/module/{id}/preview_state"

    class Request(BaseModel):
        state: bool

    async def do(self, id: str, request: Request) -> None:
        if isinstance(module := object_registry.get(id, None), PipelineModule):
            module.preview = request.state


class _(Endpoint):
    method = "POST"
    path = "/temporal/module/{id}/render_graph"

    async def do(self, id: str) -> Optional[str]:
        def render() -> Optional[str]:
            if isinstance(module := object_registry.get(id, None), MeasuringModule):
                return image_to_base64(pil_to_np(module.plot()), True, "fast")

        return await get_event_loop().run_in_executor(None, render)


class _(Endpoint):
    method = "POST"
    path = "/temporal/module/{id}/render_sample"

    class Request(BaseModel):
        size: tuple[int, int] = (256, 256)

    async def do(self, id: str, request: Request) -> Optional[str]:
        def render() -> Optional[str]:
            if isinstance(module := object_registry.get(id, None), PipelineModule):
                return image_to_base64(module.sample(request.size), True, "fast")

        return await get_event_loop().run_in_executor(None, render)


class _(Endpoint):
    method = "POST"
    path = "/temporal/module/{id}/render_video"

    async def do(self, id: str) -> Optional[str]:
        # NOTE/FIXME: Not a big deal, including it on top just messes up
        # registration ordering a little
        from temporal.pipeline_modules.tool.video_rendering import VideoRenderingModule

        def render() -> Optional[str]:
            with shared.state_lock:
                project = shared.state.active_project

            if isinstance(module := object_registry.get(id, None), VideoRenderingModule):
                return encode_with_mime_type("video", "mp4", module.render(project.general, False).read_bytes())

        return await get_event_loop().run_in_executor(None, render)
