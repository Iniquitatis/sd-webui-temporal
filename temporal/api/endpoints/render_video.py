from asyncio import get_event_loop
from typing import Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.utils.bytes import bytes_to_base64


class _(Endpoint):
    method = "POST"
    path = "/temporal/render_video"

    class Request(BaseModel):
        uuid: str

    async def do(self, request: Request) -> Optional[str]:
        # NOTE/FIXME: Not a big deal, including it on top just messes up
        # registration ordering a little
        from temporal.pipeline_modules.tool.video_rendering import VideoRenderingModule

        def render() -> Optional[str]:
            with self.engine._state_lock:
                project = self.engine.state.active_project

            if (project is not None and
                (module := project.pipeline.find_module(request.uuid, VideoRenderingModule)) is not None):
                return bytes_to_base64(module.render(project.general, False).read_bytes())

        return await get_event_loop().run_in_executor(None, render)
