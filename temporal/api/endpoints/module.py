from asyncio import get_event_loop
from typing import Any, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.api.session import global_session
from temporal.object import object_registry
from temporal.pipeline_module import PipelineModule
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.base64 import encode_with_mime_type
from temporal.utils.image import image_to_base64, pil_to_np
from temporal.video import Video


class _(Endpoint):
    method = "POST"
    path = "/temporal/module/sample"

    class Request(BaseModel):
        data: dict[str, Any]
        size: tuple[int, int] = (256, 256)

    async def do(self, request: Request) -> Optional[str]:
        def render() -> Optional[str]:
            return image_to_base64(PipelineModule.from_json(request.data).sample(request.size), True, "fast")

        return await get_event_loop().run_in_executor(None, render)


class _(Endpoint):
    method = "POST"
    path = "/temporal/module/{project_name}/{id}/visualize"

    async def do(self, project_name: str, id: str) -> Optional[str]:
        def render() -> Optional[str]:
            with global_session.engine.state_lock:
                if (project := global_session.active_project) is None or project.general.name != project_name:
                    project = Project.load(shared.project_store.path / project_name)

                if not isinstance(module := object_registry.get(id, None), PipelineModule):
                    return

                result = module.visualize(project.general)

                if isinstance(result, np.ndarray):
                    return image_to_base64(result, True, "fast")
                elif isinstance(result, Image.Image):
                    return image_to_base64(pil_to_np(result), True, "fast")
                elif isinstance(result, Video) and (data := result.store_in_memory()):
                    return encode_with_mime_type("video", result.format, data)
                else:
                    raise TypeError(result)

        return await get_event_loop().run_in_executor(None, render)
