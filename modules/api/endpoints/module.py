from asyncio import get_event_loop
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel

from modules.api.endpoint import Endpoint
from modules.pipeline_module import PipelineModule
from modules.shared import shared
from modules.utils.image import NumpyImage, base64_to_image, ensure_image_dims, image_to_base64


class _(Endpoint):
    method = "POST"
    path = "/api/module/execute"

    class Request(BaseModel):
        data: dict[str, Any]
        image: str

    async def do(self, request: Request) -> Optional[str]:
        def render() -> Optional[str]:
            module = PipelineModule.from_json(request.data)
            image = base64_to_image(request.image)
            return image_to_base64(module.execute(image), True, "fast")

        return await get_event_loop().run_in_executor(None, render)


class _(Endpoint):
    method = "POST"
    path = "/api/module/sample"

    class Request(BaseModel):
        data: dict[str, Any]
        size: tuple[int, int] = (256, 256)

    async def do(self, request: Request) -> Optional[str]:
        def render() -> Optional[str]:
            module = PipelineModule.from_json(request.data)
            image = self._get_scaled_sample_image(request.size)
            return image_to_base64(module.sample(image), True, "fast")

        return await get_event_loop().run_in_executor(None, render)

    @staticmethod
    @lru_cache
    def _get_scaled_sample_image(size: tuple[int, int]) -> NumpyImage:
        return ensure_image_dims(shared.sample_image, size, 3)


# FIXME: Won't work, given that projects won't be saved anymore
class _(Endpoint):
    method = "POST"
    path = "/api/module/{project_name}/{id}/visualize"

    async def do(self, project_name: str, id: str) -> Optional[str]:
        def render() -> Optional[str]:
            # with global_session.engine.state_lock:
            #     if (project := global_session.active_project) is None or project.general.name != project_name:
            #         project = Project.load(shared.project_store.path / project_name)

            #     if not isinstance(module := object_registry.get(id, None), PipelineModule):
            #         return

            #     result = module.visualize(project.general)

            #     if isinstance(result, np.ndarray):
            #         return image_to_base64(result, True, "fast")
            #     elif isinstance(result, Image.Image):
            #         return image_to_base64(pil_to_np(result), True, "fast")
            #     elif isinstance(result, Video) and (data := result.store_in_memory()):
            #         return encode_with_mime_type("video", result.format, data)
            #     else:
            #         raise TypeError(result)

            return None

        return await get_event_loop().run_in_executor(None, render)
