from typing import Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.shared import shared
from temporal.utils.image import image_to_base64


class _(Endpoint):
    method = "POST"
    path = "/temporal/project_metadata"

    class Request(BaseModel):
        name: str
        include_last_image: bool = False

    class Response(BaseModel):
        last_image: Optional[str]

    async def do(self, request: Request) -> Response:
        project = shared.project_store.load_entry(request.name)

        return self.Response(
            last_image = image_to_base64(image)
                if request.include_last_image and (image := project.iteration.image) is not None
                else None,
        )
