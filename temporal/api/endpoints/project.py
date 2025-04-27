from typing import Optional

from temporal.api.endpoint import Endpoint
from temporal.shared import shared
from temporal.utils.image import image_to_base64


class _(Endpoint):
    method = "GET"
    path = "/temporal/project/{name}/last_image"

    async def do(self, name: str) -> Optional[str]:
        project = shared.project_store.load_entry(name)

        if (image := project.iteration.image) is not None:
            return image_to_base64(image, True, "fast")
        else:
            return None
