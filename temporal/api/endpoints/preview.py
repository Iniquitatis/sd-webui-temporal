from typing import Any, Optional

from temporal.api.endpoint import Endpoint
from temporal.shared import shared
from temporal.utils.image import image_to_base64


class _(Endpoint):
    method = "GET"
    path = "/temporal/preview"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_preview = None

    async def do(self) -> Optional[str]:
        if (image := shared.backend.get_preview()) is not None and image is not self.last_preview:
            self.last_preview = image

            return image_to_base64(image, True, "fast")
