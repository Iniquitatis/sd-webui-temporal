from typing import Any, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.shared import shared
from temporal.utils.image import image_to_base64


class _(Endpoint):
    method = "GET"
    path = "/temporal/state"

    class Response(BaseModel):
        state: str
        current_iteration: int
        total_iterations: int
        preview: Optional[str] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_preview = None

    async def do(self) -> Response:
        with shared.state_lock:
            if (preview := shared.state.preview) is not None and preview is not self.last_preview:
                self.last_preview = preview
                sent_preview = preview
            else:
                sent_preview = None

            return self.Response(
                state = shared.state.state,
                current_iteration = shared.state.current_iteration,
                total_iterations = shared.state.total_iterations,
                preview = image_to_base64(sent_preview, True, "fast") if sent_preview is not None else None,
            )
