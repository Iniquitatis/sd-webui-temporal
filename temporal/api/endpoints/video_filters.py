from typing import Any

from temporal.api.endpoint import Endpoint
from temporal.video_filter import VIDEO_FILTERS


class _(Endpoint):
    method = "GET"
    path = "/temporal/video_filters"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {filter.__type_name__: filter.schema() for filter in VIDEO_FILTERS}
