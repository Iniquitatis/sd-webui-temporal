from typing import Any

from temporal.api.endpoint import Endpoint
from temporal.blend_modes import BlendMode
from temporal.pipeline_module import PipelineModule
from temporal.settings import Settings
from temporal.video_filter import VideoFilter


class _(Endpoint):
    method = "GET"
    path = "/temporal/schema/blend_modes"

    async def do(self) -> dict[str, str]:
        return {x.__type_name__: x.name for x in BlendMode.__subtypes__}


class _(Endpoint):
    method = "GET"
    path = "/temporal/schema/option_categories"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {key: field.type.schema() for key, field in Settings.__fields__.items()}


class _(Endpoint):
    method = "GET"
    path = "/temporal/schema/pipeline_modules"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {x.__type_name__: x.schema() for x in PipelineModule.__subtypes__}


class _(Endpoint):
    method = "GET"
    path = "/temporal/schema/video_filters"

    async def do(self) -> dict[str, dict[str, Any]]:
        return {x.__type_name__: x.schema() for x in VideoFilter.__subtypes__}
