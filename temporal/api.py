from typing import Any

from fastapi import FastAPI

from temporal.blend_modes import BLEND_MODES
from temporal.meta.configurable import ConfigurableParam
from temporal.engine import Engine
from temporal.pipeline_module import PIPELINE_MODULES
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.shared import shared
from temporal.video_filters import VIDEO_FILTERS


def register_api(app: FastAPI, engine: Engine) -> None:
    @app.get("/temporal/blend_modes")
    async def _() -> Any:
        return {
            x.id: x.name
            for x in BLEND_MODES
        }

    @app.get("/temporal/models")
    async def _() -> Any:
        return [x for x in shared.backend.list_models()]

    @app.get("/temporal/pipeline_modules")
    async def _() -> Any:
        return {
            module.id: {
                "icon": module.icon,
                "name": module.name,
                "is_filter": issubclass(module, ImageFilter),
                "parameters": {
                    field_id: field.to_json()
                    for field_id, field in module.__fields__.items()
                    if isinstance(field, ConfigurableParam)
                },
            }
            for module in PIPELINE_MODULES
        }

    @app.get("/temporal/presets")
    async def _() -> Any:
        return [x for x in shared.preset_store.entry_names]

    @app.get("/temporal/projects")
    async def _() -> Any:
        return [x for x in shared.project_store.entry_names]

    @app.get("/temporal/samplers")
    async def _() -> Any:
        return [x for x in shared.backend.list_samplers()]

    @app.get("/temporal/schedulers")
    async def _() -> Any:
        return [x for x in shared.backend.list_schedulers()]

    @app.get("/temporal/upscalers")
    async def _() -> Any:
        return [x for x in shared.backend.list_upscalers()]

    @app.get("/temporal/vaes")
    async def _() -> Any:
        return [x for x in shared.backend.list_vaes()]

    @app.get("/temporal/video_filters")
    async def _() -> Any:
        return {
            filter.id: {
                "name": filter.name,
                "parameters": {
                    field_id: field.to_json()
                    for field_id, field in filter.__fields__.items()
                    if isinstance(field, ConfigurableParam)
                },
            }
            for filter in VIDEO_FILTERS
        }
