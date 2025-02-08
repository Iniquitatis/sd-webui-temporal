from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from temporal.backend import TextToImageParams
from temporal.blend_modes import BLEND_MODES
from temporal.meta.configurable import ConfigurableParam
from temporal.engine import Engine
from temporal.pipeline_module import PIPELINE_MODULES
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.shared import shared
from temporal.utils.image import image_to_base64
from temporal.video_filters import VIDEO_FILTERS


class RequestTextToImageParams(BaseModel):
    model: str = ""
    vae: str | None = ""
    clip_skip: int = 1
    positive_prompts: list[str] = []
    negative_prompts: list[str] = []
    width: int = 512
    height: int = 512
    sampler: str = ""
    scheduler: str = ""
    steps: int = 20
    cfg: float = 5.0
    strength: float = 0.5
    seeds: list[int] = []


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

    @app.post("/temporal/text_to_image")
    async def _(params: RequestTextToImageParams) -> Any:
        if images := shared.backend.text_to_image(TextToImageParams(**params.dict())):
            return [image_to_base64(x) for x in images]

        return []

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
