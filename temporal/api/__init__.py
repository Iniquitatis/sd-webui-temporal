from fastapi import FastAPI

from temporal.api.endpoint import ENDPOINTS
from temporal.engine import Engine
from temporal.utils.modules import import_modules, list_modules_in_directory


def register_api(app: FastAPI, engine: Engine) -> None:
    import_modules(list_modules_in_directory("temporal/api/endpoints"))

    for cls in ENDPOINTS:
        app.include_router(cls(engine).router)
