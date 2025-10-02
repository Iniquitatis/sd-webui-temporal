from fastapi import FastAPI

from modules.api.endpoint import ENDPOINTS
from modules.utils.modules import import_modules, list_modules_in_directory


def register_api(app: FastAPI) -> None:
    import_modules(list_modules_in_directory("modules/api/endpoints"))

    for cls in ENDPOINTS:
        app.include_router(cls().router)
