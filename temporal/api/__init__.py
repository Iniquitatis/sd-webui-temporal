from fastapi import FastAPI

from temporal import REPO_ROOT
from temporal.api.endpoint import ENDPOINTS
from temporal.utils.modules import import_modules, list_modules_in_directory


def register_api(app: FastAPI) -> None:
    import_modules(list_modules_in_directory(REPO_ROOT / "temporal" / "api" / "endpoints"))

    for cls in ENDPOINTS:
        app.include_router(cls().router)
