from typing import Any, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.image import image_to_base64


class _(Endpoint):
    method = "GET"
    path = "/temporal/project/data"

    async def do(self) -> dict[str, Any]:
        with shared.state_lock:
            return shared.state.active_project.to_json()


class _(Endpoint):
    method = "POST"
    path = "/temporal/project/data"

    class Request(BaseModel):
        data: dict[str, Any]

    async def do(self, request: Request) -> None:
        with shared.state_lock:
            shared.state.active_project = Project.from_json(request.data)


class _(Endpoint):
    method = "GET"
    path = "/temporal/project/last_image"

    async def do(self) -> Optional[str]:
        with shared.state_lock:
            return (
                image_to_base64(image, True, "fast")
                if (image := shared.state.active_project.iteration.image) is not None
                else None
            )


class _(Endpoint):
    method = "POST"
    path = "/temporal/project/load"

    class Request(BaseModel):
        name: str

    async def do(self, request: Request) -> None:
        with shared.state_lock:
            shared.state.active_project = shared.project_store.load_entry(request.name)


class _(Endpoint):
    method = "POST"
    path = "/temporal/project/new"

    async def do(self) -> None:
        with shared.state_lock:
            shared.state.active_project = Project()


class _(Endpoint):
    method = "POST"
    path = "/temporal/project/save"

    async def do(self) -> None:
        with shared.state_lock:
            project = shared.state.active_project

        project.save(project.general.path)
