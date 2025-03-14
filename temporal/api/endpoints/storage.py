from typing import Any, Literal

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.fs_store import FSStore
from temporal.shared import shared


StoreType = Literal["presets", "projects"]


class _(Endpoint):
    method = "POST"
    path = "/temporal/storage/delete"

    class Request(BaseModel):
        store: StoreType
        name: str

    async def do(self, request: Request) -> None:
        _get_store(request.store).delete_entry(request.name)


class _(Endpoint):
    method = "POST"
    path = "/temporal/storage/list"

    class Request(BaseModel):
        store: StoreType

    async def do(self, request: Request) -> list[str]:
        return _get_store(request.store).entry_names


class _(Endpoint):
    method = "POST"
    path = "/temporal/storage/load"

    class Request(BaseModel):
        store: StoreType
        name: str

    async def do(self, request: Request) -> dict[str, Any]:
        # FIXME: Shouldn't rely on knowledge about the store type
        if request.store == "presets":
            return _get_store(request.store).load_entry(request.name).to_json()
        elif request.store == "projects":
            project = _get_store(request.store).load_entry(request.name)
            shared.state.active_project = project
            return project.to_json()
        else:
            raise ValueError


class _(Endpoint):
    method = "POST"
    path = "/temporal/storage/new"

    class Request(BaseModel):
        store: StoreType

    async def do(self, request: Request) -> None:
        pass  # TODO


class _(Endpoint):
    method = "POST"
    path = "/temporal/storage/refresh"

    class Request(BaseModel):
        store: StoreType

    async def do(self, request: Request) -> None:
        _get_store(request.store).refresh()


class _(Endpoint):
    method = "POST"
    path = "/temporal/storage/rename"

    class Request(BaseModel):
        store: StoreType
        old_name: str
        new_name: str

    async def do(self, request: Request) -> None:
        _get_store(request.store).rename_entry(request.old_name, request.new_name)


class _(Endpoint):
    method = "POST"
    path = "/temporal/storage/save"

    class Request(BaseModel):
        store: StoreType
        name: str
        data: dict[str, Any]

    async def do(self, request: Request) -> None:
        store = _get_store(request.store)
        store.save_entry(request.name, store.type.from_json(request.data))


def _get_store(name: str) -> FSStore[Any]:
    if name == "presets":
        return shared.preset_store
    elif name == "projects":
        return shared.project_store
    else:
        raise ValueError
