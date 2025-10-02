from typing import Any, Literal

from modules.api.endpoint import Endpoint
from modules.fs_store import FSStore
from modules.shared import shared


StoreType = Literal["projects"]


class _(Endpoint):
    method = "POST"
    path = "/api/storage/{type}/list"

    async def do(self, type: StoreType) -> list[str]:
        return _get_store(type).entry_names


class _(Endpoint):
    method = "POST"
    path = "/api/storage/{type}/new"

    async def do(self, type: StoreType) -> None:
        return _get_store(type).type().to_json()


class _(Endpoint):
    method = "POST"
    path = "/api/storage/{type}/refresh"

    async def do(self, type: StoreType) -> None:
        _get_store(type).refresh()


class _(Endpoint):
    method = "POST"
    path = "/api/storage/{type}/{entry}/delete"

    async def do(self, type: StoreType, entry: str) -> None:
        _get_store(type).delete_entry(entry)


class _(Endpoint):
    method = "POST"
    path = "/api/storage/{type}/{entry}/load"

    async def do(self, type: StoreType, entry: str) -> dict[str, Any]:
        return _get_store(type).load_entry(entry).to_json()


class _(Endpoint):
    method = "POST"
    path = "/api/storage/{type}/{entry}/rename"

    async def do(self, type: StoreType, entry: str, new_name: str) -> None:
        _get_store(type).rename_entry(entry, new_name)


class _(Endpoint):
    method = "POST"
    path = "/api/storage/{type}/{entry}/save"

    async def do(self, type: StoreType, entry: str, data: dict[str, Any]) -> None:
        store = _get_store(type)
        store.save_entry(entry, store.type.from_json(data))


def _get_store(name: StoreType) -> FSStore[Any]:
    if name == "projects":
        return shared.project_store
    else:
        raise ValueError
