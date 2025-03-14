from typing import Any, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.object import object_registry, object_types
from temporal.serialization import SerializationParams, deserialize, serialize
from temporal.utils import logging


# FIXME: Temporary
logging.log_level = logging.LogLevel.DEBUG


class _(Endpoint):
    method = "GET"
    path = "/temporal/object/list"

    async def do(self) -> list[str]:
        return list(object_registry.keys())


class _(Endpoint):
    method = "GET"
    path = "/temporal/object/{type}/create"

    async def do(self, type: str) -> dict[str, Any]:
        return object_types[type]().to_json()


class _(Endpoint):
    method = "GET"
    path = "/temporal/object/{id}/data"

    async def do(self, id: str) -> Optional[dict[str, Any]]:
        if object := object_registry.get(id):
            return object.to_json()


class _(Endpoint):
    method = "GET"
    path = "/temporal/object/{id}/field"

    class Request(BaseModel):
        key: str

    async def do(self, id: str, request: Request) -> Any:
        # FIXME: Temporary
        logging.debug("Got", id, request)

        if object := object_registry.get(id):
            return serialize(
                object.__fields__[request.key].type,
                getattr(object, request.key),
                SerializationParams(),
            )


class _(Endpoint):
    method = "POST"
    path = "/temporal/object/{id}/field"

    class Request(BaseModel):
        key: str
        value: Any

    async def do(self, id: str, request: Request) -> None:
        # FIXME: Temporary
        logging.debug("Got", id, request)

        if object := object_registry.get(id):
            setattr(object, request.key, deserialize(
                object.__fields__[request.key].type,
                request.value,
                SerializationParams(),
            ))
