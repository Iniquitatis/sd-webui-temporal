from typing import Any

from temporal.api.endpoint import Endpoint
from temporal.object import object_types


class _(Endpoint):
    method = "GET"
    path = "/temporal/object/types"

    async def do(self) -> list[str]:
        return list(object_types.keys())


class _(Endpoint):
    method = "GET"
    path = "/temporal/object/{type}/schema"

    async def do(self, type: str) -> dict[str, Any]:
        return object_types[type].schema()


class _(Endpoint):
    method = "GET"
    path = "/temporal/object/{type}/subtypes"

    async def do(self, type: str) -> list[str]:
        return [x.__type_name__ for x in object_types[type].__subtypes__]
