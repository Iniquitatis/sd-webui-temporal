from typing import Any, Literal, Optional

from pydantic import BaseModel

from temporal.api.endpoint import Endpoint
from temporal.shared import shared


class _(Endpoint):
    method = "POST"
    path = "/temporal/fs_operation"

    class Request(BaseModel):
        store: Literal["presets", "projects"]
        operation: Literal["refresh", "load", "save", "rename", "delete"]
        args: dict[str, Any] = {}

    async def do(self, request: Request) -> Optional[dict[str, Any]]:
        if request.store == "presets":
            store = shared.preset_store
        elif request.store == "projects":
            store = shared.project_store
        else:
            raise ValueError

        if request.operation == "refresh":
            store.refresh()
        elif request.operation == "load":
            return store.load_entry(request.args["name"]).to_json()
        elif request.operation == "save":
            store.save_entry(request.args["name"], store.type.from_json(request.args["data"]))
        elif request.operation == "rename":
            store.rename_entry(request.args["old_name"], request.args["new_name"])
        elif request.operation == "delete":
            store.delete_entry(request.args["name"])
        else:
            raise ValueError
