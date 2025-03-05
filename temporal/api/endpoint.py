from typing import Any, Literal, Type, get_type_hints

from fastapi import APIRouter

from temporal.engine import Engine


ENDPOINTS: list[Type["Endpoint"]] = []


class Endpoint:
    method: Literal["GET", "POST"]
    path: str

    def __init_subclass__(cls) -> None:
        ENDPOINTS.append(cls)

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.router = APIRouter()
        self.router.add_api_route(
            self.path,
            self.do,
            name = "",
            methods = [self.method],
            response_model = get_type_hints(self.do)["return"],
        )

    async def do(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
