from inspect import signature
from typing import Any, Literal, Type, get_type_hints

from fastapi import APIRouter

from temporal.engine import Engine
from temporal.utils import logging


# FIXME: Temporary
logging.log_level = logging.LogLevel.DEBUG


ENDPOINTS: list[Type["Endpoint"]] = []


class Endpoint:
    method: Literal["GET", "POST"]
    path: str

    def __init_subclass__(cls) -> None:
        original_do = cls.do

        async def wrapped_do(self, *args: Any, **kwargs: Any) -> Any:
            logging.info(cls.method, cls.path.format_map(kwargs))

            # NOTE: Wrapped into a condition because sometimes values can be
            # _very_ big
            if logging.log_level == logging.LogLevel.DEBUG and len(kwargs) > 0:
                logging.debug(", ".join(f"{k} = {repr(v)}" for k, v in kwargs.items()))

            return await original_do(self, *args, **kwargs)

        setattr(wrapped_do, "__signature__", signature(original_do))

        cls.do = wrapped_do

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
