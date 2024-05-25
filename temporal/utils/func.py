from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable

Func = Callable[..., Any]
Decorator = Callable[[Func], Func]
Wrapper = Callable[..., Decorator]

def make_func_registerer(**default_params: Any) -> tuple[dict[str, SimpleNamespace], Wrapper]:
    registered = {}

    def wrapper(key: Any, *args: Any, **kwargs: Any) -> Decorator:
        def decorator(func: Func) -> Func:
            registered[key] = SimpleNamespace(func = func, **(
                deepcopy(default_params) |
                {k: v for k, v in zip(default_params.keys(), args)} |
                kwargs
            ))
            return func

        return decorator

    return registered, wrapper
