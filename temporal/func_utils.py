from copy import deepcopy
from types import SimpleNamespace

def make_func_registerer(**default_params):
    registered = {}

    def wrapper(key, *args, **kwargs):
        def decorator(func):
            registered[key] = SimpleNamespace(func = func, **(
                deepcopy(default_params) |
                {k: v for k, v in zip(default_params.keys(), args)} |
                kwargs
            ))
            return func
        return decorator

    return registered, wrapper
