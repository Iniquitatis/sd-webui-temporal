from typing import Any, Optional, Type


class Registerable:
    store: Optional[dict[Any, Type[Any]]] = None

    id: Any = "UNDEFINED"
    name: str = "UNDEFINED"

    def __init_subclass__(cls: Type["Registerable"], abstract: bool = False) -> None:
        if abstract or cls.store is None:
            return

        if cls.id not in cls.store:
            cls.store[cls.id] = cls
        else:
            raise Exception(f"Registerable with ID {cls.id} is already defined")
