from typing import Any, Optional, Type


class Registerable:
    store: Optional[list[Type[Any]]] = None

    name: str = "UNDEFINED"

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if abstract or cls.store is None:
            return

        if cls not in cls.store:
            cls.store.append(cls)
        else:
            raise Exception(f"Registerable {cls} is already defined")
