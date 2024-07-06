from typing import Any, Optional, Type


class Registerable:
    store: Optional[list[Type[Any]]] = None

    id: str = "__UNDEFINED__"
    name: str = "UNDEFINED"

    def __init_subclass__(cls: Type["Registerable"], abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if abstract or cls.store is None:
            return

        if cls.id == "__UNDEFINED__":
            cls.id = f"{cls.__module__}.{cls.__name__}"

        if cls not in cls.store:
            cls.store.append(cls)
        else:
            raise Exception(f"Registerable with ID {cls.id} is already defined")
