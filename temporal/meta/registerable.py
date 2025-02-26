from typing import Any, Optional, Type

from temporal.utils.typing import get_full_type_name


class Registerable:
    store: Optional[list[Type[Any]]] = None

    id: str = "__UNDEFINED__"
    name: str = "UNDEFINED"

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls.id = get_full_type_name(cls)

        if abstract or cls.store is None:
            return

        if cls not in cls.store:
            cls.store.append(cls)
        else:
            raise Exception(f"Registerable with ID {cls.id} is already defined")
