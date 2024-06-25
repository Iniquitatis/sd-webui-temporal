from collections.abc import Iterable
from typing import Callable, Generic, Iterator, Optional, TypeVar, cast

from temporal.ui import ReadData, ResolvedCallback, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioThing, GradioWidget


T = TypeVar("T", bound = GradioThing)
U = TypeVar("U")


class Selector(Widget, Generic[T, U]):
    def __init__(
        self,
        gr_type: Callable[..., T],
        label: str = "",
        choices: Iterable[str | tuple[U, str]] = [],
        value: Optional[U | Callable[[], U]] = None,
    ) -> None:
        super().__init__()

        self.choices = _unpack_choices(choices)

        self._instance = GradioWidget(gr_type, label = self._format_label(label), type = "index", choices = [name for _, name in self.choices], value = lambda: self._find_name(value()) if callable(value) else self._find_name(cast(U, value)))

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._instance

    def read(self, data: ReadData) -> U:
        return self.choices[data[self._instance]][0]

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {self._instance: data}

        if (choices := data.pop("choices", None)) is not None:
            self.choices = _unpack_choices(choices)

            result[self._instance]["choices"] = [name for _, name in self.choices]

        if (value := data.pop("value", None)) is not None:
            result[self._instance]["value"] = self._find_name(value)

        return result

    def setup_callback(self, callback: ResolvedCallback) -> None:
        self._instance.setup_callback(callback)

    def _find_name(self, value: U) -> Optional[str]:
        for choice_value, name in self.choices:
            if choice_value == value:
                return name

        return None


def _unpack_choices(choices: Iterable[str | tuple[U, str]]) -> list[tuple[str | U, str]]:
    return [
        (item if isinstance(item, tuple) else (item, item))
        for item in choices
    ]
