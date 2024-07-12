# NOTE: Hic sunt dracones
from collections.abc import Iterable
from typing import Any, Callable, Iterator

import gradio as gr

from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class ReorderableList(Widget):
    index: int = 0
    stack: list["ReorderableList"] = []

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.items: list[ReorderableAccordion] = []

        self._textbox = GradioWidget(gr.Textbox,
            label = self._format_label("Order"),
            elem_classes = ["temporal-reorderable-list-textbox", f"temporal-index-{ReorderableList.index}"],
        )
        self._textbox._instance.change(None, self._textbox._instance, None, _js = f"(x) => updateReorderableListOrder({ReorderableList.index}, x)")

        self._column = GradioWidget(gr.Column,
            elem_classes = ["temporal-reorderable-list", f"temporal-index-{ReorderableList.index}"],
        )

        ReorderableList.index += 1

    def __enter__(self, *args: Any, **kwargs: Any) -> "ReorderableList":
        self.stack.append(self)
        self._column.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.stack.pop()
        self._column.__exit__(*args, **kwargs)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._textbox

    def read(self, data: ReadData) -> list[int]:
        def try_parse_int(x: str, default: int = -1) -> int:
            try:
                return int(x)
            except ValueError:
                return default

        return [index for x in data[self._textbox].split("|") if (index := try_parse_int(x, -1)) != -1]

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), list):
            result[self._textbox] = {"value": "|".join(str(x) for x in value)}

        return result


class ReorderableAccordion(Widget):
    def __init__(
        self,
        label: str = "",
        value: bool = False,
        open: bool = False,
    ) -> None:
        super().__init__()

        list_widget = ReorderableList.stack[-1]
        index = len(list_widget.items)

        self._checkbox = GradioWidget(gr.Checkbox,
            label = self._format_label(label),
            value = value,
            container = False,
            elem_classes = ["temporal-reorderable-accordion-checkbox", f"temporal-index-{index}"],
        )

        self._accordion = GradioWidget(gr.Accordion,
            label = "",
            open = open,
            elem_classes = ["temporal-reorderable-accordion", f"temporal-index-{index}"],
        )

        list_widget.items.append(self)

    def __enter__(self, *args: Any, **kwargs: Any) -> "ReorderableAccordion":
        self._accordion.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._accordion.__exit__(*args, **kwargs)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield from self._checkbox.dependencies

    def read(self, data: ReadData) -> bool:
        return self._checkbox.read(data)

    def update(self, data: UpdateData) -> UpdateRequest:
        return self._checkbox.update(data)

    def setup_callback(self, callback: Callback) -> None:
        self._checkbox.setup_callback(callback)


class ReorderableAccordionSpecialCheckbox(Widget):
    def __init__(
        self,
        value: bool | Callable[[], bool] = False,
        classes: Iterable[str] = [],
    ) -> None:
        super().__init__()

        self._instance = GradioWidget(gr.Checkbox,
            value = value,
            container = False,
            elem_classes = list(classes) + ["temporal-reorderable-accordion-special-checkbox"],
        )

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield from self._instance.dependencies

    def read(self, data: ReadData) -> bool:
        return self._instance.read(data)

    def update(self, data: UpdateData) -> UpdateRequest:
        return self._instance.update(data)

    def setup_callback(self, callback: Callback) -> None:
        self._instance.setup_callback(callback)
