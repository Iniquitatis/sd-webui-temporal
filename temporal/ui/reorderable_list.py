from collections.abc import Iterable
from typing import Any, Callable, Iterator

import gradio as gr

from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.communication_box import CommunicationBox
from temporal.ui.gradio_widget import GradioWidget
from temporal.utils.collection import swap_kv


class ReorderableList(Widget):
    _stack: list["ReorderableList"] = []

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self._order: dict[int, int] = {}

        self._communication = CommunicationBox("Order", f"temporalUpdateReorderableList({self.index}, data)")

        self._column = GradioWidget(gr.Column, elem_classes = [
            "temporal-reorderable-list",
            self.index_class,
            self._communication.communication_class,
        ])

    @staticmethod
    def add_accordion(accordion: "ReorderableAccordion") -> None:
        top_list = ReorderableList._stack[-1]
        top_list._order[accordion.index] = len(top_list._order)

    def __enter__(self, *args: Any, **kwargs: Any) -> "ReorderableList":
        self._stack.append(self)
        self._column.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._stack.pop()
        self._column.__exit__(*args, **kwargs)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._communication

    def read(self, data: ReadData) -> list[int]:
        return [self._order[x] for x in data[self._communication]]

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), list):
            swapped_order = swap_kv(self._order)
            result[self._communication] = {"value": [swapped_order[x] for x in value]}

        return result

    # TODO: Send list of ordered accordion indices here
    def setup_callback(self, callback: Callback) -> None:
        return super().setup_callback(callback)


class ReorderableAccordion(Widget):
    def __init__(
        self,
        label: str = "",
        value: bool = False,
        open: bool = False,
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Accordion,
            label = "",
            open = open,
            elem_classes = ["temporal-reorderable-accordion", self.index_class],
        ) as self._accordion:
            self._checkbox = GradioWidget(gr.Checkbox,
                label = self._format_label(label),
                value = value,
                container = False,
                elem_classes = ["temporal-reorderable-accordion-checkbox"],
            )

        ReorderableList.add_accordion(self)

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
