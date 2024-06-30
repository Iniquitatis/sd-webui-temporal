# NOTE: Hic sunt dracones
from collections.abc import Iterable
from typing import Any, Callable, Iterator, Optional

import gradio as gr

from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class ModuleList(Widget):
    index = 0

    def __init__(
        self,
        keys: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()

        self._dropdown = GradioWidget(gr.Dropdown,
            label = self._format_label("Order"),
            multiselect = True,
            choices = list(keys) if keys is not None else None,
            value = list(keys) if keys is not None else None,
            elem_classes = ["temporal-module-list-dropdown", f"temporal-index-{ModuleList.index}"],
        )
        self._dropdown._instance.change(None, self._dropdown._instance, None, _js = f"(x) => updateModuleListOrder({ModuleList.index}, x)")

        # NOTE: Necessary to communicate between the web interface and Python
        # (probably there's a better way to do this, but I haven't found it yet)
        self._textbox = GradioWidget(gr.Textbox,
            label = self._format_label("Order"),
            value = "|".join(keys) if keys is not None else None,
            elem_classes = ["temporal-module-list-textbox", f"temporal-index-{ModuleList.index}"],
        )
        self._textbox._instance.change(lambda x: gr.update(value = x.split("|")), self._textbox._instance, self._dropdown._instance)

        self._column = GradioWidget(gr.Column,
            elem_classes = ["temporal-module-list", f"temporal-index-{ModuleList.index}"],
        )

        ModuleList.index += 1

    def __enter__(self, *args: Any, **kwargs: Any) -> "ModuleList":
        self._column.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._column.__exit__(*args, **kwargs)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield from self._dropdown.dependencies

    def read(self, data: ReadData) -> list[str]:
        return self._dropdown.read(data)

    def update(self, data: UpdateData) -> UpdateRequest:
        return self._dropdown.update(data)

    def setup_callback(self, callback: Callback) -> None:
        self._dropdown.setup_callback(callback)


class ModuleAccordion(Widget):
    index = 0

    def __init__(
        self,
        label: str = "",
        key: str = "",
        value: bool = False,
        open: bool = False,
    ) -> None:
        super().__init__()

        self._checkbox = GradioWidget(gr.Checkbox,
            label = self._format_label(label),
            value = value,
            elem_classes = ["temporal-module-accordion-checkbox", f"temporal-index-{ModuleAccordion.index}"],
        )

        self._accordion = GradioWidget(gr.Accordion,
            label = "",
            open = open,
            elem_classes = ["temporal-module-accordion", f"temporal-key-{key}", f"temporal-index-{ModuleAccordion.index}"],
        )

        ModuleAccordion.index += 1

    def __enter__(self, *args: Any, **kwargs: Any) -> "ModuleAccordion":
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


class ModuleAccordionSpecialCheckbox(Widget):
    def __init__(
        self,
        value: bool | Callable[[], bool] = False,
        classes: Iterable[str] = [],
    ) -> None:
        super().__init__()

        self._instance = GradioWidget(gr.Checkbox,
            value = value,
            elem_classes = list(classes) + ["temporal-module-accordion-special-checkbox"],
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
