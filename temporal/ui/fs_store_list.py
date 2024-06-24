from collections.abc import Iterable
from typing import Any, Generic, Iterator, Literal, TypeVar, cast

import gradio as gr

from modules.ui_components import ToolButton

from temporal.fs_store import FSStore
from temporal.ui import ReadData, ResolvedCallback, ResolvedCallbackInputs, ResolvedCallbackOutputs, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.utils.collection import get_first_element, get_next_element
from temporal.utils.object import copy_with_overrides


T = TypeVar("T", bound = FSStore[Any])


class FSStoreList(Widget, Generic[T]):
    def __init__(
        self,
        label: str,
        store: T,
        features: Iterable[Literal["load", "save", "rename", "delete"]] = ["load", "save", "rename", "delete"],
    ) -> None:
        super().__init__()

        self.store = store

        with GradioWidget(gr.Group):
            with GradioWidget(gr.Row):
                self._entry_name = GradioWidget(gr.Dropdown, label = self._format_label(label), choices = list(store.entry_names), allow_custom_value = True, value = get_first_element(store.entry_names, ""))
                self._refresh_entries = GradioWidget(ToolButton, value = "\U0001f504")
                self._load_entry = GradioWidget(ToolButton, value = "\U0001f4c2", visible = "load" in features)
                self._save_entry = GradioWidget(ToolButton, value = "\U0001f4be", visible = "save" in features)
                self._rename_entry = GradioWidget(ToolButton, value = "\u270f\ufe0f", visible = "rename" in features)
                self._delete_entry = GradioWidget(ToolButton, value = "\U0001f5d1\ufe0f", visible = "delete" in features)

            with GradioWidget(gr.Row, visible = False) as self._rename_row:
                self._new_entry_name = GradioWidget(gr.Textbox, label = self._format_label(label, "New name"), value = "")
                self._confirm_rename = GradioWidget(ToolButton, value = "\U00002714\ufe0f")
                self._deny_rename = GradioWidget(ToolButton, value = "\U0000274c")

        @self._refresh_entries.callback("click", [], [self._entry_name])
        def _(_: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
            store.refresh()

            return {self._entry_name: {"choices": store.entry_names}}

        @self._rename_entry.callback("click", [self._entry_name], [self._rename_row, self._new_entry_name])
        def _(inputs: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
            return {
                self._rename_row: {"visible": True},
                self._new_entry_name: {"value": inputs[self._entry_name]},
            }

        @self._confirm_rename.callback("click", [self._entry_name, self._new_entry_name], [self._entry_name, self._rename_row])
        def _(inputs: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
            entry_name = inputs[self._entry_name]
            new_entry_name = inputs[self._new_entry_name]

            if entry_name not in store.entry_names:
                return {}

            store.rename_entry(entry_name, new_entry_name)

            return {
                self._entry_name: {"choices": store.entry_names, "value": new_entry_name},
                self._rename_row: {"visible": False},
            }

        @self._deny_rename.callback("click", [], [self._rename_row])
        def _(_: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
            return {self._rename_row: {"visible": False}}

        @self._delete_entry.callback("click", [self._entry_name], [self._entry_name])
        def _(inputs: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
            entry_name = inputs[self._entry_name]

            if entry_name not in store.entry_names:
                return {}

            new_name = get_next_element(store.entry_names, entry_name, "untitled")

            store.delete_entry(entry_name)

            return {self._entry_name: {"choices": store.entry_names, "value": new_name}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._entry_name

    def read(self, data: ReadData) -> str:
        return data[self._entry_name]

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), str):
            result[self._entry_name] = {"value": value}

        return result

    def setup_callback(self, callback: ResolvedCallback) -> None:
        if callback.event == "refresh":
            self._refresh_entries.setup_callback(copy_with_overrides(callback, event = "click"))

        elif callback.event == "load":
            def func(inputs: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
                entry_name = inputs.pop(self._entry_name)

                if entry_name not in self.store.entry_names:
                    return {}

                if self in inputs:
                    inputs[self] = self.store.load_entry(entry_name)

                return cast(ResolvedCallbackOutputs, callback.func(inputs))

            self._load_entry.setup_callback(ResolvedCallback("click", func, [self._entry_name] + callback.inputs, callback.outputs))

        elif callback.event == "save":
            def func(inputs: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
                entry_name = inputs.pop(self._entry_name)

                outputs = cast(ResolvedCallbackOutputs, callback.func(inputs))

                if (entry := outputs.pop(self, {}).pop("value", None)) is not None:
                    self.store.save_entry(entry_name, entry)

                return {self._entry_name: {"choices": self.store.entry_names, "value": entry_name}} | outputs

            self._save_entry.setup_callback(ResolvedCallback("click", func, [self._entry_name] + callback.inputs, [self._entry_name] + callback.outputs))

        elif callback.event == "rename":
            self._confirm_rename.setup_callback(copy_with_overrides(callback, event = "click"))

        elif callback.event == "delete":
            self._delete_entry.setup_callback(copy_with_overrides(callback, event = "click"))

        else:
            self._entry_name.setup_callback(callback)
