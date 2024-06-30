from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from inspect import isgeneratorfunction
from typing import Any, Callable, Iterator, TypeVar, Union, cast

import gradio as gr
from gradio.blocks import Block
from gradio.components import Component

from temporal.utils.string import match_mask


GradioThing = Block | Component
UIThing = Union[GradioThing, "Widget"]

Properties = dict[str, Any]

ReadData = dict[UIThing, Any]
UpdateData = Properties
UpdateRequest = dict[UIThing, dict[str, Any]]

CallbackInputs = dict[str, Any]
CallbackOutputs = dict[str, Properties]
CallbackFunc = Callable[[CallbackInputs], CallbackOutputs]
CallbackGenerator = Callable[[CallbackInputs], Iterator[CallbackOutputs]]

ResolvedCallbackInputs = dict[UIThing, Any]
ResolvedCallbackOutputs = dict[UIThing, Properties]
ResolvedCallbackFunc = Callable[[ResolvedCallbackInputs], ResolvedCallbackOutputs]
ResolvedCallbackGenerator = Callable[[ResolvedCallbackInputs], Iterator[ResolvedCallbackOutputs]]

GradioCallbackInputs = dict[GradioThing, Any]
GradioCallbackOutputs = dict[GradioThing, Properties]
GradioCallbackFunc = Callable[[GradioCallbackInputs], GradioCallbackOutputs]
GradioCallbackGenerator = Callable[[GradioCallbackInputs], Iterator[GradioCallbackOutputs]]

T = TypeVar("T", bound = "Widget")
U = TypeVar("U", CallbackFunc, CallbackGenerator)
V = TypeVar("V", ResolvedCallbackFunc, ResolvedCallbackGenerator)


@dataclass
class Callback:
    event: str
    func: CallbackFunc | CallbackGenerator
    inputs: list[str]
    outputs: list[str]

    def resolve(self, resolver: Callable[[str], Iterator["Widget"]]) -> "ResolvedCallback":
        resolved_inputs = list(widget for id in self.inputs for widget in resolver(id))
        resolved_outputs = list(widget for id in self.outputs for widget in resolver(id))

        output_widgets_by_ids = {widget.id: widget for widget in resolved_outputs}

        def resolved_to_inputs(inputs: ResolvedCallbackInputs) -> CallbackInputs:
            return {
                thing.id: value
                for thing, value in inputs.items()
                if isinstance(thing, Widget)
            }

        def outputs_to_resolved(outputs: CallbackOutputs) -> ResolvedCallbackOutputs:
            return {
                output_widgets_by_ids[widget_id]: request
                for widget_id, request in outputs.items()
            }

        if isgeneratorfunction(self.func):
            func = cast(CallbackGenerator, self.func)

            def resolved_generator(inputs: ResolvedCallbackInputs) -> Iterator[ResolvedCallbackOutputs]:
                for outputs in func(resolved_to_inputs(inputs)):
                    yield outputs_to_resolved(outputs)

            return ResolvedCallback(self.event, resolved_generator, list(resolved_inputs), list(resolved_outputs))

        else:
            func = cast(CallbackFunc, self.func)

            def resolved_func(inputs: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
                return outputs_to_resolved(func(resolved_to_inputs(inputs)))

            return ResolvedCallback(self.event, resolved_func, list(resolved_inputs), list(resolved_outputs))


@dataclass
class ResolvedCallback:
    event: str
    func: ResolvedCallbackFunc | ResolvedCallbackGenerator
    inputs: list[UIThing]
    outputs: list[UIThing]

    def apply_to_component(self, component: GradioThing) -> None:
        gr_inputs = {
            component
            for thing in self.inputs
            for component in _iter_components_recursively(thing)
        }
        gr_outputs = {
            component
            for thing in self.outputs
            for component in _iter_components_recursively(thing)
        }

        def gr_inputs_to_resolved(inputs: GradioCallbackInputs) -> ResolvedCallbackInputs:
            return {
                thing: _read_value_recursively(thing, inputs)
                for thing in self.inputs
            }

        def resolved_to_gr_outputs(outputs: ResolvedCallbackOutputs) -> GradioCallbackOutputs:
            result: dict[GradioThing, dict[str, Any]] = {x: gr.update() for x in gr_outputs}

            for thing, properties in outputs.items():
                result |= _satisfy_update_request_recursively(thing, properties)

            return result

        if isgeneratorfunction(self.func):
            func = cast(ResolvedCallbackGenerator, self.func)

            def gr_generator(inputs: GradioCallbackInputs) -> Iterator[GradioCallbackOutputs]:
                for outputs in func(gr_inputs_to_resolved(inputs)):
                    yield resolved_to_gr_outputs(outputs)

            event_registerer = getattr(component, self.event)
            event_registerer(gr_generator, gr_inputs, gr_outputs)

        else:
            func = cast(ResolvedCallbackFunc, self.func)

            def gr_func(inputs: GradioCallbackInputs) -> GradioCallbackOutputs:
                return resolved_to_gr_outputs(func(gr_inputs_to_resolved(inputs)))

            event_registerer = getattr(component, self.event)
            event_registerer(gr_func, gr_inputs, gr_outputs)


class Widget(ABC):
    _existing_labels: set[str] = set()
    _all: list["Widget"] = []

    def __init__(
        self,
    ) -> None:
        self.id = f"__widget_{len(self._all)}__"
        self.groups: list[str] = ["all"]
        self.callbacks: list[Callback] = []
        self.resolved_callbacks: list[ResolvedCallback] = []
        self._all.append(self)

    @property
    @abstractmethod
    def dependencies(self) -> Iterator[UIThing]:
        raise NotImplementedError

    @abstractmethod
    def read(self, data: ReadData) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update(self, data: UpdateData) -> UpdateRequest:
        raise NotImplementedError

    def setup_callback(self, callback: ResolvedCallback) -> None:
        pass

    def callback(self, event: str, inputs: Iterable["Widget"], outputs: Iterable["Widget"]) -> Callable[[V], V]:
        def decorator(func: V) -> V:
            self.resolved_callbacks.append(ResolvedCallback(event, func, list(inputs), list(outputs)))
            return func

        return decorator

    def _format_label(self, label: str, suffix: str = "") -> str:
        def unique_label(string: str) -> str:
            if string in self._existing_labels:
                string = unique_label(string + " ")

            self._existing_labels.add(string)

            return string

        if label and suffix:
            new_label = f"{label}: {suffix}"
        elif label and not suffix:
            new_label = label
        elif not label and suffix:
            new_label = suffix
        else:
            new_label = ""

        return unique_label(new_label)


class UI:
    def __init__(self) -> None:
        self._widgets: dict[str, Widget] = {}

    def parse_id(self, id: str) -> Iterator[str]:
        if id.startswith("group:"):
            _, group = id.split(":")

            for widget_id in self._widgets.keys():
                if self.is_in_group(widget_id, group):
                    yield widget_id

        else:
            for widget_id in self._widgets.keys():
                if match_mask(widget_id, id):
                    yield widget_id

    def resolve_id(self, id: str) -> Iterator[Widget]:
        for id in self.parse_id(id):
            yield self._widgets[id]

    def is_in_group(self, id: str, group: str) -> bool:
        return any(match_mask(x, group) for x in self._widgets[id].groups)

    def add(self, id: str, widget: T, groups: Iterable[str] = []) -> T:
        widget.id = id if id else widget.id
        widget.groups.extend(groups)
        self._widgets[widget.id] = widget
        return widget

    def callback(self, id: str, event: str, inputs: Iterable[str], outputs: Iterable[str]) -> Callable[[U], U]:
        def decorator(func: U) -> U:
            self._widgets[id].callbacks.append(Callback(event, func, list(inputs), list(outputs)))
            return func

        return decorator

    def finalize(self, ids: Iterable[str]) -> list[GradioThing]:
        for widget in Widget._all:
            for callback in widget.callbacks:
                widget.resolved_callbacks.append(callback.resolve(self.resolve_id))

            widget.callbacks.clear()

            for callback in widget.resolved_callbacks:
                widget.setup_callback(callback)

            widget.resolved_callbacks.clear()

        self._final_widgets = [
            widget
            for id in ids
            for widget in self.resolve_id(id)
        ]
        self._final_components = [
            component
            for widget in self._final_widgets
            for component in _iter_components_recursively(widget)
        ]

        return self._final_components

    def recombine(self, *args: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}

        values = {component: arg for component, arg in zip(self._final_components, args)}

        for widget in self._final_widgets:
            result[widget.id] = _read_value_recursively(widget, values)

        return result


def _iter_components_recursively(thing: UIThing) -> Iterator[GradioThing]:
    if isinstance(thing, GradioThing):
        yield thing
    else:
        for dependency in thing.dependencies:
            yield from _iter_components_recursively(dependency)


def _read_value_recursively(thing: UIThing, inputs: GradioCallbackInputs) -> Any:
    if isinstance(thing, GradioThing):
        return inputs[thing]
    else:
        return thing.read({
            dependency: _read_value_recursively(dependency, inputs)
            for dependency in thing.dependencies
        })


def _satisfy_update_request_recursively(thing: UIThing, properties: Properties) -> GradioCallbackOutputs:
    if isinstance(thing, GradioThing):
        return {thing: gr.update(**properties)}
    else:
        result: GradioCallbackOutputs = {}

        for other_thing, other_properties in thing.update(properties).items():
            result |= _satisfy_update_request_recursively(other_thing, other_properties)

        return result
