from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from inspect import isgeneratorfunction
from typing import Any, Callable, Iterator, TypeVar, Union, cast

import gradio as gr
from gradio.blocks import Block
from gradio.components import Component


GradioThing = Block | Component
UIThing = Union[GradioThing, "Widget"]

Properties = dict[str, Any]

ReadData = dict[UIThing, Any]
UpdateData = Properties
UpdateRequest = dict[UIThing, Properties]

CallbackInputs = dict[UIThing, Any]
CallbackOutputs = dict[UIThing, Properties]
CallbackFunc = Callable[[CallbackInputs], CallbackOutputs]
CallbackGenerator = Callable[[CallbackInputs], Iterator[CallbackOutputs]]

GradioCallbackInputs = dict[GradioThing, Any]
GradioCallbackOutputs = dict[GradioThing, Properties]
GradioCallbackFunc = Callable[[GradioCallbackInputs], GradioCallbackOutputs]
GradioCallbackGenerator = Callable[[GradioCallbackInputs], Iterator[GradioCallbackOutputs]]

T = TypeVar("T", bound = "Widget")
U = TypeVar("U", CallbackFunc, CallbackGenerator)


@dataclass
class Callback:
    event: str
    func: CallbackFunc | CallbackGenerator
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

        def gr_inputs_to_resolved(inputs: GradioCallbackInputs) -> CallbackInputs:
            return {
                thing: _read_value_recursively(thing, inputs)
                for thing in self.inputs
            }

        def resolved_to_gr_outputs(outputs: CallbackOutputs) -> GradioCallbackOutputs:
            result: GradioCallbackOutputs = {x: gr.update() for x in gr_outputs}

            for thing, properties in outputs.items():
                result |= _satisfy_update_request_recursively(thing, properties)

            return result

        if isgeneratorfunction(self.func):
            func = cast(CallbackGenerator, self.func)

            def gr_generator(inputs: GradioCallbackInputs) -> Iterator[GradioCallbackOutputs]:
                for outputs in func(gr_inputs_to_resolved(inputs)):
                    yield resolved_to_gr_outputs(outputs)

            event_registerer = getattr(component, self.event)
            event_registerer(gr_generator, gr_inputs, gr_outputs)

        else:
            func = cast(CallbackFunc, self.func)

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
        self.pending_callbacks: list[Callback] = []
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

    def setup_callback(self, callback: Callback) -> None:
        pass

    def callback(self, event: str, inputs: Iterable["Widget"], outputs: Iterable["Widget"]) -> Callable[[U], U]:
        def decorator(func: U) -> U:
            self.pending_callbacks.append(Callback(event, func, list(inputs), list(outputs)))
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
    def finalize(self, *widgets: Widget) -> list[GradioThing]:
        for widget in Widget._all:
            for callback in widget.pending_callbacks:
                widget.setup_callback(callback)

            widget.pending_callbacks.clear()

        self._final_widgets = list(widgets)
        self._final_components = [
            component
            for widget in self._final_widgets
            for component in _iter_components_recursively(widget)
        ]

        return self._final_components

    def recombine(self, *args: Any) -> list[Any]:
        values = {component: arg for component, arg in zip(self._final_components, args)}

        return [
            _read_value_recursively(widget, values)
            for widget in self._final_widgets
        ]


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
