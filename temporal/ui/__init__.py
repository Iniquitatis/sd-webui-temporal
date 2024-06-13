from collections.abc import Iterable
from copy import copy
from inspect import isgeneratorfunction
from typing import Any, Callable, Iterator, Type, TypeVar

import gradio as gr
from gradio.blocks import Block
from gradio.components import Component

from temporal.utils.string import match_mask


CallbackFunc = Callable[[dict[str, Any]], dict[str, Any] | Iterator[dict[str, Any]]]

T = TypeVar("T", bound = Block | Component)


class UI:
    def __init__(self, id_formatter: Callable[[str], str]) -> None:
        self._id_formatter = id_formatter
        self._elems = {}
        self._ids = []
        self._groups = {}
        self._callbacks = {}
        self._existing_labels = set()

    def parse_ids(self, ids: Iterable[str]) -> list[str]:
        result = []

        for id in ids:
            if id.startswith("group:"):
                _, group = id.split(":")
                result.extend(x for x in self._ids if self.is_in_group(x, group))
            else:
                result.extend(x for x in self._ids if match_mask(x, id))

        return result

    def is_in_group(self, id: str, group: str) -> bool:
        return any(match_mask(x, group) for x in self._groups[id])

    def elem(self, id: str, constructor: Type[T], *args: Any, groups: list[str] = [], **kwargs: Any) -> T:
        def unique_label(string):
            if string in self._existing_labels:
                string = unique_label(string + " ")

            self._existing_labels.add(string)

            return string

        if "label" in kwargs:
            kwargs["label"] = unique_label(kwargs["label"])

        elem = constructor(*args, elem_id = self._id_formatter(id), **kwargs)

        if id:
            self._ids.append(id)
            self._elems[id] = elem
            self._groups[id] = ["all"] + groups
            self._callbacks[id] = []

        return elem

    def callback(self, id: str, event: str, inputs: Iterable[str], outputs: Iterable[str]) -> Callable[[CallbackFunc], CallbackFunc]:
        def decorator(func: CallbackFunc) -> CallbackFunc:
            self._callbacks[id].append((event, func, inputs, outputs))
            return func

        return decorator

    def finalize(self, ids: Iterable[str]) -> list[Any]:
        elems = copy(self._elems)
        elem_keys = {v: k for k, v in self._elems.items()}

        def make_wrapper_func(user_func, output_keys):
            if isgeneratorfunction(user_func):
                def generator_wrapper(inputs):
                    inputs_dict = {elem_keys[k]: v for k, v in inputs.items()}

                    for outputs_dict in user_func(inputs_dict):
                        yield {elems[x]: outputs_dict.get(x, gr.update()) for x in output_keys}

                return generator_wrapper

            else:
                def normal_wrapper(inputs):
                    inputs_dict = {elem_keys[k]: v for k, v in inputs.items()}
                    outputs_dict = user_func(inputs_dict)
                    return {elems[x]: outputs_dict.get(x, gr.update()) for x in output_keys}

                return normal_wrapper

        for id, callbacks in self._callbacks.items():
            for event, func, inputs, outputs in callbacks:
                input_keys = self.parse_ids(inputs)
                output_keys = self.parse_ids(outputs)

                event_func = getattr(self._elems[id], event)
                event_func(
                    make_wrapper_func(func, output_keys),
                    inputs = {self._elems[x] for x in input_keys},
                    outputs = {self._elems[x] for x in output_keys},
                )

        result = [self._elems[x] for x in self.parse_ids(ids)]

        self._id_formatter = lambda x: x
        self._elems.clear()
        self._existing_labels.clear()

        return result
