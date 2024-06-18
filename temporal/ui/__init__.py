from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from inspect import isgeneratorfunction
from typing import Any, Callable, Iterator, Optional, Type, TypeVar

import gradio as gr
from gradio.blocks import Block
from gradio.components import Component

from temporal.utils.string import match_mask


CallbackFunc = Callable[[dict[str, Any]], dict[str, Any] | Iterator[dict[str, Any]]]
ProcessorFunc = Callable[[Any], Any]

T = TypeVar("T", bound = Block | Component)


@dataclass
class Callback:
    event: str
    func: CallbackFunc
    inputs: list[str]
    outputs: list[str]


class UIElement:
    def __init__(
        self,
        id: str,
        gr_type: Type[Block | Component],
        gr_args: Iterable[Any],
        gr_kwargs: Mapping[str, Any],
        preprocessor: Optional[ProcessorFunc],
        postprocessor: Optional[ProcessorFunc],
        groups: Iterable[str] = []
    ) -> None:
        self.id = id
        self.instance: Any = gr_type(*gr_args, **gr_kwargs)
        self.preprocessor = preprocessor or (lambda x: x)
        self.postprocessor = postprocessor or (lambda x: x)
        self.groups: list[str] = ["all"] + list(groups)
        self.callbacks: list[Callback] = []


class UI:
    def __init__(self, id_formatter: Callable[[str], str]) -> None:
        self._id_formatter = id_formatter
        self._elems: dict[str, UIElement] = {}
        self._existing_labels: set[str] = set()

    @property
    def ids(self) -> Iterable[str]:
        return self._elems.keys()

    def parse_ids(self, ids: Iterable[str]) -> list[str]:
        result = []

        for id in ids:
            if id.startswith("group:"):
                _, group = id.split(":")
                result.extend(x for x in self.ids if self.is_in_group(x, group))
            else:
                result.extend(x for x in self.ids if match_mask(x, id))

        return result

    def is_in_group(self, id: str, group: str) -> bool:
        return any(match_mask(x, group) for x in self._elems[id].groups)

    def elem(
        self,
        id: str,
        gr_type: Type[T],
        *gr_args: Any,
        preprocessor: Optional[ProcessorFunc] = None,
        postprocessor: Optional[ProcessorFunc] = None,
        groups: Iterable[str] = [],
        **gr_kwargs: Any,
    ) -> T:
        def unique_label(string: str) -> str:
            if string in self._existing_labels:
                string = unique_label(string + " ")

            self._existing_labels.add(string)

            return string

        if "label" in gr_kwargs:
            gr_kwargs["label"] = unique_label(gr_kwargs["label"])

        if "value" in gr_kwargs and postprocessor:
            value = gr_kwargs["value"]

            if callable(value):
                new_value = lambda: postprocessor(value())
            else:
                new_value = postprocessor(value)

            gr_kwargs["value"] = new_value

        if id:
            gr_kwargs["elem_id"] = self._id_formatter(id)

        elem = UIElement(id, gr_type, gr_args, gr_kwargs, preprocessor, postprocessor, groups)

        if id:
            self._elems[id] = elem

        return elem.instance

    def callback(self, id: str, event: str, inputs: Iterable[str], outputs: Iterable[str]) -> Callable[[CallbackFunc], CallbackFunc]:
        def decorator(func: CallbackFunc) -> CallbackFunc:
            self._elems[id].callbacks.append(Callback(event, func, list(inputs), list(outputs)))
            return func

        return decorator

    def finalize(self, ids: Iterable[str]) -> list[Block | Component]:
        instances_to_ids = {v.instance: k for k, v in self._elems.items()}

        def make_wrapper_func(user_func, output_keys):
            def unpack_inputs(inputs):
                return {
                    id: self._elems[id].preprocessor(v)
                    for k, v in inputs.items()
                    if (id := instances_to_ids[k])
                }

            def pack_outputs(outputs):
                def postprocess(elem, update):
                    if "value" in update:
                        update["value"] = elem.postprocessor(update["value"])

                    return update

                return {
                    elem.instance: postprocess(elem, outputs.get(x, gr.update()))
                    for x in output_keys
                    if (elem := self._elems[x])
                }

            if isgeneratorfunction(user_func):
                def generator_wrapper(inputs):
                    for outputs_dict in user_func(unpack_inputs(inputs)):
                        yield pack_outputs(outputs_dict)

                return generator_wrapper

            else:
                def normal_wrapper(inputs):
                    return pack_outputs(user_func(unpack_inputs(inputs)))

                return normal_wrapper

        for elem in self._elems.values():
            for callback in elem.callbacks:
                input_keys = self.parse_ids(callback.inputs)
                output_keys = self.parse_ids(callback.outputs)

                event_func = getattr(elem.instance, callback.event)
                event_func(
                    make_wrapper_func(callback.func, output_keys),
                    inputs = {self._elems[x].instance for x in input_keys},
                    outputs = {self._elems[x].instance for x in output_keys},
                )

        return [self._elems[x].instance for x in self.parse_ids(ids)]
