from collections.abc import Iterable
from typing import Callable, Optional, TypeVar

import gradio as gr

from temporal.ui.selector import Selector


T = TypeVar("T")


class Radio(Selector[gr.Radio, T]):
    def __init__(
        self,
        label: str = "",
        choices: Iterable[str | tuple[T, str]] = [],
        value: Optional[T | Callable[[], T]] = None,
    ) -> None:
        super().__init__(gr.Radio, label = label, choices = choices, value = value)
