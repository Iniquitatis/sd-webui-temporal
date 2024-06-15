# NOTE: Hic sunt dracones
from collections.abc import Iterable
from typing import Any, Optional

import gradio as gr


class ModuleList(gr.Dropdown):
    index = 0

    def __init__(self, *args: Any, keys: Optional[Iterable[str]] = [], **kwargs: Any) -> None:
        kwargs.pop("label", "")
        classes = kwargs.pop("elem_classes", [])

        super().__init__(*args, **kwargs,
            label = "Order",
            multiselect = True,
            choices = list(keys) if keys is not None else None,
            value = list(keys) if keys is not None else None,
            elem_classes = classes + ["temporal-module-list-dropdown", f"temporal-index-{ModuleList.index}"],
        )
        self.change(None, self, None, _js = f"(x) => updateModuleListOrder({ModuleList.index}, x)")

        # NOTE: Necessary to communicate between the web interface and Python
        # (probably there's a better way to do this, but I haven't found it yet)
        self._textbox = gr.Textbox(*args, **kwargs,
            label = "Order",
            value = "|".join(keys) if keys is not None else None,
            elem_classes = classes + ["temporal-module-list-textbox", f"temporal-index-{ModuleList.index}"],
        )
        self._textbox.change(lambda x: gr.update(value = x.split("|")), self._textbox, self)

        self._column = gr.Column(*args, **kwargs,
            elem_classes = classes + ["temporal-module-list", f"temporal-index-{ModuleList.index}"],
        )

        ModuleList.index += 1

    def __enter__(self, *args: Any, **kwargs: Any) -> gr.Column:
        return self._column.__enter__(*args, **kwargs)

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        return self._column.__exit__(*args, **kwargs)

    def get_block_name(self) -> str:
        return "dropdown"


class ModuleAccordion(gr.Checkbox):
    index = 0

    def __init__(self, *args: Any, key: str, **kwargs: Any) -> None:
        label = kwargs.pop("label", "")
        classes = kwargs.pop("elem_classes", [])

        super().__init__(*args, **kwargs,
            label = label,
            elem_classes = classes + ["temporal-module-accordion-checkbox", f"temporal-index-{ModuleAccordion.index}"],
        )

        self._accordion = gr.Accordion(*args, **kwargs,
            label = "",
            elem_classes = classes + ["temporal-module-accordion", f"temporal-key-{key}", f"temporal-index-{ModuleAccordion.index}"],
        )

        ModuleAccordion.index += 1

    def __enter__(self, *args: Any, **kwargs: Any) -> gr.Accordion:
        return self._accordion.__enter__(*args, **kwargs)

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        return self._accordion.__exit__(*args, **kwargs)

    def get_block_name(self) -> str:
        return "checkbox"


class ModuleAccordionSpecialCheckbox(gr.Checkbox):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        classes = kwargs.pop("elem_classes", [])

        super().__init__(*args, **kwargs,
            elem_classes = classes + ["temporal-module-accordion-special-checkbox"],
        )

    def get_block_name(self) -> str:
        return "checkbox"
