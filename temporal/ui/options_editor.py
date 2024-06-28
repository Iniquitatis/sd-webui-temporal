from typing import Iterator

from temporal.global_options import GlobalOptions
from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.option_category_editor import OptionCategoryEditor


class OptionsEditor(Widget):
    def __init__(
        self,
        value: GlobalOptions = GlobalOptions(),
    ) -> None:
        super().__init__()

        self._categories = {
            key: OptionCategoryEditor(value = getattr(value, key))
            for key in value.__fields__.keys()
        }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield from self._categories.values()

    def read(self, data: ReadData) -> GlobalOptions:
        return GlobalOptions(**{key: data[widget] for key, widget in self._categories.items()})

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), GlobalOptions):
            result |= {widget: {"value": getattr(value, key)} for key, widget in self._categories.items()}

        return result
