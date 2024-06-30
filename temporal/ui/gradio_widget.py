from typing import Any, Callable, Generic, Iterator, ParamSpec, TypeVar, cast

from temporal.ui import Callback, GradioThing, ReadData, UIThing, UpdateData, UpdateRequest, Widget


T = TypeVar("T", bound = GradioThing)
P = ParamSpec("P")


class GradioWidget(Widget, Generic[T]):
    def __init__(
        self,
        gr_type: Callable[P, T],
        *gr_args: P.args,
        **gr_kwargs: P.kwargs,
    ) -> None:
        super().__init__()

        if "label" in gr_kwargs:
            gr_kwargs["label"] = self._format_label(cast(str, gr_kwargs["label"]))

        self._instance = gr_type(*gr_args, **gr_kwargs)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._instance

    def read(self, data: ReadData) -> Any:
        return data[self._instance]

    def update(self, data: UpdateData) -> UpdateRequest:
        return {self._instance: data}

    def setup_callback(self, callback: Callback) -> None:
        callback.apply_to_component(self._instance)

    def __enter__(self, *args: Any, **kwargs: Any) -> "GradioWidget[T]":
        if enter := getattr(self._instance, "__enter__", None):
            enter(*args, **kwargs)
            return self
        else:
            raise NotImplementedError

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if exit := getattr(self._instance, "__exit__", None):
            exit(*args, **kwargs)
        else:
            raise NotImplementedError
