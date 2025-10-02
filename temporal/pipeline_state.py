from typing import Generator, Type, Union

from modules.utils.image import NumpyImage


class PipelineState:
    progress: Type["_Progress"]
    finish: Type["_Finish"]
    fail: Type["_Fail"]


class _Progress(PipelineState):
    def __init__(self, image: NumpyImage, preview: bool) -> None:
        self.image = image
        self.preview = preview


class _Finish(PipelineState):
    def __init__(self, image: NumpyImage, preview: bool) -> None:
        self.image = image
        self.preview = preview


class _Fail(PipelineState):
    def __init__(self, message: str = "") -> None:
        self.message = message


PipelineState.progress = _Progress
PipelineState.finish = _Finish
PipelineState.fail = _Fail


PipelineResult = Generator[Union[_Progress, _Finish, _Fail], None, None]
