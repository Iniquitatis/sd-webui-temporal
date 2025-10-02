from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

from modules.processing_params import ProcessingParams
from modules.utils.image import NumpyImage


class Backend(ABC):
    @abstractmethod
    def list_models(self) -> Iterable[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def list_vaes(self) -> Iterable[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def list_upscalers(self) -> Iterable[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def list_samplers(self) -> Iterable[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def list_schedulers(self) -> Iterable[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def image_to_image(self, image: NumpyImage, params: ProcessingParams, width: int, height: int) -> Optional[NumpyImage]:
        raise NotImplementedError

    @abstractmethod
    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
        raise NotImplementedError

    @abstractmethod
    def interrupt(self) -> None:
        raise NotImplementedError
