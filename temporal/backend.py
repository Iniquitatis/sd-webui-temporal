from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from temporal.meta.serializable import Serializable
from temporal.utils.image import PILImage


class Backend:
    @property
    @abstractmethod
    def samplers(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def has_schedulers(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def schedulers(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_interrupted(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_done_saving(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process_images(self, preview: bool = False, **overrides: Any) -> Optional[list[PILImage]]:
        raise NotImplementedError

    @abstractmethod
    def process_batches(self, images: list[tuple[PILImage, int, int]], pixels_per_batch: int = 1048576, preview: bool = False, **overrides: Any) -> Optional[list[list[PILImage]]]:
        raise NotImplementedError

    @abstractmethod
    def set_preview_image(self, image: PILImage) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_image(self, image: PILImage, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
        raise NotImplementedError


class BackendData(Serializable):
    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def positive_prompt(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def negative_prompt(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def images(self) -> list[PILImage]:
        raise NotImplementedError

    @property
    @abstractmethod
    def width(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def height(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def denoising_strength(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def seed(self) -> int:
        raise NotImplementedError
