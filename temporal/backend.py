from abc import ABC, abstractmethod
from collections.abc import Iterable
from math import ceil
from pathlib import Path
from typing import Optional

from temporal.general_data import GeneralData
from temporal.processing_params import ProcessingParams
from temporal.utils.collection import batched
from temporal.utils.image import NumpyImage


class Backend(ABC):
    @abstractmethod
    def list_models(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def list_vaes(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def list_upscalers(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def list_samplers(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def list_schedulers(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def image_to_image(self, images: list[NumpyImage], params: ProcessingParams, width: int, height: int, preview: bool = False) -> Optional[list[NumpyImage]]:
        raise NotImplementedError

    @abstractmethod
    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
        raise NotImplementedError

    @abstractmethod
    def get_preview(self) -> Optional[NumpyImage]:
        raise NotImplementedError

    @abstractmethod
    def set_preview(self, image: Optional[NumpyImage] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_image(self, image: NumpyImage, general: GeneralData, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def are_images_saved(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def interrupt(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_interrupted(self) -> bool:
        raise NotImplementedError

    def image_to_image_batched(self, images: list[NumpyImage], params: ProcessingParams, width: int, height: int, pixels_per_batch: int = 1048576, preview: bool = False) -> Optional[list[NumpyImage]]:
        first_image = images[0]
        pixels_per_image = first_image.shape[1] * first_image.shape[0]
        batch_size = ceil(pixels_per_batch / pixels_per_image)

        result = []

        for batch in batched(images, batch_size):
            if processed_images := self.image_to_image(batch, params, width, height, preview):
                result.extend(processed_images)
            else:
                return None

        return result
