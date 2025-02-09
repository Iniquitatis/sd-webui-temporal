from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from math import ceil
from pathlib import Path
from typing import Optional

from temporal.processing_params import ImageToImageParams, TextToImageParams
from temporal.project import Project
from temporal.utils.collection import batched
from temporal.utils.image import NumpyImage
from temporal.utils.object import copy_with_overrides


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
    def text_to_image(self, params: TextToImageParams, preview: bool = False) -> Optional[list[NumpyImage]]:
        raise NotImplementedError

    @abstractmethod
    def image_to_image(self, params: ImageToImageParams, preview: bool = False) -> Optional[list[NumpyImage]]:
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
    def save_image(self, image: NumpyImage, project: Project, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
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

    def images_to_batches(self, params: ImageToImageParams, images: list[tuple[NumpyImage, int, int]], pixels_per_batch: int = 1048576, preview: bool = False) -> Optional[list[list[NumpyImage]]]:
        first_image, _, _ = images[0]
        pixels_per_image = first_image.shape[1] * first_image.shape[0]
        batch_size = ceil(pixels_per_batch / pixels_per_image)

        result = defaultdict(list)

        for batch in batched((
            (image_index, image, image_seed)
            for image_index, (image, starting_seed, count) in enumerate(images)
            for image_seed, _ in enumerate(range(count), starting_seed)
        ), batch_size):
            if not (processed_images := self.image_to_image(copy_with_overrides(params,
                images = [image for _, image, _ in batch],
                seeds = [seed for _, _, seed in batch],
            ), preview)):
                return None

            for (image_index, _, _), image in zip(batch, processed_images[:len(batch)]):
                result[image_index].append(image)

        return list(result.values())
