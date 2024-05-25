from pathlib import Path

import numpy as np

from temporal.serialization import load_object, save_object
from temporal.utils.fs import load_json, recreate_directory, save_json
from temporal.utils.image import NumpyImage, PILImage, ensure_image_dims, np_to_pil, pil_to_np
from temporal.utils.numpy import average_array, make_eased_weight_array, saturate_array


class ImageBuffer:
    def __init__(self, width: int, height: int, channels: int, count: int) -> None:
        self.array = np.zeros((count, height, width, channels))
        self.last_index = 0

    @property
    def width(self) -> int:
        return self.array.shape[2]

    @property
    def height(self) -> int:
        return self.array.shape[1]

    @property
    def channels(self) -> int:
        return self.array.shape[3]

    @property
    def count(self) -> int:
        return self.array.shape[0]

    def init(self, im: PILImage) -> None:
        npim = self._convert_image_to_np(im)

        for i in range(self.count):
            self.array[i] = npim

    def add(self, im: PILImage) -> None:
        self.array[self.last_index] = self._convert_image_to_np(im)

        self.last_index += 1
        self.last_index %= self.count

    def average(self, trimming: float = 0.0, easing: float = 0.0, preference: float = 0.0) -> PILImage:
        return np_to_pil(self.array[0] if self.count == 1 else saturate_array(average_array(
            self.array,
            axis = 0,
            trim = trimming,
            power = preference + 1.0,
            weights = np.roll(make_eased_weight_array(self.count, easing), self.last_index),
        )))

    def load(self, path: Path) -> None:
        if data := load_json(path / "data.json"):
            load_object(self, data, path)

    def save(self, path: Path) -> None:
        recreate_directory(path)
        save_json(path / "data.json", save_object(self, path))

    def _convert_image_to_np(self, im: PILImage) -> NumpyImage:
        return pil_to_np(ensure_image_dims(
            im,
            "RGBA" if self.channels == 4 else "RGB",
            (self.width, self.height),
        ))
