import numpy as np

from temporal.serialization import load_object, save_object
from temporal.utils.fs import load_json, recreate_directory, save_json
from temporal.utils.image import ensure_image_dims, np_to_pil, pil_to_np
from temporal.utils.numpy import average_array, make_eased_weight_array, saturate_array

class ImageBuffer:
    def __init__(self, width, height, channels, count):
        self.array = np.zeros((count, height, width, channels))
        self.last_index = 0

    @property
    def width(self):
        return self.array.shape[2]

    @property
    def height(self):
        return self.array.shape[1]

    @property
    def channels(self):
        return self.array.shape[3]

    @property
    def count(self):
        return self.array.shape[0]

    def init(self, im):
        npim = self._convert_image_to_np(im)

        for i in range(self.count):
            self.array[i] = npim

    def add(self, im):
        self.array[self.last_index] = self._convert_image_to_np(im)

        self.last_index += 1
        self.last_index %= self.count

    def average(self, trimming = 0.0, easing = 0.0, preference = 0.0):
        return np_to_pil(self.array[0] if self.count == 1 else saturate_array(average_array(
            self.array,
            axis = 0,
            trim = trimming,
            power = preference + 1.0,
            weights = np.roll(make_eased_weight_array(self.count, easing), self.last_index),
        )))

    def load(self, path):
        if data := load_json(path / "data.json"):
            load_object(self, data, path)

    def save(self, path):
        recreate_directory(path)
        save_json(path / "data.json", save_object(self, path))

    def _convert_image_to_np(self, im):
        return pil_to_np(ensure_image_dims(
            im,
            "RGBA" if self.channels == 4 else "RGB",
            (self.width, self.height),
        ))
