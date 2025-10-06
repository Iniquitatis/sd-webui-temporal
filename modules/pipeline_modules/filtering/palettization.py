from functools import lru_cache
from typing import Optional

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage, apply_channelwise
from modules.utils.numpy import FloatArray, FloatType, stretch_array


class PalettizationFilter(ImageFilter):
    name = "Palettization"

    palette: Optional[NumpyImage] = Field(None, name = "Palette", channels = 3)
    stretch: bool = Field(False, name = "Stretch")
    dithering: str = Field("none", name = "Dithering", choices = {"none": "None", "bayer": "Bayer", "blue_noise": "Blue noise", "floyd_steinberg": "Floyd-Steinberg"}, display = "radio")
    matrix_exponent: int = Field(1, name = "Matrix exponent", minimum = 1, maximum = 5, step = 1, dependencies = {"dithering": ["bayer", "blue_noise"]}, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        if self.palette is None:
            return image

        palette = self.palette.reshape((self.palette.shape[1] * self.palette.shape[0], 3))

        if self.stretch:
            palette = apply_channelwise(palette, lambda x: stretch_array(x, 256))

        if self.dithering == "none":
            return _quantize(image, palette)
        elif self.dithering == "bayer":
            return _dither_ordered(image, palette, _create_bayer_matrix(2 ** self.matrix_exponent))
        elif self.dithering == "blue_noise":
            return _dither_ordered(image, palette, _create_blue_noise_matrix(2 ** self.matrix_exponent))
        elif self.dithering == "floyd_steinberg":
            return _dither_floyd_steinberg(image, palette)
        else:
            raise NotImplementedError


def _quantize(image: NumpyImage, palette: FloatArray) -> NumpyImage:
    height, width, channels = image.shape

    flat_image = image.reshape(-1, channels)
    _, indices = KDTree(palette).query(flat_image, k = 1)

    return palette[indices].reshape(height, width, channels)


def _dither_ordered(image: NumpyImage, palette: FloatArray, matrix: FloatArray) -> NumpyImage:
    height, width, channels = image.shape

    y, x = np.indices((height, width))

    stride = matrix.shape[0]

    dither = matrix[y % stride, x % stride] - 0.5
    dither *= pdist(palette, metric = "euclidean").min()
    dither = np.stack([dither] * channels, axis = -1)

    perturbed = np.clip(image + dither, 0.0, 1.0)

    flat_perturbed = perturbed.reshape(-1, channels)
    _, indices = KDTree(palette).query(flat_perturbed, k = 1)

    return palette[indices].reshape(height, width, channels)


@njit
def _dither_floyd_steinberg(image: NumpyImage, palette: FloatArray) -> NumpyImage:
    C1 = 7 / 16
    C2 = 3 / 16
    C3 = 5 / 16
    C4 = 1 / 16

    height, width, channels = image.shape
    palette_length = palette.shape[0]

    padded = np.zeros((height + 2, width + 2, channels), dtype = FloatType)
    padded[1:height + 1, 1:width + 1] = image

    output = np.empty_like(image)

    # NOTE: The biggest WTF to date. Looping order must be YX according to the
    # original algorithm, but here it only works with XY. Maybe it's because the
    # algorithm expects arrays to be in WH layout, while in numpy it's HW?
    for x in range(width):
        for y in range(height):
            old_pixel = padded[y + 1, x + 1]

            best_index = 0
            minimum_distance = np.inf

            for i in range(palette_length):
                distance = 0.0

                for channel in range(channels):
                    difference = old_pixel[channel] - palette[i, channel]
                    distance += difference * difference

                if distance < minimum_distance:
                    best_index = i
                    minimum_distance = distance

            new_pixel = palette[best_index]
            output[y, x] = new_pixel

            quant_error = old_pixel - new_pixel

            for channel in range(channels):
                padded[y + 1, x + 2, channel] += quant_error[channel] * C1
                padded[y + 2, x    , channel] += quant_error[channel] * C2
                padded[y + 2, x + 1, channel] += quant_error[channel] * C3
                padded[y + 2, x + 2, channel] += quant_error[channel] * C4

    return output


@lru_cache
def _create_bayer_matrix(size: int, normalize: bool = True) -> FloatArray:
    if size == 1:
        return np.zeros((1, 1), dtype = FloatType)

    quadrant = _create_bayer_matrix(size // 2, normalize = False)

    matrix = np.block([
        [4 * quadrant + 0, 4 * quadrant + 2],
        [4 * quadrant + 3, 4 * quadrant + 1],
    ])

    return matrix / (size * size) if normalize else matrix


@lru_cache
def _create_blue_noise_matrix(size: int) -> FloatArray:
    if size == 1:
        return np.zeros((1, 1), dtype = FloatType)

    cell_count = size * size
    sigma = max(1.0, size / 32.0)
    pattern = np.zeros((size, size), dtype = FloatType)
    threshold_map = np.full((size, size), np.inf, dtype = FloatType)

    def place_next(phase_sign: float) -> tuple[np.intp, ...]:
        target_value = (1.0 + phase_sign) / 2.0
        write_value = (1.0 - phase_sign) / 2.0

        candidates = pattern == target_value
        activation = gaussian_filter(pattern, sigma = sigma, mode = "wrap") * phase_sign
        scores = np.where(candidates, activation, -np.inf)
        index = np.unravel_index(np.argmax(scores), pattern.shape)

        pattern[index] = write_value

        return index

    for _ in range(cell_count // 2):
        place_next(-1.0)

    for _ in range(cell_count // 2):
        place_next(1.0)

    for i in range(cell_count):
        index = place_next(-1.0)
        threshold_map[index] = i

    return threshold_map / (cell_count - 1)
