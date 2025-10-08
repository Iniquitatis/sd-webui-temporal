from typing import Optional

import numpy as np
from scipy.ndimage import convolve

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage, apply_channelwise, ensure_image_dims
from modules.utils.numpy import FloatArray, FloatType, IntType, saturate_array


class ConvolutionFilter(ImageFilter):
    name = "Convolution"

    kernel: Optional[NumpyImage] = Field(None, name = "Kernel", channels = 1)
    radius: float = Field(0.0, name = "Radius", minimum = 0.0, maximum = 50.0, step = 0.1, display = "slider")
    control: Optional[NumpyImage] = Field(None, name = "Control mask", channels = 1)

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        def prepare_kernel(radius: int) -> FloatArray:
            diameter = 1 + radius * 2

            kernel = self.kernel

            if kernel is None:
                kernel = np.ones((diameter, diameter), dtype = FloatType)

            kernel = ensure_image_dims(kernel, (diameter, diameter))
            kernel = kernel[..., 0]
            kernel /= np.sum(kernel)

            return kernel

        radius = round(self.radius)

        levels = list(range(radius + 1)) if self.control is not None else [radius]

        results = []

        for radius in levels:
            kernel = prepare_kernel(radius)
            result = apply_channelwise(image, lambda x: convolve(x, kernel, mode = "mirror"))
            results.append(result)

        control = self.control

        if control is None:
            control = np.ones_like(image)

        control = ensure_image_dims(control, (image.shape[1], image.shape[0]))
        control = control[..., 0]

        result = _interpolate_stack(np.stack(results, axis = 0), control)

        return saturate_array(result)


def _interpolate_stack(stack: FloatArray, x: FloatArray) -> NumpyImage:
    count = stack.shape[0]

    if count == 1:
        return stack[0].copy()

    height, width = stack.shape[1:3]

    layer_indices = x * (count - 1)

    base_layer_indices = np.floor(layer_indices).astype(IntType)
    next_layer_indices = np.minimum(base_layer_indices + 1, count - 1)

    layer_factor = layer_indices - base_layer_indices

    y_coords, x_coords = np.ogrid[:height, :width]

    base_values = stack[base_layer_indices, y_coords, x_coords]
    next_values = stack[next_layer_indices, y_coords, x_coords]

    return base_values + (next_values - base_values) * layer_factor[..., None]
