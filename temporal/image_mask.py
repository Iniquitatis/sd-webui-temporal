from typing import Optional

import skimage

from temporal.object import Field, Object
from temporal.utils.image import NumpyImage, match_image
from temporal.utils.math import lerp, normalize
from temporal.utils.numpy import saturate_array


class ImageMask(Object):
    image: Optional[NumpyImage] = Field(None, name = "Image")
    normalized: bool = Field(False, name = "Normalized")
    inverted: bool = Field(False, name = "Inverted")
    blurring: float = Field(0.0, name = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, display = "slider")

    def mask(self, image: NumpyImage, other: NumpyImage) -> NumpyImage:
        if self.image is None or image is other:
            return other

        factor = match_image(self.image, image)

        if self.normalized:
            factor = normalize(factor, factor.min(), factor.max())

        if self.inverted:
            factor = 1.0 - factor

        if self.blurring > 0.0:
            factor = saturate_array(skimage.filters.gaussian(factor, round(self.blurring), channel_axis = -1))

        return lerp(image, other, factor)
