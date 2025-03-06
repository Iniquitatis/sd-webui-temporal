from typing import Optional

import skimage

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.image import NumpyImage, match_image
from temporal.utils.math import lerp, normalize
from temporal.utils.numpy import saturate_array


class ImageMask(Serializable):
    image: Optional[NumpyImage] = Field(None)
    normalized: bool = Field(False)
    inverted: bool = Field(False)
    blurring: float = Field(0.0)

    def mask(self, image: NumpyImage, other: NumpyImage) -> NumpyImage:
        if self.image is None or image is other:
            return other

        factor = match_image(self.image, image, channels = False)

        if self.normalized:
            factor = normalize(factor, factor.min(), factor.max())

        if self.inverted:
            factor = 1.0 - factor

        if self.blurring > 0.0:
            factor = saturate_array(skimage.filters.gaussian(factor, round(self.blurring), channel_axis = -1))

        return lerp(image, other, factor)
