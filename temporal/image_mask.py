from typing import Optional

import skimage

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.math import lerp, normalize


class ImageMask(Serializable):
    image: Optional[NumpyImage] = Field(None)
    normalized: bool = Field(False)
    inverted: bool = Field(False)
    blurring: float = Field(0.0)

    def mask(self, npim: NumpyImage, processed: NumpyImage) -> NumpyImage:
        if self.image is None or npim is processed:
            return processed

        factor = ensure_image_dims(self.image, size = (npim.shape[1], npim.shape[0]))

        if self.normalized:
            factor = normalize(factor, factor.min(), factor.max())

        if self.inverted:
            factor = 1.0 - factor

        if self.blurring > 0.0:
            factor = skimage.filters.gaussian(factor, round(self.blurring), channel_axis = -1)

        return lerp(npim, processed, factor)
