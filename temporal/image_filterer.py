import skimage

from temporal.blend_modes import BLEND_MODES
from temporal.image_filters import IMAGE_FILTERS, ImageFilter
from temporal.image_mask import ImageMask
from temporal.meta.serializable import Serializable, field
from temporal.utils.collection import reorder_dict
from temporal.utils.image import NumpyImage, match_image
from temporal.utils.math import lerp, normalize
from temporal.utils.numpy import saturate_array


class ImageFilterer(Serializable):
    filter_order: list[str] = field(factory = list)
    filters: dict[str, ImageFilter] = field(factory = lambda: {id: cls() for id, cls in IMAGE_FILTERS.items()})

    def filter_image(self, npim: NumpyImage, amount_scale: float, seed: int) -> NumpyImage:
        for filter in reorder_dict(self.filters, self.filter_order or []).values():
            if not filter.enabled:
                continue

            npim = self._blend(
                npim,
                filter.process(npim, seed),
                filter.amount * (amount_scale if filter.amount_relative else 1.0),
                filter.blend_mode,
                filter.mask,
            )

        return saturate_array(npim)

    @staticmethod
    def _blend(npim: NumpyImage, processed: NumpyImage, amount: float, blend_mode: str, mask: ImageMask) -> NumpyImage:
        if npim is processed or amount == 0.0:
            return npim

        processed = BLEND_MODES[blend_mode].blend(npim, processed)

        if amount == 1.0 and mask.image is None:
            return processed

        if mask.image is not None:
            factor = match_image(mask.image, npim)

            if mask.normalized:
                factor = normalize(factor, factor.min(), factor.max())

            if mask.inverted:
                factor = 1.0 - factor

            if mask.blurring:
                factor = skimage.filters.gaussian(factor, round(mask.blurring), channel_axis = 2)

        else:
            factor = 1.0

        factor *= amount

        return lerp(npim, processed, factor)
