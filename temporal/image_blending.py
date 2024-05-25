from typing import Optional

import numpy as np

from temporal.utils.func import make_func_registerer
from temporal.utils.image import NumpyImage, join_hsv_to_rgb, split_hsv


BLEND_MODES, blend_mode = make_func_registerer(name = "")


def blend_images(npim: NumpyImage, modulator: Optional[NumpyImage], mode: str) -> NumpyImage:
    if modulator is None:
        return npim

    return BLEND_MODES[mode].func(npim, modulator)


@blend_mode("normal", "Normal")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return s


@blend_mode("add", "Add")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return b + s


@blend_mode("subtract", "Subtract")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return b - s


@blend_mode("multiply", "Multiply")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return b * s


@blend_mode("divide", "Divide")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return b / np.maximum(s, 1e-6)


@blend_mode("lighten", "Lighten")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return np.maximum(b, s)


@blend_mode("darken", "Darken")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return np.minimum(b, s)


@blend_mode("hard_light", "Hard light")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    result = np.zeros_like(s)
    less_idx = np.where(s <= 0.5)
    more_idx = np.where(s >  0.5)
    result[less_idx] = BLEND_MODES["multiply"].func(b[less_idx], 2.0 * s[less_idx])
    result[more_idx] = BLEND_MODES["screen"].func(b[more_idx], 2.0 * s[more_idx] - 1.0)
    return result


@blend_mode("soft_light", "Soft light")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    def D(b):
        result = np.zeros_like(b)
        less_idx = np.where(b <= 0.25)
        more_idx = np.where(b >  0.25)
        result[less_idx] = ((16.0 * b[less_idx] - 12.0) * b[less_idx] + 4.0) * b[less_idx]
        result[more_idx] = np.sqrt(b[more_idx])
        return result

    result = np.zeros_like(s)
    less_idx = np.where(s <= 0.5)
    more_idx = np.where(s >  0.5)
    result[less_idx] = b[less_idx] - (1.0 - 2.0 * s[less_idx]) * b[less_idx] * (1.0 - b[less_idx])
    result[more_idx] = b[more_idx] + (2.0 * s[more_idx] - 1.0) * (D(b[more_idx]) - b[more_idx])
    return result


@blend_mode("color_dodge", "Color dodge")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    result = np.zeros_like(s)
    b0_mask = b == 0.0
    s1_mask = s == 1.0
    else_mask = np.logical_not(np.logical_or(b0_mask, s1_mask))
    else_idx = np.where(else_mask)
    result[np.where(b0_mask)] = 0.0
    result[np.where(s1_mask)] = 1.0
    result[else_idx] = np.minimum(1.0, b[else_idx] / (1.0 - s[else_idx]))
    return result


@blend_mode("color_burn", "Color burn")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    result = np.zeros_like(s)
    b1_mask = b == 1.0
    s0_mask = s == 0.0
    else_mask = np.logical_not(np.logical_or(b1_mask, s0_mask))
    else_idx = np.where(else_mask)
    result[np.where(b1_mask)] = 1.0
    result[np.where(s0_mask)] = 0.0
    result[else_idx] = 1.0 - np.minimum(1.0, (1.0 - b[else_idx]) / s[else_idx])
    return result


@blend_mode("overlay", "Overlay")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return BLEND_MODES["hard_light"].func(s, b)


@blend_mode("screen", "Screen")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return b + s - (b * s)


@blend_mode("difference", "Difference")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return np.abs(b - s)


@blend_mode("exclusion", "Exclusion")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    return b + s - 2.0 * b * s


@blend_mode("hue", "Hue")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    bh, bs, bv = split_hsv(b)
    sh, ss, sv = split_hsv(s)
    return join_hsv_to_rgb(sh, bs, bv)


@blend_mode("saturation", "Saturation")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    bh, bs, bv = split_hsv(b)
    sh, ss, sv = split_hsv(s)
    return join_hsv_to_rgb(bh, ss, bv)


@blend_mode("value", "Value")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    bh, bs, bv = split_hsv(b)
    sh, ss, sv = split_hsv(s)
    return join_hsv_to_rgb(bh, bs, sv)


@blend_mode("color", "Color")
def _(b: NumpyImage, s: NumpyImage) -> NumpyImage:
    bh, bs, bv = split_hsv(b)
    sh, ss, sv = split_hsv(s)
    return join_hsv_to_rgb(sh, ss, bv)
