from abc import abstractmethod
from typing import Type

import numpy as np

from temporal.meta.registerable import Registerable
from temporal.meta.serializable import Serializable
from temporal.utils.image import NumpyImage, join_hsv_to_rgb, split_hsv


BLEND_MODES: list[Type["BlendMode"]] = []


class BlendMode(Registerable, Serializable, abstract = True):
    store = BLEND_MODES

    @staticmethod
    @abstractmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        raise NotImplementedError


class NormalBlendMode(BlendMode):
    name = "Normal"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return s


class AddBlendMode(BlendMode):
    name = "Add"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return b + s


class SubtractBlendMode(BlendMode):
    name = "Subtract"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return b - s


class MultiplyBlendMode(BlendMode):
    name = "Multiply"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return b * s


class DivideBlendMode(BlendMode):
    name = "Divide"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return b / np.maximum(s, 1e-6)


class LightenBlendMode(BlendMode):
    name = "Lighten"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return np.maximum(b, s)


class DarkenBlendMode(BlendMode):
    name = "Darken"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return np.minimum(b, s)


class HardLightBlendMode(BlendMode):
    name = "Hard light"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        result = np.zeros_like(s)
        less_idx = np.where(s <= 0.5)
        more_idx = np.where(s >  0.5)
        result[less_idx] = MultiplyBlendMode.blend(b[less_idx], 2.0 * s[less_idx])
        result[more_idx] = ScreenBlendMode.blend(b[more_idx], 2.0 * s[more_idx] - 1.0)
        return result


class SoftLightBlendMode(BlendMode):
    name = "Soft light"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
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


class ColorDodgeBlendMode(BlendMode):
    name = "Color dodge"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        result = np.zeros_like(s)
        b0_mask = b == 0.0
        s1_mask = s == 1.0
        else_mask = np.logical_not(np.logical_or(b0_mask, s1_mask))
        else_idx = np.where(else_mask)
        result[np.where(b0_mask)] = 0.0
        result[np.where(s1_mask)] = 1.0
        result[else_idx] = np.minimum(1.0, b[else_idx] / (1.0 - s[else_idx]))
        return result


class ColorBurnBlendMode(BlendMode):
    name = "Color burn"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        result = np.zeros_like(s)
        b1_mask = b == 1.0
        s0_mask = s == 0.0
        else_mask = np.logical_not(np.logical_or(b1_mask, s0_mask))
        else_idx = np.where(else_mask)
        result[np.where(b1_mask)] = 1.0
        result[np.where(s0_mask)] = 0.0
        result[else_idx] = 1.0 - np.minimum(1.0, (1.0 - b[else_idx]) / s[else_idx])
        return result


class OverlayBlendMode(BlendMode):
    name = "Overlay"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return HardLightBlendMode.blend(s, b)


class ScreenBlendMode(BlendMode):
    name = "Screen"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return b + s - (b * s)


class DifferenceBlendMode(BlendMode):
    name = "Difference"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return np.abs(b - s)


class ExclusionBlendMode(BlendMode):
    name = "Exclusion"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        return b + s - 2.0 * b * s


class HueBlendMode(BlendMode):
    name = "Hue"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        bh, bs, bv = split_hsv(b)
        sh, ss, sv = split_hsv(s)
        return join_hsv_to_rgb(sh, bs, bv)


class SaturationBlendMode(BlendMode):
    name = "Saturation"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        bh, bs, bv = split_hsv(b)
        sh, ss, sv = split_hsv(s)
        return join_hsv_to_rgb(bh, ss, bv)


class ValueBlendMode(BlendMode):
    name = "Value"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        bh, bs, bv = split_hsv(b)
        sh, ss, sv = split_hsv(s)
        return join_hsv_to_rgb(bh, bs, sv)


class ColorBlendMode(BlendMode):
    name = "Color"

    @staticmethod
    def blend(b: NumpyImage, s: NumpyImage) -> NumpyImage:
        bh, bs, bv = split_hsv(b)
        sh, ss, sv = split_hsv(s)
        return join_hsv_to_rgb(sh, ss, bv)
