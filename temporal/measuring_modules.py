from abc import abstractmethod
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from PIL import Image
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from temporal.meta.configurable import IntParam
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_modules import PipelineModule
from temporal.session import Session
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, PILImage, save_image


class MeasuringModule(PipelineModule, abstract = True):
    icon = "\U0001f4c8"

    file_name: str = ""
    channels: list[tuple[str, str]] = []

    plot_every_nth_frame: int = IntParam("Plot every N-th frame", minimum = 1, step = 1, value = 10, ui_type = "box")

    data: Optional[NDArray[np.float_]] = Field(None)
    count: int = Field(0)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if frame_index % self.plot_every_nth_frame != 0:
            return images

        if self.data is None:
            self.data = np.zeros((len(images), 1, 1 + len(self.channels)))
            self.count = 0

        if self.data.shape[1] <= self.count:
            self.data = np.concatenate([self.data, np.zeros_like(self.data)], axis = 1)

        for parallel_index, image in enumerate(images):
            frame_data = self.data[parallel_index, self.count]
            frame_data[0] = frame_index
            frame_data[1:] = self.measure(image)

        self.count += 1

        for parallel_index in range(len(images)):
            save_image(self.plot(parallel_index), ensure_directory_exists(session.project.path / "metrics") / f"{self.file_name}-{parallel_index + 1:02d}.png")

        return images

    def reset(self) -> None:
        self.data = None
        self.count = 0

    @abstractmethod
    def measure(self, npim: NumpyImage) -> list[float]:
        raise NotImplementedError

    def plot(self, parallel_index: int) -> PILImage:
        if self.data is None or parallel_index >= self.data.shape[0]:
            raise ValueError

        indices = self.data[parallel_index, :self.count, 0]

        plt.title(self.name)
        plt.xlabel("Frame")
        plt.xticks(indices)
        plt.xlim(indices[0], indices[-1])
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.ylabel("Level")
        plt.grid()

        for channel, (label, color) in enumerate(self.channels, 1):
            values = self.data[parallel_index, :self.count, channel]

            plt.axhline(values[0], color = color, linestyle = ":", linewidth = 0.5)
            plt.axhline(float(np.mean(values)), color = color, linestyle = "--", linewidth = 1.0)
            plt.plot(indices, values, color = color, label = label, linestyle = "--", linewidth = 0.5, marker = "+", markersize = 3)

            if self.count > 3:
                plt.plot(indices, scipy.signal.savgol_filter(values, min(self.count, 51), 3), color = color, label = f"{label} (smoothed)", linestyle = "-")

        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format = "png")
        buffer.seek(0)

        im = Image.open(buffer)
        im.load()

        plt.close()

        return im


class ColorLevelMeanMeasuringModule(MeasuringModule):
    id = "color_level_mean_measuring"
    name = "Color level mean"
    file_name = "color_level_mean"
    channels = [
        ("Red", "darkred"),
        ("Green", "darkgreen"),
        ("Blue", "darkblue"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        red, green, blue = npim[..., 0], npim[..., 1], npim[..., 2]
        return [float(np.mean(red)), float(np.mean(green)), float(np.mean(blue))]


class ColorLevelSigmaMeasuringModule(MeasuringModule):
    id = "color_level_sigma_measuring"
    name = "Color level sigma"
    file_name = "color_level_sigma"
    channels = [
        ("Red", "darkred"),
        ("Green", "darkgreen"),
        ("Blue", "darkblue"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        red, green, blue = npim[..., 0], npim[..., 1], npim[..., 2]
        return [float(np.std(red)), float(np.std(green)), float(np.std(blue))]


class LuminanceMeanMeasuringModule(MeasuringModule):
    id = "luminance_mean_measuring"
    name = "Luminance mean"
    file_name = "luminance_mean"
    channels = [
        ("Luminance", "gray"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = -1)
        return [float(np.mean(grayscale))]


class LuminanceSigmaMeasuringModule(MeasuringModule):
    id = "luminance_sigma_measuring"
    name = "Luminance sigma"
    file_name = "luminance_sigma"
    channels = [
        ("Luminance", "gray"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = -1)
        return [float(np.std(grayscale))]


class NoiseSigmaMeasuringModule(MeasuringModule):
    id = "noise_sigma_measuring"
    name = "Noise sigma"
    file_name = "noise_sigma"
    channels = [
        ("Noise sigma", "royalblue"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        return [float(skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = -1))]
