from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from PIL import Image
from numpy.typing import NDArray

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, PILImage, save_image


class Measurement(Serializable):
    luminance_mean: float = Field(0.0)
    luminance_std: float = Field(0.0)
    color_level_mean: tuple[float, float, float] = Field((0.0, 0.0, 0.0))
    color_level_std: tuple[float, float, float] = Field((0.0, 0.0, 0.0))
    noise_sigma: float = Field(0.0)


class Metrics(Serializable):
    measurements: list[Measurement] = Field(factory = list)

    def measure(self, npim: NumpyImage) -> None:
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = 2)
        red, green, blue = npim[..., 0], npim[..., 1], npim[..., 2]

        self.measurements.append(Measurement(
            luminance_mean = _fp(np.mean(grayscale)),
            luminance_std = _fp(np.std(grayscale)),
            color_level_mean = (_fp(np.mean(red)), _fp(np.mean(green)), _fp(np.mean(blue))),
            color_level_std = (_fp(np.std(red)), _fp(np.std(green)), _fp(np.std(blue))),
            noise_sigma = _fp(skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2)),
        ))

    def plot(self) -> dict[str, PILImage]:
        result = {}

        @contextmanager
        def figure(key: str, title: str) -> Iterator[None]:
            plt.title(title)
            plt.xlabel("Frame")
            plt.ylabel("Level")
            plt.grid()

            try:
                yield
            finally:
                plt.legend()

                buffer = BytesIO()
                plt.savefig(buffer, format = "png")
                buffer.seek(0)

                im = Image.open(buffer)
                im.load()
                result[key] = im

                plt.close()

        def plot_noise_graph(data: NDArray[np.float_], label: str, color: str) -> None:
            plt.axhline(data[0], color = color, linestyle = ":", linewidth = 0.5)
            plt.axhline(_fp(np.mean(data)), color = color, linestyle = "--", linewidth = 1.0)
            plt.plot(data, color = color, label = label, linestyle = "--", linewidth = 0.5, marker = "+", markersize = 3)

            if data.size > 3:
                plt.plot(scipy.signal.savgol_filter(data, min(data.size, 51), 3), color = color, label = f"{label} (smoothed)", linestyle = "-")

        values = np.array([[
            x.luminance_mean,
            x.luminance_std,
            *x.color_level_mean,
            *x.color_level_std,
            x.noise_sigma,
        ] for x in self.measurements])

        with figure("luminance_mean", "Luminance mean"):
            plot_noise_graph(values[:, 0], "Luminance", "gray")

        with figure("luminance_std", "Luminance standard deviation"):
            plot_noise_graph(values[:, 1], "Luminance", "gray")

        with figure("color_level_mean", "Color level mean"):
            plot_noise_graph(values[:, 2], "Red", "darkred")
            plot_noise_graph(values[:, 3], "Green", "darkgreen")
            plot_noise_graph(values[:, 4], "Blue", "darkblue")

        with figure("color_level_std", "Color level standard deviation"):
            plot_noise_graph(values[:, 5], "Red", "darkred")
            plot_noise_graph(values[:, 6], "Green", "darkgreen")
            plot_noise_graph(values[:, 7], "Blue", "darkblue")

        with figure("noise_sigma", "Noise sigma"):
            plot_noise_graph(values[:, 8], "Noise sigma", "royalblue")

        return result

    def plot_to_directory(self, dir: Path) -> None:
        ensure_directory_exists(dir)

        for key, im in self.plot().items():
            save_image(im, dir / f"{key}.png")


# NOTE: Because type checker may give a false positive
def _fp(value: Any) -> float:
    return float(value)
