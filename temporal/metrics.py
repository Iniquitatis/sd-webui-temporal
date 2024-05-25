from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from PIL import Image

from temporal.serialization import load_object, save_object
from temporal.utils.fs import ensure_directory_exists, load_json, save_json
from temporal.utils.image import PILImage, pil_to_np, save_image

class Metrics:
    def __init__(self) -> None:
        self.luminance_mean = []
        self.luminance_std = []
        self.color_level_mean = []
        self.color_level_std = []
        self.noise_sigma = []

    def measure(self, im: PILImage) -> None:
        npim = pil_to_np(im)
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = 2)
        red, green, blue = npim[..., 0], npim[..., 1], npim[..., 2]

        self.luminance_mean.append(np.mean(grayscale))
        self.luminance_std.append(np.std(grayscale))
        self.color_level_mean.append([np.mean(red), np.mean(green), np.mean(blue)])
        self.color_level_std.append([np.std(red), np.std(green), np.std(blue)])
        self.noise_sigma.append(skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2))

    def load(self, path: Path) -> None:
        if data := load_json(path / "data.json"):
            load_object(self, data, path)

    def save(self, path: Path) -> None:
        ensure_directory_exists(path)
        save_json(path / "data.json", save_object(self, path))

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

        def plot_noise_graph(data, label, color):
            plt.axhline(data[0], color = color, linestyle = ":", linewidth = 0.5)
            plt.axhline(np.mean(data), color = color, linestyle = "--", linewidth = 1.0)
            plt.plot(data, color = color, label = label, linestyle = "--", linewidth = 0.5, marker = "+", markersize = 3)

            if data.size > 3:
                plt.plot(scipy.signal.savgol_filter(data, min(data.size, 51), 3), color = color, label = f"{label} (smoothed)", linestyle = "-")

        with figure("luminance_mean", "Luminance mean"):
            plot_noise_graph(np.array(self.luminance_mean), "Luminance", "gray")

        with figure("luminance_std", "Luminance standard deviation"):
            plot_noise_graph(np.array(self.luminance_std), "Luminance", "gray")

        with figure("color_level_mean", "Color level mean"):
            plot_noise_graph(np.array(self.color_level_mean)[..., 0], "Red", "darkred")
            plot_noise_graph(np.array(self.color_level_mean)[..., 1], "Green", "darkgreen")
            plot_noise_graph(np.array(self.color_level_mean)[..., 2], "Blue", "darkblue")

        with figure("color_level_std", "Color level standard deviation"):
            plot_noise_graph(np.array(self.color_level_std)[..., 0], "Red", "darkred")
            plot_noise_graph(np.array(self.color_level_std)[..., 1], "Green", "darkgreen")
            plot_noise_graph(np.array(self.color_level_std)[..., 2], "Blue", "darkblue")

        with figure("noise_sigma", "Noise sigma"):
            plot_noise_graph(np.array(self.noise_sigma), "Noise sigma", "royalblue")

        return result

    def plot_to_directory(self, dir: Path) -> None:
        ensure_directory_exists(dir)

        for key, im in self.plot().items():
            save_image(im, dir / f"{key}.png")
