import json
from contextlib import contextmanager
from io import BytesIO
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from PIL import Image

from temporal.fs import safe_get_directory
from temporal.serialization import load_object, save_object

class Metrics:
    def __init__(self):
        self.luminance_mean = []
        self.luminance_std = []
        self.color_level_mean = []
        self.color_level_std = []
        self.noise_sigma = []

    def measure(self, im):
        npim = skimage.img_as_float(im)
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = 2)
        red, green, blue = npim[..., 0], npim[..., 1], npim[..., 2]

        self.luminance_mean.append(np.mean(grayscale))
        self.luminance_std.append(np.std(grayscale))
        self.color_level_mean.append([np.mean(red), np.mean(green), np.mean(blue)])
        self.color_level_std.append([np.std(red), np.std(green), np.std(blue)])
        self.noise_sigma.append(skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2))

    def load(self, project_dir):
        metrics_dir = project_dir / "metrics"

        if (data_path := (metrics_dir / "data.json")).is_file():
            with open(data_path, "r", encoding = "utf-8") as data_file:
                load_object(self, json.load(data_file), metrics_dir)

    def save(self, project_dir):
        metrics_dir = safe_get_directory(project_dir / "metrics")

        with open(metrics_dir / "data.json", "w", encoding = "utf-8") as data_file:
            json.dump(save_object(self, metrics_dir), data_file, indent = 4)

    def clear(self, project_dir):
        self.luminance_mean.clear()
        self.luminance_std.clear()
        self.color_level_mean.clear()
        self.color_level_std.clear()
        self.noise_sigma.clear()

        if (metrics_dir := (project_dir / "metrics")).is_dir():
            rmtree(metrics_dir)

    def plot(self, project_dir, save_images = False):
        metrics_dir = safe_get_directory(project_dir / "metrics")

        result = []

        @contextmanager
        def figure(title, path):
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

                if save_images:
                    im.save(path)

                result.append(im)

                plt.close()

        def plot_noise_graph(data, label, color):
            plt.axhline(data[0], color = color, linestyle = ":", linewidth = 0.5)
            plt.axhline(np.mean(data), color = color, linestyle = "--", linewidth = 1.0)
            plt.plot(data, color = color, label = label, linestyle = "--", linewidth = 0.5, marker = "+", markersize = 3)

            if data.size > 3:
                plt.plot(scipy.signal.savgol_filter(data, min(data.size, 51), 3), color = color, label = f"{label} (smoothed)", linestyle = "-")

        with figure("Luminance mean", metrics_dir / "luminance_mean.png"):
            plot_noise_graph(np.array(self.luminance_mean), "Luminance", "gray")

        with figure("Luminance standard deviation", metrics_dir / "luminance_std.png"):
            plot_noise_graph(np.array(self.luminance_std), "Luminance", "gray")

        with figure("Color level mean", metrics_dir / "color_level_mean.png"):
            plot_noise_graph(np.array(self.color_level_mean)[..., 0], "Red", "darkred")
            plot_noise_graph(np.array(self.color_level_mean)[..., 1], "Green", "darkgreen")
            plot_noise_graph(np.array(self.color_level_mean)[..., 2], "Blue", "darkblue")

        with figure("Color level standard deviation", metrics_dir / "color_level_std.png"):
            plot_noise_graph(np.array(self.color_level_std)[..., 0], "Red", "darkred")
            plot_noise_graph(np.array(self.color_level_std)[..., 1], "Green", "darkgreen")
            plot_noise_graph(np.array(self.color_level_std)[..., 2], "Blue", "darkblue")

        with figure("Noise sigma", metrics_dir / "noise_sigma.png"):
            plot_noise_graph(np.array(self.noise_sigma), "Noise sigma", "royalblue")

        return result
