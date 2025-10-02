from abc import abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.ticker import MaxNLocator

from modules.general_data import GeneralData
from modules.object import Field, Static
from modules.pipeline_module import PipelineModule
from modules.pipeline_state import PipelineResult, PipelineState
from modules.utils.fs import ensure_directory_exists
from modules.utils.image import NumpyImage, PILImage, save_image
from modules.utils.matplotlib import get_figure_as_image
from modules.utils.numpy import FloatArray


class MeasuringModule(PipelineModule, abstract = True):
    is_visualizable = True
    visualization_type = "image"

    file_name: str = Static("", flags = {"private"})
    channels: list[tuple[str, str]] = Static([], flags = {"private"})

    plot_every_nth_iteration: int = Field(10, name = "Plot every N-th iteration", minimum = 1, step = 1, display = "box")
    iteration: int = Field(0, flags = {"runtime"})
    data: Optional[FloatArray] = Field(None, flags = {"runtime"})
    count: int = Field(0, flags = {"runtime"})

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        if self.iteration % self.plot_every_nth_iteration != 0:
            self.iteration += 1
            return

        if self.data is None:
            self.data = np.zeros((1, 1 + len(self.channels)))
            self.count = 0

        if self.data.shape[0] <= self.count:
            self.data = np.concatenate([self.data, np.zeros_like(self.data)], axis = 0)

        iter_data = self.data[self.count]
        iter_data[0] = self.iteration
        iter_data[1:] = self.measure(image)

        self.count += 1

        save_image(self.plot(), ensure_directory_exists(general.path / "metrics") / f"{self.file_name}.png")

        self.iteration += 1

        yield PipelineState.finish(image = image, preview = self.preview)

    def visualize(self, general: GeneralData) -> PILImage:
        return self.plot()

    @abstractmethod
    def measure(self, image: NumpyImage) -> list[float]:
        raise NotImplementedError

    def plot(self) -> PILImage:
        if self.data is None:
            raise ValueError

        indices = self.data[:self.count, 0]

        plt.title(self.name)
        plt.xlabel("Iteration")
        plt.xticks(indices)
        plt.xlim(indices[0], indices[-1])
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.ylabel("Level")
        plt.grid()

        for channel, (label, color) in enumerate(self.channels, 1):
            values = self.data[:self.count, channel]

            plt.axhline(values[0], color = color, linestyle = ":", linewidth = 0.5)
            plt.axhline(float(np.mean(values)), color = color, linestyle = "--", linewidth = 1.0)
            plt.plot(indices, values, color = color, label = label, linestyle = "--", linewidth = 0.5, marker = "+", markersize = 3)

            if self.count > 3:
                plt.plot(indices, scipy.signal.savgol_filter(values, min(self.count, 51), 3), color = color, label = f"{label} (smoothed)", linestyle = "-")

        plt.legend()

        image = get_figure_as_image()

        plt.close()

        return image
