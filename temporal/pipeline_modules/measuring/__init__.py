from abc import abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.ticker import MaxNLocator

from temporal.general_data import GeneralData
from temporal.object import Field, Param, Static
from temporal.pipeline_module import PipelineModule
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, PILImage, save_image
from temporal.utils.matplotlib import get_figure_as_image
from temporal.utils.numpy import FloatArray


class MeasuringModule(PipelineModule, abstract = True):
    file_name: str = Static("")
    channels: list[tuple[str, str]] = Static([])

    plot_every_nth_iteration: int = Param("Plot every N-th iteration", minimum = 1, step = 1, value = 10, ui_type = "box")

    data: Optional[FloatArray] = Field(None, flags = {"private"})
    count: int = Field(0, flags = {"private"})

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        if iter_index % self.plot_every_nth_iteration != 0:
            return image

        if self.data is None:
            self.data = np.zeros((1, 1 + len(self.channels)))
            self.count = 0

        if self.data.shape[0] <= self.count:
            self.data = np.concatenate([self.data, np.zeros_like(self.data)], axis = 0)

        iter_data = self.data[self.count]
        iter_data[0] = iter_index
        iter_data[1:] = self.measure(image)

        self.count += 1

        save_image(self.plot(), ensure_directory_exists(general.path / "metrics") / f"{self.file_name}.png")

        return image

    def reset(self, general: GeneralData) -> None:
        self.data = None
        self.count = 0

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
