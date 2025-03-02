from abc import abstractmethod
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image
from matplotlib.ticker import MaxNLocator

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_module import PipelineModule
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, PILImage, save_image
from temporal.utils.numpy import FloatArray


class MeasuringModule(PipelineModule, abstract = True):
    file_name: str = ""
    channels: list[tuple[str, str]] = []

    plot_every_nth_frame: int = Param("Plot every N-th frame", minimum = 1, step = 1, value = 10, ui_type = "box")

    data: Optional[FloatArray] = Field(None, flags = {"private"})
    count: int = Field(0, flags = {"private"})

    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        if frame_index % self.plot_every_nth_frame != 0:
            return image

        if self.data is None:
            self.data = np.zeros((1, 1 + len(self.channels)))
            self.count = 0

        if self.data.shape[0] <= self.count:
            self.data = np.concatenate([self.data, np.zeros_like(self.data)], axis = 0)

        frame_data = self.data[self.count]
        frame_data[0] = frame_index
        frame_data[1:] = self.measure(image)

        self.count += 1

        save_image(self.plot(), ensure_directory_exists(general.path / "metrics") / f"{self.file_name}.png")

        return image

    def reset(self) -> None:
        self.data = None
        self.count = 0

    @abstractmethod
    def measure(self, npim: NumpyImage) -> list[float]:
        raise NotImplementedError

    def plot(self) -> PILImage:
        if self.data is None:
            raise ValueError

        indices = self.data[:self.count, 0]

        plt.title(self.name)
        plt.xlabel("Frame")
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

        buffer = BytesIO()
        plt.savefig(buffer, format = "png")
        buffer.seek(0)

        im = Image.open(buffer)
        im.load()

        plt.close()

        return im
