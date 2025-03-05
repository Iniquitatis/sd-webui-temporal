from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

from temporal.utils.image import PILImage


def get_figure_as_image() -> PILImage:
    with BytesIO() as stream:
        plt.savefig(stream, format = "png")

        stream.seek(0)

        image = Image.open(stream)
        image.load()

        return image
