from math import ceil

import numpy as np
import skimage
from scipy.spatial import KDTree

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.painting import PaintingModule
from temporal.seed import Seed
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp, normalize, ratio
from temporal.utils.numpy import FloatArray, FloatType, IntArray, IntType, random_array, saturate_array


class VoronoiModule(PaintingModule):
    name = "Voronoi"

    type: str = Field("diagram", name = "Type", choices = {"diagram": "Diagram", "edges": "Edges", "distances": "Distances"}, display = "radio")
    scale: float = Field(1.0, name = "Scale", minimum = 1.0, step = 0.01, display = "box")
    randomness: float = Field(0.0, name = "Randomness", minimum = 0.0, maximum = 1.0, step = 0.01, display = "slider")
    use_global_seed: bool = Field(False, name = "Use global seed")
    seed: Seed = Field(Seed, name = "Seed", dependencies = {"use_global_seed": False})
    blurring: float = Field(0.0, name = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, dependencies = {"type": "diagram"}, display = "slider")
    dilation: int = Field(0, name = "Dilation", minimum = 0, maximum = 50, step = 1, dependencies = {"type": "edges"}, display = "slider")
    color_a: Color = Field(lambda: Color(0.0, 0.0, 0.0), name = "Color A", channels = 4, dependencies = {"type": ["edges", "distances"]})
    color_b: Color = Field(lambda: Color(1.0, 1.0, 1.0), name = "Color B", channels = 4, dependencies = {"type": ["edges", "distances"]})

    def draw(self, size: tuple[int, int], general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        shape = size[1], size[0]
        seed = seed if self.use_global_seed else self.seed.fixed_value

        counts, distances, indices = self._query(shape, seed)

        if self.type == "diagram":
            colors = _random_nd(tuple(ceil(x) for x in counts) + (3,), 0.0, 1.0, seed).reshape(-1, 3)

            result = colors[indices]

            if self.blurring > 0.0:
                result = saturate_array(skimage.filters.gaussian(result, round(self.blurring), channel_axis = -1))

            return result

        elif self.type == "edges":
            edges = np.zeros(shape, dtype = np.bool_)
            edges[:-1, :] |= (indices[:-1, :] != indices[1:, :])
            edges[:, :-1] |= (indices[:, :-1] != indices[:, 1:])

            pattern = edges.astype(FloatType)

            if self.dilation > 0:
                footprint = skimage.morphology.disk(self.dilation)
                pattern = skimage.morphology.dilation(pattern, footprint, out = pattern)

            return lerp(
                self.color_a.to_numpy(4),
                self.color_b.to_numpy(4),
                pattern.reshape((size[1], size[0], 1)),
            )

        elif self.type == "distances":
            pattern = normalize(distances, distances.min(), distances.max())

            return lerp(
                self.color_a.to_numpy(4),
                self.color_b.to_numpy(4),
                pattern.reshape((size[1], size[0], 1)),
            )

        else:
            raise ValueError(f"Incorrect type {self.type}")

    def _query(self, shape: tuple[int, ...], seed: int) -> tuple[FloatArray, FloatArray, IntArray]:
        dim_count = len(shape)

        counts = np.array([x * (self.scale + 1.0) for x in ratio(shape)])

        grid = np.stack(np.meshgrid(*(
            np.arange(ceil(x)) * (1.0 / (x - 1.0)) for x in counts
        )), axis = -1)

        offsets = _random_nd(grid.shape, -0.5, 0.5, seed) * (1.0 / counts) * self.randomness

        point_coords = (grid + offsets).reshape(-1, dim_count) * shape
        pixel_coords = np.indices(shape).reshape(dim_count, -1).T

        distances, indices = KDTree(point_coords).query(pixel_coords)

        return counts, distances.reshape(shape), indices.reshape(shape)


def _random_nd(shape: tuple[int, ...], low: float, high: float, seed: int) -> FloatArray:
    BASE = 1024

    *base_dims, row_length = shape
    row_count = np.prod(base_dims, dtype = IntType)

    seed_offsets = np.zeros(row_count, dtype = IntType)

    for coords in np.unravel_index(np.arange(row_count), base_dims):
        seed_offsets = seed_offsets * BASE + coords

    result = np.empty((row_count, row_length), dtype = FloatType)

    for i in range(row_count):
        result[i] = random_array((row_length,), low, high, seed + seed_offsets[i])

    return result.reshape(shape)
