import numpy as np
import skimage
from scipy.spatial import KDTree

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.painting import PaintingModule
from temporal.seed import Seed
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp, normalize
from temporal.utils.numpy import FloatArray, FloatType, random_array, saturate_array


class VoronoiModule(PaintingModule):
    name = "Voronoi"

    type: str = Field("diagram", name = "Type", choices = {"diagram": "Diagram", "edges": "Edges", "distances": "Distances"}, display = "radio")
    point_count: int = Field(20, name = "Point count", minimum = 1, step = 1, display = "box")
    use_global_seed: bool = Field(False, name = "Use global seed")
    seed: Seed = Field(Seed, name = "Seed", dependencies = {"use_global_seed": False})
    blurring: float = Field(0.0, name = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, dependencies = {"type": "diagram"}, display = "slider")
    dilation: int = Field(0, name = "Dilation", minimum = 0, maximum = 50, step = 1, dependencies = {"type": "edges"}, display = "slider")
    color_a: Color = Field(lambda: Color(0.0, 0.0, 0.0), name = "Color A", channels = 4, dependencies = {"type": "distances"})
    color_b: Color = Field(lambda: Color(1.0, 1.0, 1.0), name = "Color B", channels = 4, dependencies = {"type": "distances"})

    def draw(self, size: tuple[int, int], general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        seed = seed if self.use_global_seed else self.seed.fixed_value

        if self.type == "diagram":
            result = self._generate_diagram((size[1], size[0], 3), seed)

            if self.blurring > 0.0:
                result = saturate_array(skimage.filters.gaussian(result, round(self.blurring), channel_axis = -1))

            return result

        elif self.type == "edges":
            pattern = self._generate_edges((size[1], size[0]), seed)

            if self.dilation > 0:
                footprint = skimage.morphology.disk(self.dilation)
                pattern = skimage.morphology.dilation(pattern, footprint, out = pattern)

            return lerp(
                Color(0.0, 0.0, 0.0).to_numpy(4),
                Color(1.0, 1.0, 1.0).to_numpy(4),
                pattern.reshape((size[1], size[0], 1)),
            )

        elif self.type == "distances":
            return lerp(
                self.color_a.to_numpy(4),
                self.color_b.to_numpy(4),
                self._generate_distances((size[1], size[0]), seed).reshape((size[1], size[0], 1)),
            )

        else:
            raise ValueError(f"Incorrect type {self.type}")

    def _generate_diagram(self, shape: tuple[int, int, int], seed: int) -> FloatArray:
        points = random_array((self.point_count, 2), seed = seed) * shape[:2]
        colors = random_array((self.point_count, shape[2]), seed = seed)

        coords = np.indices(shape[:2]).reshape(2, -1).T

        closest_indices = KDTree(points).query(coords)[1]
        closest_indices = closest_indices.reshape(shape[:2])

        return colors[closest_indices]

    def _generate_edges(self, shape: tuple[int, int], seed: int) -> FloatArray:
        points = random_array((self.point_count, 2), seed = seed) * shape[:2]

        coords = np.indices(shape[:2]).reshape(2, -1).T

        closest_indices = KDTree(points).query(coords)[1]
        closest_indices = closest_indices.reshape(shape[:2])

        edges = np.zeros(shape[:2], dtype = np.bool_)
        edges[:-1, :] |= (closest_indices[:-1, :] != closest_indices[1:, :])
        edges[:, :-1] |= (closest_indices[:, :-1] != closest_indices[:, 1:])

        return edges.astype(FloatType)

    def _generate_distances(self, shape: tuple[int, int], seed: int) -> FloatArray:
        points = random_array((self.point_count, 2), seed = seed) * shape[:2]

        coords = np.indices(shape[:2]).reshape(2, -1).T

        distances = KDTree(points).query(coords)[0]
        distances = distances.reshape(shape[:2])

        return normalize(distances, distances.min(), distances.max())
