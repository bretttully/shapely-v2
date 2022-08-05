import abc
import dataclasses
from typing import List, Tuple, Union
import uuid

import geopandas as gpd
import numpy as np
import shapely.affinity
import shapely.geometry
import shapely.ops
from shapely.validation import explain_validity

GenericPoly = Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]


def strings_to_uuid_v5(*args: str) -> str:
    """Creates a reproducible hash of the argument strings into a v5 uuid string"""
    if not args:
        raise ValueError("You must pass at least one string")
    _NAMESPACE_UUID = uuid.UUID(int=0)
    return str(uuid.uuid5(_NAMESPACE_UUID, "_".join(args)))


@dataclasses.dataclass(frozen=True)
class RandomPolyGenerator(abc.ABC):
    top_left_tile: Tuple[int, int] = (18807, 23776)  # chosen only so that it can be loaded into QGIS easily (@z=16)
    px_per_tile: int = 1024

    def __call__(self, seed: int) -> gpd.GeoDataFrame:
        """Use this method to fill the unit square with polygons and their associated 'confidence'"""
        np.random.seed(seed)
        polys = self._fill_unit_square()
        polys = self._scale(polys)
        polys = self._translate(polys)
        polys = self._clip_by_confidence(polys)
        return gpd.GeoDataFrame(polys, columns=["uid", "confidence", "geometry"], geometry="geometry")

    @abc.abstractmethod
    def _fill_unit_square(self) -> List[Tuple[float, GenericPoly]]:
        """Use this method to fill the unit square with polygons and their associated 'confidence'"""

    def _scale(self, polygons: List[Tuple[float, GenericPoly]]) -> List[Tuple[float, GenericPoly]]:
        return [
            (confidence, shapely.affinity.scale(geom, xfact=self.px_per_tile, yfact=self.px_per_tile))
            for confidence, geom in polygons
        ]

    def _translate(self, polygons: List[Tuple[float, GenericPoly]]) -> List[Tuple[float, GenericPoly]]:
        return [
            (confidence, shapely.affinity.translate(geom, xoff=self.top_left_tile[0], yoff=self.top_left_tile[1]))
            for confidence, geom in polygons
        ]

    def _clip_by_confidence(self, polygons: List[Tuple[float, GenericPoly]]) -> List[Tuple[str, float, GenericPoly]]:
        """Clips the polygons based on confidence ordering, also creates 'row' UIDs for each polygon"""
        polygons.sort(reverse=True)
        new_polygons = []
        region = None
        for i, (confidence, poly) in enumerate(polygons):
            uid = strings_to_uuid_v5(str(i))
            if i == 0:
                newpoly = poly
                region = poly  # highest confidence, no need to do anything
            else:
                newpoly = poly.difference(region)  # clip to region
                region = region.union(poly)  # update region
            if not newpoly.is_valid:
                raise RuntimeError(
                    f"Invalid geometry after clip: {explain_validity(poly)=}, {explain_validity(newpoly)=}"
                )
            new_polygons.append((uid, confidence, newpoly))
        return new_polygons


@dataclasses.dataclass(frozen=True)
class RandomTargetsGenerator(RandomPolyGenerator):
    min_arc_length: float = 0.01
    radius_step: float = 0.01

    def _fill_unit_square(self) -> List[Tuple[float, GenericPoly]]:
        """See RandomExample._fill_unit_square"""
        polygons = []
        radius = 0.5 - self.radius_step
        while radius > 4 * self.radius_step:
            polygons.append(self._make_ring(radius))
            radius -= 2 * self.radius_step
        return polygons

    def _make_ring(self, radius: float) -> Tuple[float, GenericPoly]:
        outer = self._make_circle(radius)
        for _ in range(10):
            inner_radius = radius - self.radius_step
            inner = self._make_circle(inner_radius)
            poly = outer.difference(inner)
            if poly.is_valid:
                return np.random.uniform(0.5, 1), poly
        raise RuntimeError(f"Couldn't make ring for {radius=}")

    def _make_circle(self, radius: float) -> GenericPoly:
        delta_theta = self.min_arc_length / radius
        n_steps = int(np.round(2 * np.pi / delta_theta))
        theta = np.linspace(0, 2 * np.pi, num=n_steps, endpoint=False)
        radii = radius + np.random.uniform(-0.6 * self.radius_step, 0.6 * self.radius_step, size=n_steps)
        xy = radii * np.vstack((np.cos(theta), np.sin(theta)))
        # shift into the unit square
        xy[0, :] += 0.5
        xy[1, :] += 0.5
        return shapely.geometry.Polygon(xy.T)
