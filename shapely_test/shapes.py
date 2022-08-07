import abc
import dataclasses
import gzip
import io
import json
from pathlib import Path
from typing import List, Tuple, Union
import uuid

import geopandas as gpd
import numpy as np
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
from shapely.validation import explain_validity

from ._compat import SHAPELY_GE_20

GenericPoly = Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]


def strings_to_uuid_v5(*args: str) -> str:
    """Creates a reproducible hash of the argument strings into a v5 uuid string"""
    if not args:
        raise ValueError("You must pass at least one string")
    _NAMESPACE_UUID = uuid.UUID(int=0)
    return str(uuid.uuid5(_NAMESPACE_UUID, "_".join(args)))


def clip_to_unit_square(gdf: gpd.GeoDataFrame, inplace=True) -> gpd.GeoDataFrame:
    bbox = shapely.geometry.box(0, 0, 1, 1)
    if SHAPELY_GE_20:
        new_geoms = shapely.intersection(gdf.geometry, bbox)
    else:
        new_geoms = gdf.geometry.apply(lambda g: g.intersection(bbox))
    return gdf.set_geometry(new_geoms, inplace=inplace)


def scale(gdf: gpd.GeoDataFrame, xfact: float, yfact: float, inplace=True) -> gpd.GeoDataFrame:
    if SHAPELY_GE_20:
        new_geoms = shapely.transform(gdf.geometry, lambda x: x * [xfact, yfact])
    else:
        new_geoms = gdf.geometry.apply(lambda g: shapely.affinity.scale(g, xfact, yfact, origin=(0, 0)))
    return gdf.set_geometry(new_geoms, inplace=inplace)


def translate(gdf: gpd.GeoDataFrame, xoff: float, yoff: float, inplace=True) -> gpd.GeoDataFrame:
    if SHAPELY_GE_20:
        new_geoms = shapely.transform(gdf.geometry, lambda x: x + [xoff, yoff])
    else:
        new_geoms = gdf.geometry.apply(lambda g: shapely.affinity.translate(g, xoff, yoff))
    return gdf.set_geometry(new_geoms, inplace=inplace)


def snap_to_integers(gdf: gpd.GeoDataFrame, inplace=True) -> gpd.GeoDataFrame:
    """Round coordinates to the nearest int, with .5 rounded towards +inf"""
    if SHAPELY_GE_20:
        new_geoms = shapely.set_precision(gdf.geometry, 1)
    else:
        new_geoms = gdf.geometry.apply(snap_geom_to_integers)
    return gdf.set_geometry(new_geoms, inplace=inplace)


def snap_geom_to_integers(geom: GenericPoly) -> GenericPoly:
    """Round coordinates to the nearest int, with .5 rounded towards +inf"""
    return shapely.ops.transform(lambda x, y: tuple(np.floor(np.asarray((x, y)) + 0.5)), geom)


@dataclasses.dataclass(frozen=True)
class RandomPolyGenerator(abc.ABC):
    top_left_tile: Tuple[int, int] = (18807, 23776)  # chosen only so that it can be loaded into QGIS easily (@z=16)
    px_per_tile: int = 256 * 2 ** (21 - 16)  # This gives the number of z21 pixels in a single z16 tile
    use_cached_data_if_it_exists: bool = False
    snap_to_integers: bool = True

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Get the name of this generator"""

    @staticmethod
    def load_geojson_gz(path: Path) -> gpd.GeoDataFrame:
        assert path.name.endswith(".geojson.gz"), f"{path=} has invalid extension"
        features = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        return gpd.GeoDataFrame.from_features(features)

    @staticmethod
    def to_geojson_gz(gdf: gpd.GeoDataFrame, path: Path):
        assert path.name.endswith(".geojson.gz"), f"{path=} has invalid extension"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fout, io.BytesIO() as stream:
            gdf.to_file(stream, driver="GeoJSON")
            fout.write(gzip.compress(stream.getvalue(), compresslevel=9))

    def __call__(self, seed: int) -> gpd.GeoDataFrame:
        """Use this method to fill the unit square with polygons and their associated 'confidence'"""
        outpath = Path(__file__).resolve().parent / f"cache/{self.name()}-{seed}.geojson.gz"
        if outpath.exists() and self.use_cached_data_if_it_exists:
            gdf = self.load_geojson_gz(outpath)
            # set the precision in shapely v2
            if SHAPELY_GE_20:
                if self.snap_to_integers:
                    snap_to_integers(gdf)
            return gdf
        np.random.seed(seed)

        polys = self._fill_unit_square()
        polys = self._clip_by_confidence(polys)
        gdf = gpd.GeoDataFrame(polys, columns=["uid", "confidence", "geometry"], geometry="geometry")

        clip_to_unit_square(gdf)
        scale(gdf, xfact=self.px_per_tile, yfact=self.px_per_tile)
        translate(gdf, xoff=self.top_left_tile[0], yoff=self.top_left_tile[1])
        if self.snap_to_integers:
            snap_to_integers(gdf)

        self.to_geojson_gz(gdf, outpath)
        return gdf

    @abc.abstractmethod
    def _fill_unit_square(self) -> List[Tuple[float, GenericPoly]]:
        """Use this method to fill the unit square with polygons and their associated 'confidence'"""

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
class RandomSpotsGenerator(RandomPolyGenerator):
    vertices_per_spot: int = 20
    mean_radius: float = 0.01
    n_spots: int = 900

    @classmethod
    def name(cls) -> str:
        """See RandomExample.name"""
        return "spots"

    def _fill_unit_square(self) -> List[Tuple[float, GenericPoly]]:
        """See RandomExample._fill_unit_square"""
        polygons = []
        centers = np.random.uniform(-0.5, 0.5, size=(self.n_spots, 2))
        confidences = np.random.uniform(0.5, 1, size=self.n_spots)
        radii = np.random.normal(self.mean_radius, 0.5 * self.mean_radius, size=self.n_spots)
        for confidence, radius, (cx, cy) in zip(confidences, radii, centers):
            polygons.append((confidence, self._make_spot(radius, cx, cy)))
        return polygons

    def _make_spot(self, radius: float, cx: float, cy: float) -> GenericPoly:
        for _ in range(10):
            poly = self._make_circle(radius, cx, cy)
            if poly.is_valid:
                return poly
        raise RuntimeError(f"Couldn't make ring for {radius=}")

    def _make_circle(self, radius: float, cx: float, cy: float) -> GenericPoly:
        theta = np.linspace(0, 2 * np.pi, num=self.vertices_per_spot, endpoint=False)
        radii = radius * (1 + np.random.uniform(-0.25, 0.25, size=self.vertices_per_spot))
        xy = radii * np.vstack((np.cos(theta), np.sin(theta)))
        # shift into the unit square
        xy[0, :] += 0.5 + cx
        xy[1, :] += 0.5 + cy
        return shapely.geometry.Polygon(xy.T)


@dataclasses.dataclass(frozen=True)
class RandomCenterTargetsGenerator(RandomPolyGenerator):
    min_arc_length: float = 0.01
    radius_step: float = 0.01

    @classmethod
    def name(cls) -> str:
        """See RandomExample.name"""
        return "random_center_targets"

    def _fill_unit_square(self) -> List[Tuple[float, GenericPoly]]:
        """See RandomExample._fill_unit_square"""
        polygons = []
        cx, cy = np.random.uniform(-0.25, 0.25, size=2)
        radius = 0.5 - self.radius_step
        while radius > 4 * self.radius_step:
            polygons.append(self._make_ring(radius, cx, cy))
            radius -= 1.5 * self.radius_step
        return polygons

    def _make_ring(self, radius: float, cx: float, cy: float) -> Tuple[float, GenericPoly]:
        outer = self._make_circle(radius, cx, cy)
        for _ in range(10):
            inner_radius = radius - self.radius_step
            inner = self._make_circle(inner_radius, cx, cy)
            poly = outer.difference(inner)
            if poly.is_valid:
                return np.random.uniform(0.5, 1), poly
        raise RuntimeError(f"Couldn't make ring for {radius=}")

    def _make_circle(self, radius: float, cx: float, cy: float) -> GenericPoly:
        delta_theta = self.min_arc_length / radius
        n_steps = int(np.round(2 * np.pi / delta_theta))
        theta = np.linspace(0, 2 * np.pi, num=n_steps, endpoint=False)
        xy = radius * np.vstack((np.cos(theta), np.sin(theta)))
        # shift into the unit square
        xy[0, :] += 0.5 + cx
        xy[1, :] += 0.5 + cy
        return shapely.geometry.Polygon(xy.T)


@dataclasses.dataclass(frozen=True)
class RandomRadiusTargetsGenerator(RandomPolyGenerator):
    min_arc_length: float = 0.01
    radius_step: float = 0.01

    @classmethod
    def name(cls) -> str:
        """See RandomExample.name"""
        return "random_radius_targets"

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
