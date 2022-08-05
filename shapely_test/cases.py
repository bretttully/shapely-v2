import concurrent.futures
import dataclasses
import os
import time
from typing import Dict

import geopandas as gpd
import pandas as pd
import shapely

from shapely_test.shapes import RandomPolyGenerator, snap_to_integers


@dataclasses.dataclass(frozen=True)
class TestCase:
    data_generator: dataclasses.InitVar[RandomPolyGenerator]
    seed: dataclasses.InitVar[RandomPolyGenerator]
    gdf1: gpd.GeoDataFrame = dataclasses.field(init=False)
    gdf2: gpd.GeoDataFrame = dataclasses.field(init=False)

    def __post_init__(self, data_generator, seed):
        object.__setattr__(self, "gdf1", data_generator(seed))
        object.__setattr__(self, "gdf2", data_generator(seed + 1_000_000_000))
        _ = self.gdf1.sindex, self.gdf2.sindex  # pre-calculate to avoid impacting timings

    def gdf1_is_valid(self) -> bool:
        return self.gdf1.is_valid.all()

    def gdf2_is_valid(self) -> bool:
        return self.gdf2.is_valid.all()

    def gdf1_gdf1_overlay_intersection(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf1, self.gdf1, how="intersection", keep_geom_type=True))

    def gdf1_gdf2_overlay_intersection(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf1, self.gdf2, how="intersection", keep_geom_type=True))

    def gdf2_gdf2_overlay_intersection(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf2, self.gdf2, how="intersection", keep_geom_type=True))

    def gdf1_gdf1_overlay_union(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf1, self.gdf1, how="union", keep_geom_type=True))

    def gdf1_gdf2_overlay_union(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf1, self.gdf2, how="union", keep_geom_type=True))

    def gdf2_gdf2_overlay_union(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf2, self.gdf2, how="union", keep_geom_type=True))

    def gdf1_gdf1_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf1, self.gdf1, how="difference", keep_geom_type=True))

    def gdf1_gdf2_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf1, self.gdf2, how="difference", keep_geom_type=True))

    def gdf2_gdf2_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf2, self.gdf2, how="difference", keep_geom_type=True))

    def gdf2_gdf1_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._snap_to_integers(gpd.overlay(self.gdf2, self.gdf1, how="difference", keep_geom_type=True))

    @staticmethod
    def _snap_to_integers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf["geometry"] = gdf["geometry"].apply(snap_to_integers)
        return gdf


@dataclasses.dataclass(frozen=True)
class ExecutionAndTimingTest:
    data_generator: RandomPolyGenerator

    def __call__(self, seed: int) -> pd.DataFrame:
        test_case = TestCase(self.data_generator, seed)
        functions = [f for f in dir(test_case) if f.startswith("gdf1_") or f.startswith("gdf2_")]
        results = []
        for func in functions:
            t0 = time.monotonic()
            try:
                gdf = getattr(test_case, func)()
                err = None if not isinstance(gdf, gpd.GeoDataFrame) or gdf.is_valid.all() else "Invalid geometries"
            except Exception as e:
                err = str(e)
            finally:
                elapsed = time.monotonic() - t0
            results.append(
                {
                    "shapely_version": shapely.__version__,
                    "data_generator": self.data_generator.name(),
                    "seed": seed,
                    "function": func,
                    "error": err,
                    "time_ms": 1000 * elapsed,
                }
            )
        return pd.DataFrame(results)


def run_execution_and_timing_test(
    data_generator: RandomPolyGenerator, n_procs: int = os.cpu_count(), n_tests: int = 10 * os.cpu_count()
) -> pd.DataFrame:
    test = ExecutionAndTimingTest(data_generator)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_procs, n_tests)) as pool:
        return pd.concat(pool.map(test, range(n_tests)), ignore_index=True)
