import concurrent.futures
import dataclasses
import os
from pprint import pprint
import time
from typing import Dict
import shapely

import geopandas as gpd
import pandas as pd

from shapely_test.shapes import RandomPolyGenerator


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

    def gdf1_gdf1_overlay_intersection(self):
        return gpd.overlay(self.gdf1, self.gdf1, how="intersection", keep_geom_type=True)

    def gdf1_gdf2_overlay_intersection(self):
        return gpd.overlay(self.gdf1, self.gdf2, how="intersection", keep_geom_type=True)

    def gdf2_gdf2_overlay_intersection(self):
        return gpd.overlay(self.gdf2, self.gdf2, how="intersection", keep_geom_type=True)

    def gdf1_gdf1_overlay_union(self):
        return gpd.overlay(self.gdf1, self.gdf1, how="union", keep_geom_type=True)

    def gdf1_gdf2_overlay_union(self):
        return gpd.overlay(self.gdf1, self.gdf2, how="union", keep_geom_type=True)

    def gdf2_gdf2_overlay_union(self):
        return gpd.overlay(self.gdf2, self.gdf2, how="union", keep_geom_type=True)

    def gdf1_gdf1_overlay_difference(self):
        return gpd.overlay(self.gdf1, self.gdf1, how="difference", keep_geom_type=True)

    def gdf1_gdf2_overlay_difference(self):
        return gpd.overlay(self.gdf1, self.gdf2, how="difference", keep_geom_type=True)

    def gdf2_gdf2_overlay_difference(self):
        return gpd.overlay(self.gdf2, self.gdf2, how="difference", keep_geom_type=True)


@dataclasses.dataclass(frozen=True)
class ExecutionAndTimingTest:
    data_generator: RandomPolyGenerator

    def __call__(self, seed: int) -> Dict:
        test_case = TestCase(self.data_generator, seed)
        funcs = {
            "gdf1_gdf1_overlay_intersection": test_case.gdf1_gdf1_overlay_intersection,
            "gdf1_gdf2_overlay_intersection": test_case.gdf1_gdf2_overlay_intersection,
            "gdf2_gdf2_overlay_intersection": test_case.gdf2_gdf2_overlay_intersection,
            "gdf1_gdf1_overlay_union": test_case.gdf1_gdf1_overlay_union,
            "gdf1_gdf2_overlay_union": test_case.gdf1_gdf2_overlay_union,
            "gdf2_gdf2_overlay_union": test_case.gdf2_gdf2_overlay_union,
            "gdf1_gdf1_overlay_difference": test_case.gdf1_gdf1_overlay_difference,
            "gdf1_gdf2_overlay_difference": test_case.gdf1_gdf2_overlay_difference,
            "gdf2_gdf2_overlay_difference": test_case.gdf2_gdf2_overlay_difference,
        }
        result = {"shapely_version": shapely.__version__, "data_generator": str(self.data_generator), "seed": seed}
        for name, func in funcs.items():
            errcol, timecol = f"err_{name}", f"time_{name}"
            t0 = time.monotonic()
            try:
                gdf = func()
                result[errcol] = None if gdf.is_valid.all() else "Invalid geometries"
            except Exception as e:
                result[errcol] = str(e)
            finally:
                result[timecol] = time.monotonic() - t0
        return result


def run_execution_and_timing_test(
    data_generator: RandomPolyGenerator, n_procs: int = os.cpu_count(), n_tests: int = 10 * os.cpu_count()
) -> pd.DataFrame:
    test = ExecutionAndTimingTest(data_generator)

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_procs, n_tests)) as pool:
        cf_result = list(pool.map(test, range(n_tests)))
    result = pd.DataFrame(cf_result)

    time_cols = [c for c in result.columns if c.startswith("time_")]
    print("*" * 80)
    print("Timing")
    print(result[time_cols].describe().T)

    err_cols = [c for c in result.columns if c.startswith("err_")]
    print("*" * 80)
    print("Errors")
    pprint({c: result[c].dropna().unique().tolist() for c in err_cols})

    return result
