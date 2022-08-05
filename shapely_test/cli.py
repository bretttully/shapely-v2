import enum
import os

import typer

from shapely_test import shapes
from shapely_test.cases import run_execution_and_timing_test

app = typer.Typer()


class GeneratorType(str, enum.Enum):
    targets = "targets"

    def build(self) -> shapes.RandomPolyGenerator:
        if self == self.targets:
            return shapes.RandomTargetsGenerator()
        raise NotImplementedError(f"Invalid {self = }")


@app.command()
def run_eatt(
    name: str,
    generator: GeneratorType,
    n_procs: int = typer.Option(-1, "--nprocs", "-j", help="Number of processes to use. -1 means all available"),
    n_tests: int = typer.Option(-1, "--ntests", "-n", help="Number of tests to run. -1 run 10x the nprocs"),
):
    """Run the execution and timing test"""
    n_procs = os.cpu_count() if n_procs < 0 else n_procs
    n_tests = 10 * n_procs if n_tests < 0 else n_tests
    assert n_procs > 0 and n_tests > 0
    df = run_execution_and_timing_test(generator.build(), n_procs, n_tests)
    df.to_csv(f"test-{generator.value}-{name}.csv.gz", index=False, compression="gzip")
