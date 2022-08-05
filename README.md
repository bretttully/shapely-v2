# shapely-v2

Exploring shapely v2 and specifically the performance of the overlay methods


# Installation

There are two environment yaml files provided, one for `shapely 2.0a` and one for `shapely 1.8` that can be installed using the following

```bash
mamba env create -f .conda/environment-v20a1.yaml
mamba env create -f .conda/environment-v182.yaml
```

Set up inplace installation in each env

```bash
conda activate shapely-v2
python setup.py develop
```

and

```bash
conda activate shapely-v1
python setup.py develop
```

# Execution

Using jupyter (particularly from a tool like VS Code) we can then select the environment in which we wish to run the notebooks in this repo.

Executing [`generators.ipynb`](https://github.com/bretttully/shapely-v2/blob/main/generators.ipynb) shows the different types of geometries that we can create.

[`timing_and_execution_test-v182.ipynb`](https://github.com/bretttully/shapely-v2/blob/main/timing_and_execution_test-v182.ipynb) and [`timing_and_execution_test-v20a1.ipynb`](https://github.com/bretttully/shapely-v2/blob/main/timing_and_execution_test-v20a1.ipynb) shows the result of running the timing and exection test for the two different shapely versions.
