# shapely-v2

Exploring shapely v2 and specifically the performance of the overlay methods


# Installation

There are two environment yaml files provided, one for `shapely 2.0a` and one for `shapely 1.8` that can be installed using the following

```bash
mamba env create -f .conda/environment-v20a1.yaml
mamba env create -f .conda/environment-v182.yaml
```

# Execution

In each of those env, we can then run some tests. For example, in the v2 env we can run the random "targets" test using the following:

```bash
conda activate shapely-v2
python setup.py develop
shapely_test --help
shapely_test v2 random_radius_targets -j 2 -n 4
```

To switch to the other env, first `conda deactivate` and then run:

```bash
conda activate shapely-v1
python setup.py develop
shapely_test --help
shapely_test v1 random_radius_targets -j 2 -n 4
```