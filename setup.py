from setuptools import find_packages, setup

name = "shapely_test"
pysrc_dir = "."
packages = [p for p in find_packages(pysrc_dir) if not p.startswith("tests")]
package_dir = {"": pysrc_dir}

setup(
    name=name,
    version="0.0.0",
    description="Application for testing shapely v2",
    url=f"https://github.com/bretttully/shapely-v2",
    author="Brett Tully",
    author_email="brett.tully@gmail.com",
    packages=packages,
    package_dir=package_dir,
    include_package_data=True,
    zip_safe=False,
)
