from setuptools import find_packages, setup

setup(
    name = 'HousePricePrediction',
    version = 0.2,
    packages = find_packages(where="src"),
    package_dir = {"":"src"}
)