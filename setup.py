from setuptools import find_namespace_packages, setup

setup(
    name="lekayla",
    version="0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
)
