from setuptools import setup, find_packages

setup(
    name="HIVOpen",
    versoion="0.1.0",
    description="Extension of the SEIRS model to HIV dynamics allowing for open cohorts",
    author="Ryan O'Dea, Emily Stewart, Patrick Thorton",
    packages=find_packages(),
    include_package_data=True,
    package_data={"HIVOpen": ["data/*.csv"]},
    install_requires=["numpy", "pandas", "scipy", "matplotlib"],
)
