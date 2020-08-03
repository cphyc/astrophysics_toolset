#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "scipy>=1.4.1",
    "numpy>=1.10",
    "kaitaistruct>=0.8",
    "numba>=0.40",
    "cython-fortran-file>=0.2.2",
    "yt>=3.6",
]
setup_requirements = [
    "pytest-runner",
]

test_requirements = ["pytest==5.4.1", "mpmath==1.1.0"]

extensions = [
    Extension(
        "**/*",
        ["**/*.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    author="Corentin Cadiou",
    author_email="c.cadiou@ucl.ac.uk",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python package with some astrophysics-related stuff",
    entry_points={
        "console_scripts": [
            "astrophysics_toolset=astrophysics_toolset.cli:main",
            "ramses_change_ncpu=astrophysics_toolset.ramses.ramses_change_ncpu",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="astrophysics_toolset",
    name="astrophysics_toolset",
    packages=find_packages(include=["astrophysics_toolset", "astrophysics_toolset.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/cphyc/astrophysics_toolset",
    version="0.2.3",
    zip_safe=False,
    ext_modules=cythonize(
        extensions, language_level=3, annotate=True, gdb_debug=True, nthreads=4
    ),
)
