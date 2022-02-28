#!/usr/bin/env python

"""The setup script."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "**/*",
        ["**/*.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(
        extensions, language_level=3, annotate=True, gdb_debug=True, nthreads=4
    ),
)
