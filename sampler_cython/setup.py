# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "utils",                      # module name
        ["utils.pyx"],
        include_dirs=[np.get_include()],
        libraries=["lapacke", "lapack", "blas"],  # link against LAPACKE, LAPACK, and BLAS
    )
]

setup(
    name="utils",
    ext_modules=cythonize(ext_modules),
)