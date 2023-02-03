from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="fps_cython",
    ext_modules=cythonize("./models/fps_cython.pyx"),
    include_dirs=[np.get_include()]
)