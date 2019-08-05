from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("pythagorean_triples.pyx")
)

# build command
# python setup.py build_ext --inplace
