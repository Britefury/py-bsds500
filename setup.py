import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(
           os.path.join('bsds', 'correspond_pixels.pyx'),                 # our Cython source
           language="c++",
      ))
