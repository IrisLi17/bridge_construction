from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension("ikfastpy", 
                            ["ikfastpy.pyx", 
                             "ur_kin_wrapper.cpp"], language="c++")],
      cmdclass = {'build_ext': build_ext})