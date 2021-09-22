from distutils.core import setup, Extension


module1 = Extension('rotations',
                    define_macros=[('MAJOR_VERSION', '1'),
                                   ('MINOR_VERSION', '0')],
                    # include_dirs=['/User/bytedance/miniconda3/include'],
                    sources=['rotations.cpp'])

setup(name='rotations',
      version='1.0',
      description='Rotation utilities for pybullet',
      ext_modules=[module1])
