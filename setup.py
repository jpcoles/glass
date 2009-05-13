import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

incdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
csamplex = Extension('csamplex',
                     sources = ['solvers/samplex/csamplex.c'],
		     include_dirs=[incdir],
             undef_macros=['DEBUG'],
             extra_compile_args=['-Wall'])

setup (name = 'Glass',
       version = '1.0',
       description = 'Gravitational Lensing and Something Something',
       ext_modules = [csamplex])

