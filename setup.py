import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

incdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
csamplex = Extension('solvers.samplex.csamplex',
                     sources = ['solvers/samplex/csamplex.c'],
		     include_dirs=[incdir],
             undef_macros=['DEBUG'],
             extra_compile_args=['-Wall', 
                                 '-ftree-vectorizer-verbose=2', '-ftree-vectorize',
                                 '-fno-omit-frame-pointers'])

setup(name = 'Glass',
      author = 'Jonathan Coles',
      author_email = 'jonathan@physik.uzh.ch',
      version = '1.0',
      description = 'Gravitational Lensing and Something Something',
      package_dir = {'glass': ''},
      packages = ['', 'solvers', 'solvers.samplex',
                  'basis', 'basis.pixels', 'basis.bessel',
                  'massmodel', 'misc'],
      ext_modules = [csamplex])

