import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

libraries=None #['rt']
extra_compile_args = [#'-mssse3', 
                      #'-mfpmath=sse',
                      '-ftree-vectorizer-verbose=3', '-ftree-vectorize',
                      '-fno-omit-frame-pointer',
                      '-floop-optimize2',
                      '-funroll-loops',
                      '-fprefetch-loop-arrays',
                      '-fstrict-aliasing',
                      '-fnested-functions',
                      #'-malign-double',
                      #'-march=core2',
                      #'-O0',
                      '-Wall']

#extra_compile_args += ['-arch_only i386']


incdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
csamplex = Extension('solvers.samplex.csamplex',
                     sources = ['solvers/samplex/csamplex.c'],
		     include_dirs=[incdir],
             undef_macros=['DEBUG'],
             libraries=libraries,
             extra_compile_args=extra_compile_args)

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

