import sys, os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib, get_python_inc
import distutils.sysconfig 
from distutils.ccompiler import new_compiler

compiler='clang'
#compiler='gcc'
#compiler='gcc-6'
#compiler='icc'

try:
    import numpy
except ImportError:
    print 'Could not import numpy, which is necessary for compilation.'
    sys.exit(1)

libraries=[#'profiler', 'gomp'
]
#libraries=None #['rt']

extra_link_args = []
extra_compile_args = []

if compiler=='clang':
    extra_compile_args = ['-msse3', 
                          '-fno-omit-frame-pointer',
                          '-funroll-loops',
                          '-fprefetch-loop-arrays',
                          '-fstrict-aliasing',
                          '-std=c99',
                          '-O3',
                          '-mtune=native',
                          '-Wall']

if compiler=='gcc':
    extra_compile_args = ['-msse3', 
                          #'-mfpmath=sse',
                          '-ftree-vectorize',
                          '-ftree-vectorizer-verbose=4',
                          #'-funsafe-math-optimizations',
                          '-fno-omit-frame-pointer',
                          #'-floop-optimize2',
                          '-funroll-loops',
                          '-fprefetch-loop-arrays',
                          '-fstrict-aliasing',
                          '-mpreferred-stack-boundary=4',
                          '-std=c99',
                          #'-I/local/ATLAS/include',
                          #'-malign-double',
                          #'-march=core2',
                          '-O3',
                          '-mtune=native',
                          '-openmp',
                          '-Wall']
    extra_link_args = ['-lgomp']

if compiler=='gcc-6':
    extra_compile_args = ['-msse3', 
                          #'-mfpmath=sse',
                          '-ftree-vectorize',
                          '-ftree-vectorizer-verbose=4',
                          #'-funsafe-math-optimizations',
                          '-fno-omit-frame-pointer',
                          #'-floop-optimize2',
                          '-funroll-loops',
                          '-fprefetch-loop-arrays',
                          '-fstrict-aliasing',
                          '-mpreferred-stack-boundary=4',
                          '-std=c99',
                          #'-I/local/ATLAS/include',
                          #'-malign-double',
                          #'-march=core2',
                          '-O3',
                          '-mtune=native',
                          '-openmp',
                          '-Wall']
    extra_link_args = ['-lgomp']

if compiler=='icc':
    extra_compile_args = [
                          '-fno-omit-frame-pointer',
                          '-funroll-loops',
                          '-fstrict-aliasing',
                          '-std=c99',
                          '-vec-report=0',
                          '-par-report=0',
                          '-O3',
                          '-xHost',
                          '-mtune=native',
                          '-Wall',
                          '-openmp']

try:
    numpy_inc = [numpy.get_include()]
except AttributeError:
    numpy_inc = [numpy.get_numpy_include()]

crwalk = Extension('glass.solvers.rwalk.csamplex',
                     sources = ['glass/solvers/rwalk/csamplex_omp.c', 'glass/solvers/rwalk/WELL44497a.c'],
		     include_dirs=numpy_inc,
             undef_macros=['DEBUG'],
             libraries=libraries,
             extra_compile_args=extra_compile_args,
             extra_link_args=extra_link_args)

setup(name = 'Glass',
      author = 'Jonathan Coles',
      author_email = 'jonathan@jpcoles.com',
      version = '1.0',
      description = 'Gravitational Lensing AnalysiS Software',
      package_dir = {'glass': 'glass'},
      packages = ['', 'glass', 
                  'glass.solvers', 'glass.solvers.rwalk',
                  'glass.basis', 'glass.basis.pixels', 'glass.basis.bessel',
                  'glass.massmodel', 'glass.misc'],
      ext_modules = [crwalk])

