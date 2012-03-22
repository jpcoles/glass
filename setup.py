import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

libraries=[#'profiler', 'gomp'
]
#libraries=None #['rt']
extra_compile_args = [#'-mssse3', 
                      #'-mfpmath=sse',
                      '-ftree-vectorizer-verbose=4', '-ftree-vectorize',
                      '-funsafe-math-optimizations',
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
                      #'-O0',
                      '-Wall']

#extra_compile_args += ['-fnested-functions']
#extra_compile_args += ['-arch_only x86_64']
extra_link_args = [] #'-L/local/ATLAS/lib']
#extra_compile_args += ['-fopenmp','-msse2']
extra_compile_args += ['-msse2']
#extra_link_args = ['-lgomp']

incdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
csamplex = Extension('glass.solvers.samplex.csamplex',
                     sources = ['glass/solvers/samplex/csamplex_omp.c'],
		     include_dirs=[incdir],
             undef_macros=['DEBUG'],
             libraries=libraries,
             extra_compile_args=extra_compile_args,
             extra_link_args=extra_link_args)

csamplexsimple = Extension('glass.solvers.samplexsimple.csamplex',
                     sources = ['glass/solvers/samplexsimple/csamplex_omp.c'],
		     include_dirs=[incdir],
             undef_macros=['DEBUG'],
             libraries=libraries,
             extra_compile_args=extra_compile_args,
             extra_link_args=extra_link_args)

crwalk = Extension('glass.solvers.rwalk.csamplex',
                     sources = ['glass/solvers/rwalk/csamplex_omp.c', 'glass/solvers/rwalk/WELL44497a.c'],
		     include_dirs=[incdir],
             undef_macros=['DEBUG'],
             libraries=libraries,
             extra_compile_args=extra_compile_args,
             extra_link_args=extra_link_args)

setup(name = 'Glass',
      author = 'Jonathan Coles',
      author_email = 'jonathan@physik.uzh.ch',
      version = '1.0',
      description = 'Gravitational Lensing and Something Something',
      package_dir = {'glass': 'glass'},
      packages = ['', 
                  'glass.solvers', 'glass.solvers.samplex', 'glass.solvers.lpsolve', 
                  'glass.solvers.samplexsimple',
                  'glass.solvers.rwalk',
                  'glass.basis', 'glass.basis.pixels', 'glass.basis.bessel',
                  'glass.massmodel', 'glass.misc'],
      ext_modules = [csamplex, csamplexsimple, crwalk])

