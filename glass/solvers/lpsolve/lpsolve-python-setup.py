from distutils.core import setup, Extension
from os import getenv
import sys
import os
p = sys.prefix
print p
NUMPYPATH = '.'
#/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy
if os.path.isdir(p + '/include/numpy'):
  NUMPY = 'NUMPY'
elif os.path.isdir(p + '/Lib/site-packages/numpy/core/include/numpy'):
  NUMPY = 'NUMPY'
  NUMPYPATH = p + '/Lib/site-packages/numpy/core/include'
elif os.path.isdir(p + '/lib/python2.6/site-packages/numpy/core/include/numpy'):
  NUMPY = 'NUMPY'
  NUMPYPATH = p + '/lib/python2.6/site-packages/numpy/core/include'
else:
  NUMPY = 'NONUMPY'
print 'numpy: ' + NUMPY
windir = getenv('windir')
if windir == None:
  WIN32 = 'NOWIN32'
  LPSOLVE55 = '../../lpsolve55'
else:
  WIN32 = 'WIN32'
  LPSOLVE55 = '../../lpsolve55/bin/win32'
setup (name = "lpsolve55",
       version = "5.5.0.9",
       description = "Linear Program Solver, Interface to lpsolve",
       author = "Peter Notebaert",
       author_email = "lpsolve@peno.be",
       url = "http://www.peno.be/",
       py_modules=['lp_solve', 'lp_maker'],
       ext_modules = [Extension("lpsolve55",
				["lpsolve.c", "hash.c", "pythonmod.c"],
                                define_macros=[('PYTHON', '1'), (WIN32, '0'), ('NODEBUG', '1'), ('DINLINE', 'static'), (NUMPY, '1'), ('_CRT_SECURE_NO_WARNINGS', '1')],
                                include_dirs=['../..', NUMPYPATH],
                                library_dirs=[LPSOLVE55],
				libraries = ["lpsolve55"])
		      ]
)
