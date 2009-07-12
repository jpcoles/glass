from __future__ import division
from environment import env
from basis import PixelBasis as basis_class
from solvers.samplex.samplex import Samplex

def _foo(nvars):
   return Samplex(nvars, nthreads=env().ncpus) 

def globject(name):
    co = env().new_object(name)
    co.basis = basis_class()
    env().model_gen_factory = _foo
    return co

def pixrad(r):
    env().current_object().basis.pixrad = r

def priors(*ps):
    env().current_object().basis.prior_list = ps

def subdiv(n):
    n = int(n)
    assert (n%2==1), "subdiv: n must be odd"
    env().current_object().basis.subdivision = n

def hires(r, refine=1):
    assert r > 0 and refine>=3 and refine%2==1, 'hires: Minimum refinement value is 3. Must be odd too.'
    env().current_object().basis.hiresR       = r
    env().current_object().basis.hires_levels = refine
    
