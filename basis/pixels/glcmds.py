from __future__ import division
from environment import env
from basis import PixelBasis as basis_class
from solvers.samplex.samplex import Samplex

def globject(name):
    co = env.new_object(name)
    co.basis = basis_class()
    env.model_gen_factory = lambda nvars: Samplex(nvars, nthreads=env.ncpus)

def pixrad(r):
    env.current_object().basis.pixrad = r

def priors(*ps):
    env.current_object().basis.prior_list = ps

