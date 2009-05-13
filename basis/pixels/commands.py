from environment import env
from basis import PixelBasis as basis_class

def globject(name):
    co = env.new_object(name)
    co.basis = basis_class()

def pixrad(r):
    env.current_object().basis.L = r

def priors(*ps):
    env.current_object().basis.prior_list = ps

