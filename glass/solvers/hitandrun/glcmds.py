from glass.command import command
from glass.log import log as Log

@command
def samplex_random_seed(env, s):
    env.model_gen_options['rngseed'] = s

@command
def samplex_add_noise(env, n=1e-6):
    assert 0, 'samplex_add_noise: DEPRECATED FUNCTION'

@command
def samplex_burnin_factor(env, b):
    assert b > 0
    env.model_gen_options['burnin factor'] = b

@command
def samplex_thin(env, thin):
    assert thin > 0
    if thin < 1000:
        Log('*** Samplex: The current "thinning" value is %i. It is recommended to be at least 1000. ***' % thin)
    env.model_gen_options['thin'] = thin
