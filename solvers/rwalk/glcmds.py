from environment import env, command

@command
def samplex_random_seed(s):
    env().model_gen_options['rngseed'] = s

@command
def samplex_add_noise(n=1e-6):
    assert 0, 'samplex_add_noise: DEPRECATED FUNCTION'
    env().model_gen_options['add noise'] = n

@command
def samplex_stride(s=1):
    assert 0, 'samplex_stride: DEPRECATED FUNCTION'
    env().model_gen_options['stride'] = s

@command
def samplex_acceptance(rate=0.25, tol=0.05):
    env().model_gen_options['acceptance rate'] = rate
    env().model_gen_options['acceptance tol'] = tol

@command
def samplex_redo_factor(f):
    env().model_gen_options['redo factor'] = f

