from glass.environment import command

@command
def samplex_random_seed(env, s):
    env.model_gen_options['rngseed'] = s

@command
def samplex_add_noise(env, n=1e-6):
    assert 0, 'samplex_add_noise: DEPRECATED FUNCTION'
    env.model_gen_options['add noise'] = n

@command
def samplex_stride(env, s=1):
    assert 0, 'samplex_stride: DEPRECATED FUNCTION'
    env.model_gen_options['stride'] = s

@command
def samplex_acceptance(env, rate=0.25, tol=0.05):
    env.model_gen_options['acceptance rate'] = rate
    env.model_gen_options['acceptance tol'] = tol

@command
def samplex_redo_factor(env, f):
    env.model_gen_options['redo factor'] = f

@command
def samplex_start_twiddle(env, t):
    env.model_gen_options['twiddle'] = t
