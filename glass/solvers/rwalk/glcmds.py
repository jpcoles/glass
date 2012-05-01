from glass.command import command

@command
def samplex_random_seed(env, s):
    env.model_gen_options['rngseed'] = s

@command
def samplex_add_noise(env, n=1e-6):
    assert 0, 'samplex_add_noise: DEPRECATED FUNCTION'

@command
def samplex_stride(env, s=1):
    assert 0, 'samplex_stride: DEPRECATED FUNCTION'

@command
def samplex_acceptance(env, rate=0.25, tol=0.05):
    assert rate > 0
    assert tol > 0
    env.model_gen_options['acceptance rate'] = rate
    env.model_gen_options['acceptance tol'] = tol

@command
def samplex_redo_factor(env, f):
    assert f > 0
    env.model_gen_options['redo factor'] = f

@command
def samplex_redo_exponent(env, e):
    env.model_gen_options['redo exp'] = e

@command
def samplex_start_twiddle(env, t):
    assert t > 0
    env.model_gen_options['twiddle'] = t

@command
def samplex_burnin_factor(env, b):
    assert b > 0
    env.model_gen_options['burnin factor'] = b

