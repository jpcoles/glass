from environment import env, command

@command
def samplex_random_seed(s):
    env().model_gen_options['rngseed'] = s

@command
def samplex_add_noise(n=1e-6):
    env().model_gen_options['add noise'] = n

@command
def samplex_stride(s=1):
    env().model_gen_options['stride'] = s

@command
def samplex_acceptance(rate=0.25, tol=0.05):
    env().model_gen_options['acceptance rate'] = rate
    env().model_gen_options['acceptance tol'] = tol

