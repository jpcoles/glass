from glass.command import command

@command
def samplex_random_seed(env,s):
    env.model_gen_options['rngseed'] = s

@command
def samplex_objective_function(env,type):
    assert type in ['facet', 'random']
    env.model_gen_options['objf choice'] = type

@command
def samplex_solution_type(env,type):
    assert type in ['vertex', 'interior']
    env.model_gen_options['solution type'] = type

@command
def samplex_add_noise(env,n=True):
    env.model_gen_options['add noise'] = n

@command
def samplex_stride(env,s=1):
    env.model_gen_options['stride'] = s
