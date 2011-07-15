from environment import env, command

@command
def samplex_random_seed(s):
    env().model_gen_options['rngseed'] = s

@command
def samplex_objective_function(type):
    assert type in ['facet', 'random']
    env().model_gen_options['objf choice'] = type

@command
def samplex_solution_type(type):
    assert type in ['vertex', 'interior']
    env().model_gen_options['solution type'] = type

@command
def samplex_add_noise(n=True):
    env().model_gen_options['add noise'] = n
