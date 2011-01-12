from environment import env

def samplex_random_seed(s):
    env().model_gen_options['rngseed'] = s

def samplex_objective_function(type):
    assert type in ['facet', 'random']
    env().model_gen_options['objf choice'] = type

def samplex_solution_type(type):
    assert type in ['vertex', 'interior']
    env().model_gen_options['solution type'] = type
