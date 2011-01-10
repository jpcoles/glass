from environment import env

def samplex_random_seed(s):
    env().model_gen_options['rngseed'] = s

