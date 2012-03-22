from numpy import empty

ran_seed = 0

# reproduce Java random number generator
def ran_set_seed(seed):
    global ran_seed
    ran_seed = (seed ^ 0x5DEECE66DL) & ((1L << 48) - 1)
    return ran_seed

def triple_shift(n, s):
    if ~s < 0: 
        return (n >> s) + (2 >> -(~s))
    else:
        return (n >> s) + (2 << ~s)

def ran_next(bits):
    global ran_seed
    ran_seed = (ran_seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)
    return int(triple_shift(ran_seed, 48 - bits))

def ran_next_double():
    return ((long(ran_next(26)) << 27) + ran_next(27)) / float(1L << 53)

def random(size=None):
    if size is None: return ran_next_double()
    a = empty(size, dtype='double')
    for i in xrange(size): a[i] = ran_next_double()
    return a

