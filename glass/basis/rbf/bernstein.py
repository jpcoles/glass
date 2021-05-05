from scipy.special import comb


def B(n,k,t):
    return ( comb(n,k) * (1-t)**(n-k) * (x**k) )

