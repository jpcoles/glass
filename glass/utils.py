import numpy as np

def dist_range(a, sigma='1sigma'):
    frac = {'1sigma': 0.6827,
            '2sigma': 0.9545,
            '3sigma': 0.9973,
            'all'   : 1.0}.get(sigma, sigma)
    N = len(a)
    s = np.sort(a, axis=0)
    n = int(frac/2. * N)
    mid = (N-1) // 2
    if N % 2 == 0:
        M = (s[mid] + s[mid+1]) / 2
        L = s[max(mid   - n, 0)]
        R = s[min(mid+1 + n, len(s)-1)]
    else:
        M = s[mid]
        L = s[mid - n]
        R = s[mid + n]
    return M, R, L
