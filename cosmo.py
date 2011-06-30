from __future__ import division
from math import sin,sinh, sqrt
from numpy import abs
from scipy.integrate import quadrature, quad
from environment import env

def age_factor():
    M = env().omega_matter
    L = env().omega_lambda
    K = 1 - (M+L)
    f = lambda a: a / sqrt(a*M + a**4 * L + a**2 * K)
    q =  quad(f, 0, 1)[0]
    return q

def angdist(zi, zf):
    return _angdist(zi,zf, env().omega_matter, env().omega_lambda, env().filled_beam)

def _angdist(zi, zf, M,L, filled_beam, tol=1e-4):

    if zf < zi:
        zi,zf = zf,zi

    #---------------------------------------------------------------------------
    # Curvature of the universe.
    #---------------------------------------------------------------------------
    tol = 1e-4
    k = 0
    if M+L+tol < 1:
        k = -1
    elif M+L-tol > 1:
        k = 1

    if filled_beam:
        f = lambda z: 1. / sqrt(M * (z+1)**3 + (1-M-L) * (z+1)**2 + L)
    else:
        f = lambda z: 1. / sqrt(M * (z+1)**3 + (1-M-L) * (z+1)**2 + L) / (z+1)**2

    factor = quad(f, zi, zf)[0]

    if filled_beam:

        if k == 0:
            delksi = factor
            dist = delksi / (zf+1)
        else:
            delksi = sqrt(abs(M+L-1)) * factor
            if k == 1:
                dist = sin (delksi)/(zf+1)/sqrt(abs(M+L-1))
            else:
                dist = sinh(delksi)/(zf+1)/sqrt(abs(M+L-1))

    else:
        dist = (zi+1) * factor

    return dist

def scales(zl, zs):
    """Returns the following scaling factors as a dictionary:
        time:  time scale              (g days/arcsec^2)
        angdist: angular distance        (g kpc/arcs)
        critden: critical density        (g M_sol/arcsec^2)

        where g = 978.
    """
    gfac = 365.25e9 / (206265**2) # H0^-1 in g days/arcsec^2
    cee   = 8.393000e-7           # speed of light in kpc/day
    csfpg = 1.665000e15           # c^2/4piG in M_sol/kpc

    Dl = angdist(0, zl)   # radians
    return {'time':    (1+zl)*Dl*gfac, 
            'timebg':  (1+zl)*Dl, 
            'angdist':    cee*Dl*gfac*206265, # 206265 arcsec/rad
            'critden':    cee*Dl*gfac*csfpg,
            'g': 978}

#   return {'time':    (1+zl)*Dl*gfac, 
#           'timebg':  (1+zl)*Dl, 
#           'angdist':    cee*Dl*gfac*206265, # 206265 arcsec/rad
#           'critden':    cee*Dl*gfac*csfpg,
#           'g': 978}

    #return zl, (1+zl)*gfac*Dr, (1+zl)*Dr, cee*gfac*Dl*206265, cee*gfac*Dr*csfpg


