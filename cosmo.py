from __future__ import division
from math import sin,sinh, sqrt
from numpy import abs
from environment import env

def angdist(z1, z2):
    M = env().omega_matter
    L = env().omega_lambda

    #---------------------------------------------------------------------------
    # Curvature of the universe.
    #---------------------------------------------------------------------------
    tol = 1e-4
    k = 0
    if M+L+tol < 1:
        k = -1
    elif M+L-tol > 1:
        k = 1

    dz = 5e-4
    if    z1 < z2: zi,zf = z1,z2
    else:          zi,zf = z2,z1

    if env().filled_beam:
        z = zi + dz/2.
        factor = 0
        while z <= zf:
            w = z + 1
            factor += dz / sqrt(M * w**3 + (1-M-L) * w**2 + L)
            z += dz

        if k == 0:
            delksi = factor
            dis = delksi / (z2+1)
        else:
            delksi = sqrt(abs(M+L-1) * factor)
            if k == 1:
                dis = sin(delksi)/(z2+1)/sqrt(abs(M+L-1))
            else:
                dis = sinh(delksi)/(z2+1)/sqrt(abs(M+L-1))

    else:
        z = zi + dz/2
        factor = 0
        while z <= zf:
            w = z + 1
            factor += dz / sqrt(M * w**3 + (1-M-L) * w**2 + L) / w**2
            z += dz
        dis = (zi+1) * factor

    return dis

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


