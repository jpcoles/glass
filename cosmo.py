from __future__ import division
from math import sinh, sqrt
from numpy import abs
from environment import env

def angdist(z1, z2):
    M = env.omega_matter
    L = env.omega_lambda
    k = 0
    if M+L < 0: k = -1
    if M+L > 0: k =  1
    if abs(M+L-1 < 1e-4): k = 0
    dz = 5e-4
    if    z1 < z2: zi,zf = z1,z2
    else:          zi,zf = z2,z1
    if env.filled_beam:
        z = zi + dz/2.
        factor = 0
        while z <= zf:
            w = z + 1
            factor += dz / sqrt(M * w**3 + (1-M-L) * w**2 + L)
            z += dz

        if k != 0:
            delksi = sqrt(abs(M+L-1) * factor)
            if k == 1:
                dis = sin(delksi)/(zs+1)/sqrt(abs(M+L-1))
            else:
                dis = sinh(delksi)/(zs+1)/sqrt(abs(M+L-1))
        else:
            delksi = factor
            dis = delksi / (z2+1)

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
    gfac=8.584977
    cee=8.393e-7
    csfpg=1.665e15

    Dl = Dr = angdist(0, zl)
    if zs != 0: Dr *= angdist(0,zs) / angdist(zl,zs)
    return zl, (1+zl)*gfac*Dr, (1+zl)*Dr, cee*gfac*Dl*206265, cee*gfac*Dr*csfpg


