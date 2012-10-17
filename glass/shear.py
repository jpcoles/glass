from __future__ import division
from numpy import array, vectorize
from math import pi, cos, sin

shift = 10 # arcsec

def poten(n, r):
    x,y = r.real, r.imag
    if n==0: return (x**2 - y**2)/2
    if n==1: return x*y
    assert 0

def poten_dx(n, r):
    x,y = r.real, r.imag
    if n == 0: return x
    if n == 1: return y
    assert 0

def poten_dy(n, r):
    x,y = r.real, r.imag
    if n == 0: return -y
    if n == 1: return x
    assert 0

def poten_dxdx(n, r):
    x,y = r.real, r.imag
    if n == 0: return 1
    if n == 1: return 0
    assert 0

def poten_dydy(n, r):
    x,y = r.real, r.imag
    if n == 0: return -1
    if n == 1: return 0
    assert 0

def poten_dxdy(n, r):
    x,y = r.real, r.imag
    if n == 0: return 0
    if n == 1: return 1
    assert 0

def maginv(n, r, theta):
    #print 'maginv', r, theta, a
    xx    = poten_dxdx(n,r)
    yy    = poten_dydy(n,r)
    delta = poten_dxdy(n,r)

    theta *= pi/180
    cs = cos(2*theta)
    sn = sin(2*theta)

    kappa = (xx+yy)/2
    gamma = (xx-yy)/2
    return array([    0 - sn*gamma + cs*delta,
                  kappa + cs*gamma + sn*delta,
                  kappa - cs*gamma - sn*delta])

def Xmaginv(n, r, theta):
    assert (n==1 or n==2)
    x,y = r.real, r.imag
    if n == 1:
        xx, yy, xy = self.cs, -self.cs, self.sn
    else:
        xx, yy, xy = -self.sn, self.sn, self.cs

    kappa = (xx+yy)/2
    gamma = (xx-yy)/2
    delta = xy
    theta *= pi/90
    cs = cos(theta)
    sn = sin(theta)
    #print "maginv ", kappa, gamma, delta, theta, cs, sn
    return [    0 - sn*gamma + cs*delta,
            kappa + cs*gamma + sn*delta,
            kappa - cs*gamma - sn*delta]

