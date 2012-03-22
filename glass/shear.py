from __future__ import division
from numpy import vectorize
from math import pi, cos, sin

class Shear:
    def __init__(self, phi):
        self.phi = pi/90 * (phi + 67.5)
        self.cs2 = cos(self.phi)
        self.sn2 = sin(self.phi)

        print "Shear  ", self.cs2, self.sn2, self.phi

    def poten(self, n0, r):
        derivs = n0
        if not isinstance(n0, list): derivs = [n0]
        l = []
        for n in derivs:
            assert (n==1 or n==2)
            x,y = r.real, r.imag
            if n == 1: l.append( self.cs2 * (x**2 - y**2)/2 + self.sn2*x*y)
            else:      l.append(-self.sn2 * (x**2 - y**2)/2 + self.cs2*x*y)

        if not isinstance(n0, list): 
            return l[0]
        return l

    def poten_dx(self, r):  return  self.cs2*r.real + self.sn2*r.imag
    def poten_d2x(self, r): return -self.sn2*r.real + self.cs2*r.imag

    def poten_dy(self, r):  return -self.cs2*r.imag + self.sn2*r.real
    def poten_d2y(self, r): return  self.sn2*r.imag + self.cs2*r.real

    def poten_x(self, n, r):
        assert (n==1 or n==2)
        x,y = r.real, r.imag
        if n == 1:
            return  self.cs2*x + self.sn2*y
        else:
            return -self.sn2*x + self.cs2*y

    def poten_y(self, n, r):
        assert (n==1 or n==2)
        x,y = r.real, r.imag
        if n == 1:
            return -self.cs2*y + self.sn2*x
        else:
            return  self.sn2*y + self.cs2*x
        
    def maginv(self, n, r, theta):
        assert (n==1 or n==2)
        x,y = r.real, r.imag
        if n == 1:
            xx, yy, xy = self.cs2, -self.cs2, self.sn2
        else:
            xx, yy, xy = -self.sn2, self.sn2, self.cs2

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

