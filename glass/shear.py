
from numpy import array, vectorize
from math import pi, cos, sin

class Shear:

    def __init__(self, shift=10, name='shear'):
        self.nParams = 2
        #self.shift = 10 # arcsec
        self.name = name
        self.shift = shift

    def poten(self, r):
        x,y = r.real, r.imag
        n0 = (x**2 - y**2)/2
        n1 = x*y
        return array([n0,n1])

    def poten_dx(self, r):
        return array([r.real, r.imag])

    def poten_dy(self, r):
        return array([-r.imag, r.real])

    def poten_dxdx(self, r):
        return array([1, 0])

    def poten_dydy(self, r):
        return array([-1, 0])

    def poten_dxdy(self, r):
        return array([0, 1])

    def maginv(self, r, theta):
        #print 'maginv', r, theta, a
        xx    = self.poten_dxdx(r)
        yy    = self.poten_dydy(r)
        delta = self.poten_dxdy(r)

        theta *= pi/180
        cs = cos(2*theta)
        sn = sin(2*theta)

        kappa = (xx+yy)/2
        gamma = (xx-yy)/2


        k,g,d = kappa[0],gamma[0], delta[0]
        n0 = ([ 0 - sn*g + cs*d,
                k + cs*g + sn*d,
                k - cs*g - sn*d])

        k,g,d = kappa[1],gamma[1], delta[1]
        n1 = ([ 0 - sn*g + cs*d,
                k + cs*g + sn*d,
                k - cs*g - sn*d])
        return array([n0, n1])

