from __future__ import division
from numpy import array, log, abs, pi

class ExternalMass:
    pass

class PointMass(ExternalMass):

    def __init__(self, rx, ry, rc=0, name='ptmass'):
        self.r = complex(rx, ry)
        self.rcore = rc
        self.nParams = 1
        self.shift = 0
        self.name = name

    def poten(self, r):
        return log(abs(r - self.r)) / pi

    def poten_dx(self, r):
        return (r.real-self.r.real) / abs(r-self.r)**2 / pi

    def poten_dy(self, r):
        return (r.imag-self.r.imag) / abs(r-self.r)**2 / pi


class PowerLawMass(ExternalMass):

    def __init__(self, rx, ry, alpha, name='plmass'):
        self.r = complex(rx, ry)
        self.alpha = alpha
        self.nParams = 1
        self.shift = 0
        self.name = name

    def poten(self, r):
        return abs(r-self.r)**self.alpha / pi

    def poten_dx(self, r):
        return self.alpha * (r.real-self.r.real) * abs(r-self.r) ** (self.alpha-2) / pi

    def poten_dy(self, r):
        return self.alpha * (r.imag-self.r.imag) * abs(r-self.r) ** (self.alpha-2) / pi

    def kappa(self, r):
        return self.alpha**2 * abs(r-self.r)**(self.alpha-2) / pi

