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

