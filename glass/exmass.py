
from numpy import array, log, abs, pi, cos, sin

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
        dr = r-self.r
        return self.alpha * dr.real * abs(dr) ** (self.alpha-2) / pi

    def poten_dy(self, r):
        dr = r-self.r
        return self.alpha * dr.imag * abs(dr) ** (self.alpha-2) / pi

    #def kappa(self, r):
        #return self.alpha**2 * abs(r-self.r)**(self.alpha-2) / pi

    def poten_dxdx(self, r):
        a = self.alpha
        dr = r-self.r
        return a * abs(dr)**(a-4) * ((a-1)*dr.real**2 + dr.imag**2) / pi

    def poten_dydy(self, r):
        a = self.alpha
        dr = r-self.r
        return a * abs(dr)**(a-4) * ((a-1)*dr.imag**2 + dr.real**2) / pi

    def poten_dxdy(self, r):
        a = self.alpha
        dr = r-self.r
        return (a-2)*a*dr.real*dr.imag * abs(dr)**(a-4) / pi

    def maginv(self, r, theta):
        xx    = self.poten_dxdx(r)
        yy    = self.poten_dydy(r)
        delta = self.poten_dxdy(r)

        theta *= pi/180
        cs = cos(2*theta)
        sn = sin(2*theta)

        kappa = (xx+yy)/2
        gamma = (xx-yy)/2
        return array([    0 - sn*gamma + cs*delta,
                      kappa + cs*gamma + sn*delta,
                      kappa - cs*gamma - sn*delta])

class Isothermal(PowerLawMass):

    def __init__(self, rx, ry, name='isothermal'):
        PowerLawMass.__init__(self,rx,ry,1,name)


