'''
   #------------------------------------------------------------------------
   # Sersic.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Sersic mass model:
    - pars[0] = Sigma0
    - pars[1] = Re
    - pars[2] = n
    - pars[3] = intpnts (for deprojection)'''

from numpy import linspace, empty
from numpy import pi
from numpy import cos, exp
from scipy.integrate.quadrature import simps, trapz

integrator = simps

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def surf(r,pars):
    '''Calculates surface density'''
    sigma0, Re, ns = pars[:3]
    bn = 1.9992*ns-0.3271
    return sigma0*exp(-bn*((r/Re)**(1.0/ns)-1.0))

def den(r,pars):
    '''Calculates density (solves the Abel integral)'''
    sigma0, Re, ns, intpnts = pars
    bn = 1.9992*ns-0.3271
    theta = linspace(0.,pi/2.-1.0e-6,num=intpnts)
    cth = cos(theta)
    rho = empty(len(r), 'double')
    for i in xrange(len(r)):
        y = exp(-bn*((r[i]/cth/Re)**(1./ns)-1.0))*\
            (-bn/Re**(1./ns)*(r[i]/cth)**(1./ns-1.0)*1./ns)
        rho[i] = (-1./pi)*integrator(y/cth,theta)
    return rho

#-----------------------------------------------------------------------------
# Test the functions. This runs if the python module is directly called.
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import *
    
    rmin = 0.01
    rmax = 10.
    pnts = 1000
    r = linspace(rmin,rmax,num=pnts)
    pars = [1.,1.,1.,1000]
    surface = surf(r,pars)
    density = den(r,pars)

    loglog(r,density,label='density')
    plot(r,surface,label='surface density')
    legend()
    show()
