'''
   #------------------------------------------------------------------------
   # Hernquist.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Hernquist mass model:
    - pars[0] = M
    - pars[1] = a
    - pars[2] = alp
    - pars[3] = intpnts (for projection)'''

from numpy import linspace, empty
from numpy import pi
from numpy import cos
from scipy.integrate.quadrature import simps, trapz

integrator = simps

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def surf(r,pars):
    '''Calculates surface density'''
    M, a, alp, intpnts = pars
    theta = linspace(0.,pi/2.-1.0e-6,num=intpnts)
    cth = cos(theta)
    cth2 = cth**2.0
    surf = empty(len(r), 'double')
    for i in xrange(len(r)):
        q = r[i]/cth
        y = M*(3.0-alp)/(4.0*pi*a**3.0)/\
            ((q/a)**alp*(1.0+(q/a))**(4.0-alp))
        surf[i] = 2.0*r[i]*integrator(y/cth2,theta)
    return surf

def den(r,pars):    
    '''Calculates density'''
    M, a, alp = pars[:3]
    return M*(3.0-alp)/(4.0*pi*a**3.0)*\
           (r/a)**(-alp)*(1.0+r/a)**(-4.0+alp)

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
