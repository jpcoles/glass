'''
   #------------------------------------------------------------------------
   # Plummer.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Plummer mass model:
    - pars[0] = M
    - pars[1] = a
    - pars[2] = G
    - pars[3] = intpnts (for projection)
'''

from __future__ import division
from numpy import pi, sqrt
from numfuncs import numsurf

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def den(r,pars):    
    '''Calculates density'''
    M, a, G = pars[:3]
    return (3*M/(4*pi*a**3))*(1+(r/a)**2)**(-2.5) 

def surf(r,pars):
    '''Calculates surface density'''
    return numsurf(r,pars,den)

def cummass(r,pars):    
    '''Calculates cumulative mass'''
    M, a, G = pars[:3]
    return M*r**3/(a**2+r**2)**(3/2)

def pot(r,pars):    
    '''Calculates potential'''
    M, a, G = pars[:3]
    return -G*M/sqrt(a**2+r**2) 

def fr(r,pars):    
    '''Calculates radial force'''
    M, a, G = pars[:3]
    return -G*M*r/(a**2+r**2)**(3/2)

#-----------------------------------------------------------------------------
# Test the functions. This runs if the python module is directly called.
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import loglog, plot, legend, show, figure
    from numpy import linspace
    
    rmin = 0.01
    rmax = 10
    pnts = 1000
    r = linspace(rmin,rmax,num=pnts)
    G = 1
    pars = [1,1,G,1000]
    surface = surf(r,pars)
    density = den(r,pars)
    cummass = cummass(r,pars)
    pot = pot(r,pars)
    fr = fr(r,pars)
    fr2 = -G*cummass/r**2

    loglog(r,density,label='density')
    plot(r,surface,label='surface density')
    legend()

    figure()
    loglog(r,cummass,label='cumulative mass')
    legend()

    figure()
    plot(r,pot,label='potential')
    legend() 

    figure()
    plot(r,fr,label='radial force')
    plot(r,fr2,label='G*M(r)/r**2')
    legend()
 
    show()
    
