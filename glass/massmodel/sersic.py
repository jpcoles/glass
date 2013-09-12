'''
   #------------------------------------------------------------------------
   # Sersic.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Sersic mass model:
    - pars[0] = Sigma0
    - pars[1] = Re
    - pars[2] = n
    - pars[3] = intpnts (for deprojection)
'''

from __future__ import division
from numpy import exp
from numfuncs import numden

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

class Sersic:
    def __init__(self, pars):
        self.pars = pars

    def surf(self, r):
        '''Calculates surface density'''
        sigma0, Re, ns = self.pars[:3]
        bn = 1.9992*ns-0.3271
        return sigma0*exp(-bn*((r/Re)**(1/ns)-1))

    def dsurf(self, r):
        '''Calculates radial derivative of surface density'''
        sigma0, Re, ns = self.pars[:3]
        bn = 1.9992*ns-0.3271
        return sigma0*exp(-bn*((r/Re)**(1/ns)-1))*\
               (-bn/Re**(1/ns)*r**(1/ns-1)*1/ns)

    def den(self, r):
        '''Calculates density'''
        return numden(r,self)

#-----------------------------------------------------------------------------
# Test the functions. This runs if the python module is directly called.
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import loglog, plot, legend, show
    from numpy import linspace
    
    rmin = 0.01
    rmax = 5
    pnts = 1000
    r = linspace(rmin,rmax,num=pnts)
    pars = [1,1,1,1000]
    surface = surf(r,pars)
    density = den(r,pars)

    loglog(r,density,label='density')
    plot(r,surface,label='surface density')
    legend()
    show()
