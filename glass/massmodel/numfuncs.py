'''
   #------------------------------------------------------------------------
   # numfuncs.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Contains global functions for use by mass models. These include:
     - numsurf (numerically calculate the surface density given an input
       density)
     - numden (num calc. the density given an input surface density)
'''

from __future__ import division
from numpy import linspace, empty
from numpy import pi
from numpy import cos
from scipy.integrate.quadrature import simps, trapz

_integrator = simps

def numsurf(r,mm,intpnts):
    '''Numerically calculates the surface density'''
    intpnts = mm.pars[-1]
    theta = linspace(0,pi/2-1e-6,num=intpnts)
    cth = cos(theta)
    cth2 = cth**2
    surf = empty(len(r), 'double')
    for i in xrange(len(r)):
        q = r[i]/cth
        y = mm.den(q)
        surf[i] = 2*r[i]*_integrator(y/cth2,theta)
    return surf

def numden(r,mm,intpnts):
    '''Numerically calculates the density (solves the Abel integral)'''
    intpnts = mm.pars[-1]
    theta = linspace(0,pi/2-1e-6,num=intpnts)
    cth = cos(theta)
    rho = empty(len(r), 'double')
    for i in xrange(len(r)):
        y = mm.dsurf(r[i]/cth)
        rho[i] = (-1/pi)*_integrator(y/cth,theta)
    return rho

