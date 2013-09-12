'''
   #------------------------------------------------------------------------
   # datafile.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Contains global classes and functions for handling tab. ASCII data:
     - DataFileMassModel is used to load in ASCII tabulated data for
       the mass model which is then interpolated/integrated over. 
'''

from __future__ import division
from numpy import loadtxt
from numpy import interp
from numfuncs import numsurf, numden
from scipy.misc.common import derivative

class _DataFileMassModel:

    def __init__(self, denfile, surffile):
        '''Reads in the density/surface density data. Note
           that ***NO UNIT CONVERSION*** is done. Be sure
           that the units is these ASCII files match your
           expectations...'''
        if denfile is not None:
            # Load in the density data file:
            fden = loadtxt(denfile,
                           dtype = {'names': ('r', 'den', 'err'),
                                    'formats': ('f8', 'f8', 'f8')})
            self.rin = fden['r']
            self.denin = fden['den']
        else:
            self.rin = None
            self.denin = None
        if surffile is not None:
           # Load in the surface density data file:
            fsurf = loadtxt(surffile,
                           dtype = {'names': ('R', 'sigma', 'err'),
                                    'formats': ('f8', 'f8', 'f8')})
            self.rsin = fsurf['R']
            self.sdenin = fsurf['sigma']
        else:
            self.rsin = None
            self.sdenin = None

    def surf(self, r, pars):
        '''Calculates surface density'''
        if self.rsin is None:
            # Need to projcet den to get surf:
            return numsurf(r,pars,self.den)
        else:
            # Need to interpolate input array:
            return interp(r,self.rsin,self.sdenin,right=0)

    def dsurf(self, r, pars):
        '''Calculates radial derivative of surface density'''
        return derivative(lambda x: interp(x,self.rsin,self.sdenin,right=0), r)

    def den(self,r,pars):
        '''Calculates density'''
        if self.rin is None:
            # Need to deprojcet surf to get den:
            return numden(r,pars,self.dsurf)
        else:
            # Need to interpolate input array:
            return interp(r,self.rin,self.denin,right=0)

def fromfile(denfile=None, surffile=None):

    return _DataFileMassModel(denfile, surffile)

#-----------------------------------------------------------------------------
# Test the functions. This runs if the python module is directly called.
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import loglog, plot, legend, show
    from numpy import linspace
    
    rmin = 0.01
    rmax = 10
    pnts = 1000
    r = linspace(rmin,rmax,num=pnts)
    pars = [99]

    denfile = '../Data/triaxgal_denstar.txt'
    surffile = '../Data/triaxgal_sdenzstar.txt'

    light = fromfile(denfile=denfile, surffile=surffile)
    light2 = fromfile(denfile=denfile)
    light3 = fromfile(surffile=surffile)

    surface = light.surf(r,pars)
    density = light.den(r,pars)
    surface2 = light2.surf(r,pars)
    density2 = light2.den(r,pars)
    surface3 = light3.surf(r,pars)
    density3 = light3.den(r,pars)

    loglog(r,density,label='density')
    plot(r,surface,label='surface density')
    plot(r,density2,label='density 2')
    plot(r,surface2,label='surface density 2')
    plot(r,density3,label='density 3')
    plot(r,surface3,label='surface density 3')
    legend()
    show()

