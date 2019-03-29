'''
   #------------------------------------------------------------------------
   # Hernquist.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Hernquist mass model:
    - pars[0] = M (total mass in Msun)
    - pars[1] = a (scale length in kpc)
    - pars[2] = alp (central slope)
    - pars[3] = G
    - pars[4] = intpnts (for projection)
'''


from numpy import pi
from .numfuncs import numsurf

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

class Hernquist:
    
    def __init__(self, pars, intpnts):
        self.pars = pars
        self.intpnts = intpnts

    def den(self, r):    
        '''Calculates density'''
        M, a, alp, G = self.pars[:4]
        return M*(3-alp)/(4*pi*a**3)*\
               (r/a)**(-alp)*(1+r/a)**(-4+alp)

    def surf(self, r):
        '''Calculates surface density'''
        return numsurf(r,self,self.intpnts)

    def cummass(self, r):    
        '''Calculates cumulative mass'''
        M, a, alp, G = self.pars[:4]
        return M*(r/a)**(-alp)*((r+a)/a)**alp*r**3/(a+r)**3

    def pot(self, r):    
        '''Calculates gravitational potential'''
        M, a, alp, G = self.pars[:4]
        return G*(M/(a*(2-alp)))*((1+(r/a)**(-1))**(alp-2)-1)

    def fr(self, r):    
        '''Calculates radial force'''
        M, a, alp, G = self.pars[:4]
        return  -G*M*(1+1/r*a)**(-3+alp)/r**2


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
    pars = [1,1,1,G,1000]
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
    
