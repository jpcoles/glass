'''
   #------------------------------------------------------------------------
   # Galaxy.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   General galaxy mass model. Parameters for M31 are given in Geehan et 
   al. 2006. For the form used here see Penarrubia et al. 2006.  
    - pars[0] = M
    - pars[1] = a
    - pars[2] = b
    - pars[3] = Mb
    - pars[4] = sc
    - pars[5] = Mh
    - pars[6] = rs
    - pars[7] = c200
    - pars[8] = G
    - pars[9] = intpnts (for projection)
'''

from __future__ import division
from numpy import pi, abs, sqrt, log
from numfuncs import numsurf

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def potxyz(x,y,z,pars):    
    '''Calculates gravitational potential [with G!!]'''
    Md, a, b, Mb, sc, Mh, rs, c200, G = pars[:9]

    r = sqrt(x**2 + y**2 + z**2)
    Phi_mn = -G*Md/sqrt(x**2+y**2+(a+sqrt(z**2+b**2))**2)
    Phi_b = -G*Mb/(r+sc)
    Phi_nfw = -G*Mh*log(1+r/rs)/r/(log(1+c200)-c200/(1+c200))

    return Phi_nfw + Phi_mn + Phi_b

def fxyz(x,y,z,pars):    
    '''Calculate forces [with G!!]'''
    Md, a, b, Mb, sc, Mh, rs, c200, G = pars[:9]

    K = -G*Mh/(log(1+c200)-c200/(1+c200))
    r = sqrt(x**2 + y**2 + z**2)
    dPhidr = K*(-log(1+r/rs)/r**2 + 1/(1+r/rs)/r/rs)

    Fx_nfw = -dPhidr * x/r
    Fy_nfw = -dPhidr * y/r
    Fz_nfw = -dPhidr * z/r 

    Fx_mn = -G*M*x/((x**2+y**2+(a+sqrt(z**2+b**2))**2)**(3/2))
    Fy_mn = -G*M*y/((x**2+y**2+(a+sqrt(z**2+b**2))**2)**(3/2))
    Fz_mn = -G*M*(a+sqrt(z**2+b**2))*z/\
            ((x**2+y**2+(a+sqrt(z**2+b**2))**2)**(3/2)*sqrt(z**2+b**2))

    Fx_b = -G*Mb*x/(sqrt(x**2+y**2+z**2)+sc)**2/sqrt(x**2+y**2+z**2)
    Fy_b = -G*Mb*y/(sqrt(x**2+y**2+z**2)+sc)**2/sqrt(x**2+y**2+z**2)
    Fz_b = -G*Mb*z/(sqrt(x**2+y**2+z**2)+sc)**2/sqrt(x**2+y**2+z**2)


    return [Fx_nfw + Fx_mn + Fx_b, Fy_nfw + Fy_mn + Fy_b, Fz_nfw + Fz_mn + Fz_b]

def pot(r,pars):    
    '''Calculates gravitational potential [with G!!]'''
    Md, a, b, Mb, sc, Mh, rs, c200, G = pars[:9]

    Phi_mn = -G*Md/sqrt(r**2+(a+b)**2)
    Phi_b = -G*Mb/(r+sc)
    Phi_nfw = -G*Mh*log(1+r/rs)/r/(log(1+c200)-c200/(1+c200))

    return Phi_nfw + Phi_mn + Phi_b

def fr(r,pars):    
    '''Calculate forces [with G!!]'''
    Md, a, b, Mb, sc, Mh, rs, c200, G = pars[:9]

    K = -G*Mh/(log(1+c200)-c200/(1+c200))
    dPhidr = K*(-log(1+r/rs)/r**2 + 1/(1+r/rs)/r/rs)

    Fr_nfw = -dPhidr
    Fr_mn = -G*Md*r/((r**2+(a+b)**2)**(3/2))
    Fr_b = -G*Mb/(sqrt(r**2)+sc)**2

    return Fr_nfw + Fr_mn + Fr_b

def cummass(r,pars):    
    '''Calculate forces [with G!!]'''
    Md, a, b, Mb, sc, Mh, rs, c200, G = pars[:9]

    K = -Mh/(log(1+c200)-c200/(1+c200))
    dPhidr = K*(-log(1+r/rs)/r**2 + 1/(1+r/rs)/r/rs)

    Mr_nfw = dPhidr*r**2
    Mr_mn = Md*r**3/((r**2+(a+b)**2)**(3/2))
    Mr_b = Mb*r**2/(sqrt(r**2)+sc)**2

    return Mr_nfw + Mr_mn + Mr_b

#-----------------------------------------------------------------------------
# Test the functions. This runs if the python module is directly called.
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import loglog, plot, legend, show, figure
    from numpy import linspace, zeros
    
    #Units: 
    Msun = 1.989e30 
    kpc = 3.086e19
    kms = 1e3
    Gsi = 6.67e-11

    #Set up x,y,z coordinates:
    rmin = 0.01*kpc
    rmax = 30*kpc
    pnts = 1000
    r = linspace(rmin,rmax,num=pnts)
      
    #Values here for the Geehan M31 model: 
    pars = [8.4e10*Msun,5.4*kpc,0.26*kpc,3.3e10*Msun,0.61*kpc,6.8e11*Msun,
            8.18*kpc,22,Gsi,1000]

    #Test the functions:
    pot = pot(r,pars)
    fr = fr(r,pars)
    fr2 = -Gsi*cummass(r,pars)/r**2
    vc = sqrt(r*abs(fr))

    figure()
    plot(r/kpc,pot/kms**2,label='potential')
    legend() 

    figure()
    plot(r/kpc,fr,label='force_r')
    plot(r,fr2,label='G*M(r)/r**2')
    legend()

    figure()
    plot(r/kpc,vc/kms,label='v_c')
    legend()
 
    show()
    
