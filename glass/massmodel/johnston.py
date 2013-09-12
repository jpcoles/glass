'''
   #------------------------------------------------------------------------
   # Johnston.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   Johnston Milky Way mass model:
    - pars[0] = M
    - pars[1] = v0
    - pars[2] = rt
    - pars[3] = a
    - pars[4] = b
    - pars[5] = q0
    - pars[6] = Msphere
    - pars[7] = sc
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
    M, v0, rt, a, b, q0, Msphere, sc, G = pars[:9]
    if rt < 0:
        rc = -rt
    else:
        rc = rt*sqrt(v0/sqrt(v0**2-(rt**2*M*G/(sqrt(rt**2+(a+b)**2))**3))-1)
    r = sqrt(x**2 + y**2 + z**2)
    
    Phi_log = 1/2*v0**2*log(rc**2+x**2+y**2+(z**2)/q0**2)
    Phi_mn = -G*M/sqrt(x**2+y**2+(a+sqrt(z**2+b**2))**2)
    Phi_b = -G*Msphere/(sqrt(x**2+y**2+z**2)+sc)

    return Phi_log + Phi_mn + Phi_b

def fxyz(x,y,z,pars):    
    '''Calculate forces [with G!!]'''
    M, v0, rt, a, b, q0, Msphere, sc, G = pars[:9]
    if rt < 0:
        rc = -rt
    else:
        rc = rt*sqrt(v0/sqrt(v0**2-(rt**2*M*G/(sqrt(rt**2+(a+b)**2))**3))-1)
    r = sqrt(x**2 + y**2 + z**2)

    Fx_log = -(v0**2)*x/(rc**2+x**2+y**2+(z**2)/q0**2)
    Fy_log = -(v0**2)*y/(rc**2+x**2+y**2+(z**2)/q0**2)
    Fz_log = -(v0**2)*z/(rc**2+x**2+y**2+(z**2)/q0**2)/q0**2
    Fx_mn = -G*M*x/((x**2+y**2+(a+sqrt(z**2+b**2))**2)**(3/2))
    Fy_mn = -G*M*y/((x**2+y**2+(a+sqrt(z**2+b**2))**2)**(3/2))
    Fz_mn = -G*M*(a+sqrt(z**2+b**2))*z/\
            ((x**2+y**2+(a+sqrt(z**2+b**2))**2)**(3/2)*sqrt(z**2+b**2))
    Fx_b = -G*Msphere*x/(sqrt(x**2+y**2+z**2)+sc)**2/sqrt(x**2+y**2+z**2)
    Fy_b = -G*Msphere*y/(sqrt(x**2+y**2+z**2)+sc)**2/sqrt(x**2+y**2+z**2)
    Fz_b = -G*Msphere*z/(sqrt(x**2+y**2+z**2)+sc)**2/sqrt(x**2+y**2+z**2)


    return [Fx_log + Fx_mn + Fx_b, Fy_log + Fy_mn + Fy_b, Fz_log + Fz_mn + Fz_b]

def pot(r,pars):    
    '''Calculates gravitational potential [with G!!]'''
    M, v0, rt, a, b, q0, Msphere, sc, G = pars[:9]
    if rt < 0:
        rc = -rt
    else:
        rc = rt*sqrt(v0/sqrt(v0**2-(rt**2*M*G/(sqrt(rt**2+(a+b)**2))**3))-1)
    
    Phi_log = 1/2*v0**2*log(rc**2+r**2)
    Phi_mn = -G*M/sqrt(r**2+(a+b)**2)
    Phi_b = -G*Msphere/(sqrt(r**2)+sc)

    return Phi_log + Phi_mn + Phi_b

def fr(r,pars):    
    '''Calculate forces [with G!!]'''
    M, v0, rt, a, b, q0, Msphere, sc, G = pars[:9]
    if rt < 0:
        rc = -rt
    else:
        rc = rt*sqrt(v0/sqrt(v0**2-(rt**2*M*G/(sqrt(rt**2+(a+b)**2))**3))-1)

    Fr_log = -(v0**2)*r/(rc**2+r**2)
    Fr_mn = -G*M*r/((r**2+(a+b)**2)**(3/2))
    Fr_b = -G*Msphere*r/(sqrt(r**2)+sc)**2/sqrt(r**2)

    return Fr_log + Fr_mn + Fr_b


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
      
    #Values here for the Johnston q=1 Milky Way model:
    pars = [1e11*Msun,114*sqrt(2)*kms,-12*kpc,6.5*kpc,0.26*kpc,\
            1,3.4e10*Msun,0.7*kpc,Gsi,1000]

    #Test the functions:
    pot = pot(r,pars)
    fr = fr(r,pars)
    vc = sqrt(r*abs(fr))

    figure()
    plot(r/kpc,pot/kms**2,label='potential')
    legend() 

    figure()
    plot(r/kpc,fr,label='force_r')
    legend()

    figure()
    plot(r/kpc,vc/kms,label='v_c')
    legend()
 
    show()
    
