from __future__ import division
from numpy import logspace, sin, log10
from scipy.integrate.quadrature import simps
from spherical_deproject import abelsolve, sigpsolve, sigpsingle
from scales import density_to_physical, distance_to_physical

def sigp(vdisp, tol, light, lpars, aperture, beta, chisq_cut=5, alphalim=3, interpnts=100):

    print 'Here in sigp'
    def sigp_filter(model):
        for objmodel in model['objs']:
            sp = _sigp(objmodel, light, lpars, aperture, beta, alphalim, interpnts)
            #if not vdisp-tol <= sp <= vdisp+tol: return False

            if ((vdisp**2 - sp**2) / tol**2) > chisq_cut: return False
        return True
    return sigp_filter

def _sigp(objmodel, light, lpars, aperture, beta, alphalim, interpnts):

    obj, data = objmodel

    #-------------------------------------------------------------------------
    # Find the inner and outer images for the object.
    #-------------------------------------------------------------------------
    rs = [ abs(img.pos) for sys in obj.systems for img in sys.images ]
    imagemin, imagemax = distance_to_physical(objmodel, min(rs)), \
                         distance_to_physical(objmodel, max(rs))

    mass  = data['encmass_phys']
    R     = data['R_phys']
    sigma = data['sigma_phys']

    intpnts = lpars[-1]
    integrator = simps

    #-------------------------------------------------------------------------
    # Calculate the Abel integral to obtain rho(r)
    #-------------------------------------------------------------------------
    r = logspace(log10(min(R) / 10),
                 log10(max(R) * 10), num=interpnts)

    print 'abelsolve'
    rho = abelsolve(r, imagemin, imagemax,
                    integrator, intpnts, alphalim,
                    R, sigma, mass)
 
    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp      = 6.67e-11 * 1.989e30 / 3.086e19
    print 'sigpsolve'
    sigp     = sigpsolve(r,rho,integrator,intpnts,alphalim,Gsp,light,lpars,beta) / 1000
    print 'sigpsingle'
    sigpsing = sigpsingle(r,sigp,light,lpars,aperture,integrator)

    data['sigp:R'] = r
    data['sigp:rho']  = rho
    data['sigp:sigp'] = sigp
    data['sigp:sigp_sing'] = sigpsing

    print 'Final rms mean projected vel. dispersion:',sigpsing
    return sigpsing

