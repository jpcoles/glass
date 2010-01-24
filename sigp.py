from __future__ import division
from numpy import logspace, sin, log10, amin, amax
from scipy.integrate.quadrature import simps as integrator
from spherical_deproject import cumsolve, sigpsolve, sigpsingle, dlnrhodlnr, masstot, abelsolve
from scales import density_to_physical, distance_to_physical

def sigp(objmodel, light, lpars, aperture, beta, alphalim=3, interpnts=None):

    obj, data = objmodel

    #-------------------------------------------------------------------------
    # Find the inner and outer images for the object.
    #-------------------------------------------------------------------------
    rs = [ abs(img.pos) for src in obj.sources for img in src.images ]
    imagemin, imagemax = distance_to_physical(objmodel, amin(rs)), \
                         distance_to_physical(objmodel, amax(rs))

    mass2d  = data['encmass']
    R       = data['R_kpc']
    sigma   = data['sigma']

    intpnts = lpars[-1]

    aperture_phys = distance_to_physical(objmodel, aperture)
    lpars_phys    = lpars[:]
    lpars_phys[1] = distance_to_physical(objmodel, lpars_phys[1])

    #print aperture_phys, lpars_phys

    #-------------------------------------------------------------------------
    # Calculate the Abel integral to obtain rho(r)
    #-------------------------------------------------------------------------
    if not interpnts: interpnts = len(R)*2
    r = logspace(log10(amin(R) / 10),
                 log10(amax(R) * 10), num=interpnts)

    #print 'min(R)=%f, max(R)=%f' % (amin(R), amax(R))
    #print r

    #print 'abelsolve'
    #print '!!!!', imagemin, imagemax
    rho, mass3d = cumsolve(r, imagemin, imagemax, integrator, intpnts, alphalim, 
                           R, sigma, mass2d)

#   rhoa, mass3da = abelsolve(r, imagemin, imagemax,
#                             integrator, intpnts, alphalim,
#                             R, sigma, mass)

    drho = dlnrhodlnr(imagemin,imagemax,alphalim,r,rho,mass3d,
                      R,sigma,mass2d,r)

    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp      = 6.67e-11 * 1.989e30 / 3.086e19
    light.set_pars(lpars_phys)

    sigp = sigpsolve(r,  rho,mass3d, 
                     R,sigma,mass2d,
                     integrator,intpnts,alphalim,Gsp,
                     light,beta)/1000
    sigpsing = sigpsingle(r,sigp,light,aperture_phys,integrator)

    #rhint   = rhoint(imagemin,imagemax,alphalim,r,rho,mass3d,r)

#   sigpa     = sigpsolve(r,rhoa,mass3da,integrator,intpnts,alphalim,Gsp,light,lpars_phys,beta) / 1000
#   sigpsinga = sigpsingle(r,sigpa,light,lpars_phys,aperture_phys,integrator)
#   rhinta   = rhoint(imagemin,imagemax,alphalim,r,rhoa,mass3da,r)
#   drhoa    = dlnrhodlnr(r,rhinta)

    data['sigp:r'        ] = r
    data['sigp:rho'      ] = rho
    #data['sigp:rhoint'   ] = rhint
    data['sigp:drho'     ] = drho
    data['sigp:mass3d'   ] = mass3d
    data['sigp:sigp'     ] = sigp
    data['sigp:sigp_sing'] = sigpsing
    data['sigp:scale-factor'] = lpars_phys[1]

#   data['sigp:rhoa'      ] = rhoa
#   data['sigp:rhointa'   ] = rhinta
#   data['sigp:drhoa'     ] = drhoa
#   data['sigp:mass3da'   ] = mass3da
#   data['sigp:sigpa'     ] = sigpa
#   data['sigp:sigp_singa'] = sigpsinga

    #print data['R_phys']
    #print data['sigma_phys']
    #print data['encmass_phys']

    print 'Final rms mean projected vel. dispersion:', sigpsing

def sigpf(objmodel, vdisp, tol, chisq_cut):
    """Return True if chi-squared value for the object's sigp is <= chisq_cut."""
    obj,data = objmodel
    chisq = (vdisp - data['sigp:sigp_sing'])**2 / tol**2
    print 'chi2 is', chisq
    data['sigp:chisq'] = chisq
    return chisq <= chisq_cut

