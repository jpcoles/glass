from __future__ import division
import numpy as np
from numpy import logspace, sin, log10, amin, amax
from scipy.integrate.quadrature import simps as integrator
from spherical_deproject import cumsolve, sigpsolve, sigpsingle, dlnrhodlnr, masstot, abelsolve
from glass.scales import convert
from glass.log import log as Log
from glass.environment import DArray

def rho3d(objmodel, alphalim=3.5, interpnts=None, intpnts=None, rspan=None):
    obj, data = objmodel

    arcsec2kpc = convert('arcsec to kpc', 1, obj.dL, data['nu'])

    #-------------------------------------------------------------------------
    # Find the inner and outer images for the object.
    #-------------------------------------------------------------------------
    if rspan is None:
        rs = [ abs(img.pos) for src in obj.sources for img in src.images ]
        imagemin, imagemax = amin(rs)*arcsec2kpc, amax(rs)*arcsec2kpc
    else:
        imagemin, imagemax = np.array(rspan) * arcsec2kpc

    mass2d  = data['M(<R)']
    R       = data['R']['kpc']
    sigma   = data['Sigma(R)']


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

    rscale = convert('kpc to arcsec', 1, obj.dL, data['nu'])
    r =  DArray(r, r'$r$', {'arcsec': [rscale, r'$\mathrm{arcsec}$'],
                            'kpc':    [1, r'$\mathrm{kpc}$']})
    rho  = DArray(rho, r'$\rho(r)$', {'Msun/kpc^3': [1, r'$M_\odot/\mathrm{kpc}^3$']})
    drho = DArray(drho, r'$d\ln \rho / d\ln r$', {'Msun/kpc': [1, r'$M_\odot/\mathrm{kpc}$']})

    data['rho3d:r'   ] = r
    data['rho3d:rho' ] = rho
    data['rho3d:drho'] = drho
    data['rho3d:mass'] = mass3d

def sigp(objmodel, lightC, lpars, aperture, beta, alphalim=3.5, interpnts=None, intpnts=None, rspan=None):

    obj, data = objmodel

    arcsec2kpc = convert('arcsec to kpc', 1, obj.dL, data['nu'])

    rho3d(objmodel, alphalim=alphalim, interpnts=interpnts, intpnts=intpnts,rspan=rspan)
    r       = data['rho3d:r'   ]
    rho     = data['rho3d:rho' ]
    drho    = data['rho3d:drho']
    mass3d  = data['rho3d:mass']
    mass2d  = data['M(<R)']
    R       = data['R']['kpc']
    sigma   = data['Sigma(R)']

    aperture_phys = aperture * arcsec2kpc
    lpars_phys    = lpars[:]
    #lpars_phys[1] = lpars_phys[1] * arcsec2kpc

    #print aperture_phys, lpars_phys


    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp      = 6.67e-11 * 1.989e30 / 3.086e19
    #light.set_pars(lpars_phys)
    light = lightC(lpars_phys, intpnts)

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

    #data['sigp:rhoint'   ] = rhint
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

    Log( 'Final rms mean projected vel. dispersion: %f' % sigpsing )

def sigpf(objmodel, vdisp, tol, chisq_cut):
    """Return True if chi-squared value for the object's sigp is <= chisq_cut."""
    obj,data = objmodel
    chisq = (vdisp - data['sigp:sigp_sing'])**2 / tol**2
    Log( 'chi2 is %f' % chisq )
    data['sigp:chisq'] = chisq
    return chisq <= chisq_cut

