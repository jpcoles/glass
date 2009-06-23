from __future__ import division
from numpy import logspace, sin, log10, amin, amax
from scipy.integrate.quadrature import simps
from spherical_deproject import cumsolve, sigpsolve, sigpsingle, rhoint, dlnrhodlnr
from scales import density_to_physical, distance_to_physical

def sigp(objmodel, light, lpars, aperture, beta, alphalim=3, interpnts=100):

    obj, data = objmodel

    #-------------------------------------------------------------------------
    # Find the inner and outer images for the object.
    #-------------------------------------------------------------------------
    rs = [ abs(img.pos) for sys in obj.systems for img in sys.images ]
    imagemin, imagemax = distance_to_physical(objmodel, amin(rs)), \
                         distance_to_physical(objmodel, amax(rs))

    mass  = data['encmass_phys']
    R     = data['R_phys']
    sigma = data['sigma_phys']

    intpnts = lpars[-1]
    integrator = simps

    aperture_phys = distance_to_physical(objmodel, aperture)
    lpars_phys    = lpars[:]
    lpars_phys[1] = distance_to_physical(objmodel, lpars_phys[1])

    #print aperture_phys, lpars_phys

    #-------------------------------------------------------------------------
    # Calculate the Abel integral to obtain rho(r)
    #-------------------------------------------------------------------------
    r = logspace(log10(min(amin(R),lpars_phys[1]) / 100),
                 log10(amax(R) * 10), num=interpnts)

    #print 'min(R)=%f, max(R)=%f' % (amin(R), amax(R))
    #print r

    #print 'abelsolve'
    rho, mass3d = cumsolve(r, imagemin, imagemax,
                           integrator, intpnts, alphalim,
                           R, sigma, mass)
 
    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp      = 6.67e-11 * 1.989e30 / 3.086e19
    sigp     = sigpsolve(r,rho,mass3d,integrator,intpnts,alphalim,Gsp,light,lpars,beta) / 1000
    sigpsing = sigpsingle(r,sigp,light,lpars_phys,aperture_phys,integrator)

    rhint   = rhoint(imagemin,imagemax,alphalim,r,rho,mass3d,r)
    drho    = dlnrhodlnr(r,rhint)

    data['sigp:r'        ] = r
    data['sigp:rho'      ] = rho
    data['sigp:rhoint'   ] = rhint
    data['sigp:drho'     ] = drho
    data['sigp:mass3d'   ] = mass3d
    data['sigp:sigp'     ] = sigp
    data['sigp:sigp_sing'] = sigpsing

    print 'Final rms mean projected vel. dispersion:', sigpsing

def sigpf(objmodel, vdisp, tol, chisq_cut):
    """Return True if chi-squared value for the object's sigp is <= chisq_cut."""
    obj,data = objmodel
    return ((vdisp**2 - data['sigp:sigp_sing']**2) / tol**2) <= chisq_cut

