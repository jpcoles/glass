from numpy import logspace, log10
from spherical_deproject import abelsolve, sigpsolve, sigpsingle

def sigp(vdisp, light, lpars, aperture, beta):
    def sigp_filter(models):
        for m in models:
            for objmodel in m['objs']:
                sp = _sigp(objmodel, light, lpars, aperture, beta)
                if (sp > vdisp)
                    m['tagged'] = True
                    break
    return sigp_filter

def _sigp(objmodel, light, lpars, aperture, beta):

    obj, data = objmodel

    #-------------------------------------------------------------------------
    # Find the inner and outer images for the object.
    #-------------------------------------------------------------------------
    rs = [ abs(img.pos) for sys in obj.systems for img in sys.images ]
    imagemin, imagemax = min(rs), max(rs)

    #-------------------------------------------------------------------------
    # Calculate the Abel integral to obtain rho(r)
    #-------------------------------------------------------------------------
    r = logspace(log10(min(data['R']) / 10),
                 log10(max(data['R']) * 10), num=interpnts)

    rho = abelsolve(r, imagemin, imagemax,
                    integrator, intpnts, alphalim,
                    data['R'], data['sigma'], mass)
 
    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp      = 6.67e-11 * 1.989e30 / 3.086e19
    sigp     = sigpsolve(r,rho,integrator,intpnts,alphalim,Gsp,light,lpars,beta) / 1000
    sigpsing = sigpsingle(r,sigp,light,lpars,aperture,integrator)

    print 'Final rms mean projected vel. dispersion:',sigpsing
    return sigpsing

