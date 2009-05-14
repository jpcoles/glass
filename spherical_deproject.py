'''
   #-----------------------------------------------------------------------------
   # deproject.py | version 0.0 | Justin Read 2009 
   #-----------------------------------------------------------------------------

   These routines form the Abel deprojection module for GLASS.
   abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
             R,sigma,mass)
     - deprojects a surface density profile sigma(R) with enclosed
       surface mass profile mass(R) to produce rho(r).
     - imagemin/max define the range over which sigma(R) is to be believed
     - intpnts is the number of integration points
     - integrator defines the choice of integration technique (simps/trapz)
     - alphalim sets the min. outer slope steepness (ensures "log-finite" mass)

   sigpsolve(r,rho,integrator,intpnts,upper,alphalim,Gsp,
             lightsurf,lpars,beta)
     - calculates the projected velocity dispersion profile sigp(R)
     - rho(r) is calculated in abelsolve()
     - integrator/intpnts/alphalim as above
     - Gsp is the value of the gravitational const. (sets the units)
     - lightsurf/lpars set the light distribution
     - beta sets the constant velocity aniostropy

   sigpsingle(rin,sigp,light,lpars,aperture,integrator):
     - Reduces the projected velocity disp profile to a single
       mean dispersion value rms averaged over some aperture
     - aperture must be passed with the same units as rin'''

import sys
from operator import isNumberType
from numpy import loadtxt
from numpy import interp
from numpy import linspace, logspace, empty, zeros
from numpy import pi
from numpy import cos, exp, log10, log, sqrt, arccos
from scipy.integrate.quadrature import simps, trapz
from scipy.integrate import quad 
from scipy.misc.common import derivative

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def sersicsurf(r,pars):
    '''Defines a sersic profile for the *projected* light.
       pars[0]=sigma0
       pars[1]=re
       pars[2]=n'''   

    bn = 1.9992*pars[2]-0.3271
    return pars[0]*exp(-bn*((r/pars[1])**(1.0/pars[2])-1.0))

def sersicden(r,pars):
    '''Calculates the density for the sersic profile (solves the Abel integral).
       pars[0]=sigma0
       pars[1]=re
       pars[2]=n
       pars[3]=intpnts'''

    intpnts = pars[3]
    bn = 1.9992*pars[2]-0.3271
    theta = linspace(0.,pi/2.-1.0e-6,num=intpnts)
    cth = cos(theta)
    rho = empty(len(r), 'double')
    for i in xrange(len(r)):
        y = exp(-bn*((r[i]/cth/pars[1])**(1./pars[2])-1.0))*\
            (-bn/pars[1]**(1./pars[2])*(r[i]/cth)**(1./pars[2]-1.0)*1./pars[2])
        rho[i] = (-1./pi)*integrator(y/cth,theta)
    return rho

def sersic():
    return sersicsurf, sersicden

def hernquistsurf(r,pars):
    '''Calculates a *projected* hernquist-alpha profile for the light.
       pars[0]=rho0
       pars[1]=a
       pars[2]=alp
       pars[3]=intpnts'''
    
    intpnts = pars[3]
    theta = linspace(0.,pi/2.-1.0e-6,num=intpnts)
    cth = cos(theta)
    cth2 = cth**2.0
    surf = empty(len(r), 'double')
    for i in xrange(len(r)):
        q = r[i]/cth
        y = pars[0]*(3.0-pars[2])/(4.0*pi*pars[1]**3.0)/\
            ((q/pars[1])**pars[2]*(1.0+(q/pars[1]))**(4.0-pars[2]))
        surf[i] = 2.0*r[i]*integrator(y/cth2,theta)
    return surf

def hernquistden(r,pars):    
    '''Defines a hernquist alpha profile for the light. 
    pars[0]=rho0
    pars[1]=a
    pars[2]=alp'''

    return pars[0]*(3.0-pars[2])/(4.0*pi*pars[1]**3.0)*\
           (r/pars[1])**(-pars[2])*(1.0+r/pars[1])**(-4.0+pars[2])

def hernquist():
    return hernquistsurf, hernquistden 

def thetaintintegrand(theta,beta):
    return cos(theta)**(2.0*beta-2.0)*(1.0-beta*cos(theta)**2.0)
def thetaint(beta,a):
    '''Theta integral for inner integral in sigp(r) calc'''
    return quad(thetaintintegrand, 0, a, args=beta)[0]

def dSigdR(rmin,rmax,alphalim,R,sigma,mass,rin):
    '''dSigdR function to calculate and interpolate derivative of the surface
       density.'''

    # Check rmin/rmax values. This bit assumes ascending R values.
    # We use R[1] and not R[0] for the low range because interpolation
    # at R[0] is unreliable. 
    if rmin < min(R):
        rmin = R[1]
    if rmax > max(R):
        rmax = max(R)

    # Calculate inner/outer power law match values:
    sigc_in = interp(rmin,R,sigma)
    sigcd_in = derivative(lambda x: interp(x,R,sigma), rmin)  
    sigc_out = interp(rmax,R,sigma)
    sigcd_out = derivative(lambda x: interp(x,R,sigma), rmax)

    # Inner power [consistent with known enclosed mass & smooth simga]
    menc_in_real = interp(rmin,R,mass)
    alpin = 2.0 - sigc_in*2.0*pi/menc_in_real*rmin**2.0
    Ain = sigc_in * rmin**alpin

    # Outer power [consistent with smooth sigma & sigma' & finite mass]:
    alpout = -sigcd_out/sigc_out * rmax

    # Check outer power law implies finite mass:
    if alpout < alphalim:
        alpout = alphalim
    Aout = sigc_out * rmax**alpout

    # Interpolate for r<rmax; power law for r>rmax:
    output = empty(len(rin), 'double')
    output = derivative(lambda x: interp(x,R,sigma), rin)
    w = rin > rmax
    output[w] = -alpout*Aout*rin[w]**(-alpout-1.0)
    w = rin < rmin
    output[w] = -alpin*Ain*rin[w]**(-alpin-1.0)

    return output

def masstot(rmax,alphalim,r,rho,massr,rin):
    '''Function to interpolate the cumulative mass beyond the data [consistent
       with the assumed surface density outer power law distribution]'''

    # Check rmax value:
    if rmax > max(r):
        rmax = max(r)

    # Find the mass match point:
    gam = alphalim + 1.0
    m0 = interp(rmax,r,massr)
    Ac = interp(rmax,r,rho)*rmax**gam
 
    # Interpolate cumulative mass
    output = empty_like(rin)
    mtot = interp(rin,r,massr)
    w = rin > rmax
    if -gam+3.0 < 0:
        mtot[w] = m0 + 4.0*pi*Ac/(3.-gam)*(rin[w]**(3.0-gam)-rmax**(3.0-gam))
    else:
        mtot[w] = m0 + 4.0*pi*Ac*(log(rin[w]/rmax))

    return mtot

def gRr(r,beta,lower):
    '''Inner integral function for sigp(r)'''

    # Error checking:
    assert min(r)**2.0 - lower**2.0 >= 0,\
           'lower %f > min(r) %f' % (lower,min(r))
    assert beta <= 1,\
           'anisotropy beta %f > 1' % beta

    # Treat beta=0 as a special case:
    if beta == 0:
        return sqrt(r**2.0 - lower**2.0)
    else:
        # General solution:
        vec_thetaint = vectorize(thetaint)
        return lower**(1.0-2.0*beta)*vec_thetaint(beta,arccos(lower/r))

def sphericalcumulate(r,array,integrator):
    '''Calculate the spherical cumulative "mass" of an array'''

    intpnts = len(array)
    out = zeros(intpnts,'double')
    for i in xrange(1,intpnts):
        out[i] = integrator(4.0*pi*r[0:i]**2.0*array[0:i],r[0:i])
    return out

def abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,R,sigma,mass):
    '''Solve the Abel integral to obtain rho(r)'''

    # Some asserts to check the inputs are all sensible:
    assert imagemin >= 0.,\
           'Imagemin %f < 0' % imagemin
    assert imagemin < imagemax,\
           'Imagemin %f > Imagemax %f' % (imagemin, imagemax)
    assert imagemax > 0.,\
           'Imagemax %f < 0' % imagemax
    assert intpnts > 0,\
           'inpnts %i <= 0' % intpnts
    assert alphalim > 0,\
           'alphalim %f < 0' % alphalim
    assert alphalim >= 2,\
           'alphalim %f < 2 (this alphalim gives > "log-infinite" mass)' % alphalim

    theta = linspace(0.,pi/2.-1.0e-6,num=intpnts)
    cth = cos(theta)
    rho = empty(len(r), 'double')
    for i in xrange(len(r)):
        y = dSigdR(imagemin,imagemax,alphalim,
                   R,sigma,mass,r[i]/cth)/cth
        rho[i] = (-1./pi)*integrator(y,theta)
    return rho

def sigpsolve(r,rho,integrator,intpnts,alphalim,Gsp,
              light,lpars,beta):
    '''Solve the integral to obtain sigp(r). See Wilkinson et al. 2004
       for details. Note typos in equation (1) of their paper. Int.
       limits for f(r) should be r-->infty and GM(r)/r should be GM(r)/r**2.0'''

    lightsurf, lightden = light
    massr = sphericalcumulate(r,rho,integrator)
    rmax = max(r)
    sigp2 = empty(len(r), 'double')

    theta = linspace(0.,pi/2.-1.0e-6,num=intpnts)
    cth = cos(theta)
    cth2 = cth**2.0
    sth = sin(theta)

    for i in xrange(len(r)):
        lower = r[i]
        rint = lower/cth
        rhostar = lightden(rint,lpars)
        mtot = masstot(rmax,alphalim,r,rho,massr,rint)
        integrand = rhostar*rint**(2.0*beta-2.0)*Gsp*\
                    mtot*gRr(rint,beta,lower)
        sigp2[i] = integrator(integrand*lower*sth/cth2,theta)
    sigp2 = sigp2 * 2.0/lightsurf(r,lpars)
    sigp = sqrt(sigp2)

    return sigp

def sigpsingle(rin,sigp,light,lpars,aperture,integrator):
    '''Reduce the projected velocity disp profile to a single
       mean dispersion value rms averaged over some aperture.
       aperture must be passed with the same units as rin'''

    ap = 0
    while rin[ap] < aperture:
        ap=ap+1

    lightsurf, lightden = light
    R = rin[:ap]
    sigp2 = sigp[:ap]**2.0
    IR = lightsurf(R,lpars)

    return sqrt(integrator(sigp2*IR*2.0*pi*R,R)/\
                integrator(IR*2.0*pi*R,R))

#-----------------------------------------------------------------------------
# Main program
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import *

    #-------------------------------------------------------------------------
    # Parameters
    #-------------------------------------------------------------------------

    #Range of believeable data + outer slope limiter:
    imagemin = 1.
    imagemax = 20.
    alphalim = 2.

    #Integrator options [simps/trapz] + number of points to use:
    integrator = simps
    intpnts = 100 
    interpnts = 500
    
    #Light distribution parameters + vel anisotropy: 
    light = hernquist()
    lightsurf, lightden = light
    lpars = [1.0,15.0,1.0,intpnts]
    beta = 0.
    aperture = 30.

    #Input files [sigma(R) and mass(R)]:
    surffile = './Data/surftest.txt'
    massfile = './Data/masstesthi.txt'
    
    #Files containing correct answers:
    surftest = './Data/surftest.txt'
    rhotest = './Data/rhotest.txt'
    masstest = './Data/massthreetest.txt'
    sigtest = './Data/sigtest.txt'

    #Output directory:
    outdir = './Data/'
    
    #-------------------------------------------------------------------------
    # Read in the surface density and mass profiles
    #-------------------------------------------------------------------------
    print 'Reading data from:', surffile
    f1 = loadtxt(surffile,
                 dtype = {'names': ('R', 'sigma'),
                          'formats': ('f8', 'f8')})
    print len(f1), 'lines successfully read from surffile...'
    f2 = loadtxt(massfile,
                 dtype = {'names': ('R', 'mass'),
                          'formats': ('f8', 'f8')})
    print len(f2), 'lines successfully read from massfile...'

    # Interpolate mass file to surface density file. This bit
    # is *only* for code testing. It ensures an accurate value
    # for the enclosed mass (as should be the case also for
    # real lensing data):
    mass = interp(f1['R'],f2['R'],f2['mass'])
    
    #-------------------------------------------------------------------------
    # Calculate the Abel integral to obtain rho(r) and mass(r)
    #-------------------------------------------------------------------------
    r = logspace(log10(min(f1['R'])/10.),log10(max(f1['R'])*10.),
                 num=interpnts)
    rho = abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
                    f1['R'],f1['sigma'],mass)
    massr = sphericalcumulate(r,rho,integrator)
 
    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp = 6.67e-11 * 1.989e30 / 3.086e19
    sigp = sigpsolve(r,rho,integrator,intpnts,alphalim,Gsp,
                     light,lpars,beta)/1000.
    sigpsing = sigpsingle(r,sigp,light,lpars,aperture,integrator)

    print 'Final rms mean projected vel. dispersion:',sigpsing

    #-------------------------------------------------------------------------
    # Plot the results: test the light functions
    #-------------------------------------------------------------------------
    rl = linspace(0.1,50.,num=1000)
    hernden = hernquistden(rl,lpars)
    hernsurf = hernquistsurf(rl,lpars)
    sercden = sersicden(rl,lpars)
    sercsurf = sersicsurf(rl,lpars)
    figure()
    loglog(rl,sercden,label='Sersic density')
    loglog(rl,sercsurf,label='Sersic surface density')
    loglog(rl,hernden,label='Hernquist density')
    loglog(rl,hernsurf,label='Hernquist surface density')
    title('Testing light functions')
    xlabel(r'$r,R(\mathrm{kpc})$')
    ylabel(r'$\Sigma(R),\rho(r)$')
    legend()
    savefig(outdir+'light.pdf')
    
    #-------------------------------------------------------------------------
    # Plot the results: testing the interpolants
    #-------------------------------------------------------------------------
    figure()
    Rint = linspace(0.,imagemax,num=intpnts)
    dsigma = derivative(lambda x: interp(x,f1['R'],f1['sigma']), f1['R'])
    plot(f1['R'], f1['sigma'],label='Data')
    plot(f1['R'], dsigma,label='Gradient')
    plot([imagemax,imagemax],[min(dsigma),max(dsigma)])
    plot([imagemin,imagemin],[min(dsigma),max(dsigma)])
    title('Testing interpolants')
    xlabel(r'$R(\mathrm{kpc})$')
    ylabel(r'$\Sigma(R)$')
    legend()
    savefig(outdir+'interp.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: input surface density profile
    #-------------------------------------------------------------------------
    if imagemax > max(f1['R']):
        imagemax = max(f1['R'])
    if imagemin < min(f1['R']):
        imagemin = f1['R'][1]
    f2 = loadtxt(surftest,
                 dtype = {'names': ('R', 'sigma'),
                          'formats': ('f8', 'f8')})
    figure()
    loglog(f1['R'], f1['sigma'],label='Python input')
    plot(f2['R'], f2['sigma'],label='IDL input')
    plot([imagemax,imagemax],[min(f1['sigma']),max(f1['sigma'])])
    plot([imagemin,imagemin],[min(f1['sigma']),max(f1['sigma'])])
    title('Surface density')
    xlabel(r'$R(\mathrm{kpc})$')
    ylabel(r'$\Sigma(R)$')
    legend()
    savefig(outdir+'surfden.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: Density(r)
    #-------------------------------------------------------------------------
    f2 = loadtxt(rhotest,
                 dtype = {'names': ('r', 'rho'),
                          'formats': ('f8', 'f8')})
    figure()
    loglog(r,rho,label='Python result')
    plot(f2['r'],f2['rho'],label='Right result')
    plot([imagemax,imagemax],[min(rho),max(rho)])
    plot([imagemin,imagemin],[min(rho),max(rho)])
    title('Deprojected data')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$\rho(r)$')
    legend()
    savefig(outdir+'rho.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: Mass(r)
    #-------------------------------------------------------------------------
    #Read in the correct answer to compare:
    f2 = loadtxt(masstest,
                 dtype = {'names': ('r', 'mass'),
                          'formats': ('f8', 'f8')})

    rlong = linspace(0,1000,num=1000)
    masst = masstot(max(r),alphalim,r,rho,massr,rlong)

    figure()
    plot(r,massr,label='Python result')
    plot(f2['r'],f2['mass'],label='Right result')
    plot(rlong,masst,label='Interpolated cumulative mass')
    plot([imagemax,imagemax],[min(massr),max(massr)])
    plot([imagemin,imagemin],[min(massr),max(massr)])
    title('Cumulative mass')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$M(r)$')
    legend()
    savefig(outdir+'mass.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: sigp(r)
    #-------------------------------------------------------------------------
    #Read in the correct answer to compare:
    f2 = loadtxt(sigtest,
                 dtype = {'names': ('r', 'sigp'),
                          'formats': ('f8', 'f8')})
    figure()
    plot(r,sigp,label='Python result')
    plot(f2['r'],f2['sigp'],label='Right result')
    plot([imagemax,imagemax],[min(sigp),max(sigp)])
    plot([imagemin,imagemin],[min(sigp),max(sigp)])
    gca().set_xlim(0,imagemax*1.5)
    title('Projected velocity dispersion')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$\sigma_p(r)$')
    legend()
    savefig(outdir+'sigp.pdf')
    show()

    



