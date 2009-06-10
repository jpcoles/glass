'''
   #------------------------------------------------------------------------
   # deproject.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

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
             light,lpars,beta)
     - calculates the projected velocity dispersion profile sigp(R)
     - rho(r) is calculated in abelsolve()
     - integrator/intpnts/alphalim as above
     - Gsp is the value of the gravitational const. (sets the units)
     - light+lpars set the light distribution [see massmodel folder]
     - beta sets the constant velocity aniostropy

   sigpsingle(rin,sigp,light,lpars,aperture,integrator):
     - Reduces the projected velocity disp profile to a single
       mean dispersion value rms averaged over some aperture
     - aperture must be passed with the same units as rin'''

from __future__ import division
import sys
from numpy import loadtxt
from numpy import interp
from numpy import linspace, logspace, empty, zeros
from numpy import pi, amin, amax
from numpy import cos, exp, log10, log, sqrt, arccos
from scipy.integrate.quadrature import simps, trapz
from scipy.integrate import quad 
from scipy.misc.common import derivative

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def thetaintintegrand(theta,beta):
    return cos(theta)**(2*beta-2)*(1-beta*cos(theta)**2)
def thetaint(beta,a):
    '''Theta integral for inner integral in sigp(r) calc'''
    return quad(thetaintintegrand, 0, a, args=beta)[0]

def dSigdR(rmin,rmax,alphalim,R,sigma,mass,rin):
    '''dSigdR function to calculate and interpolate derivative of the surface
       density.'''

    # Check rmin/rmax values. This bit assumes ascending R values.
    # We use R[1] and not R[0] for the low range because interpolation
    # at R[0] is unreliable. 
    if rmin < amin(R):
        rmin = R[1]
    if rmax > amax(R):
        rmax = amax(R)

    # Calculate inner/outer power law match values:
    sigc_in = interp(rmin,R,sigma)
    sigcd_in = derivative(lambda x: interp(x,R,sigma), rmin)  
    sigc_out = interp(rmax,R,sigma)
    sigcd_out = derivative(lambda x: interp(x,R,sigma), rmax)

    # Inner power [consistent with known enclosed mass & smooth simga]
    menc_in_real = interp(rmin,R,mass)
    alpin = 2 - sigc_in*2*pi/menc_in_real*rmin**2
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
    output[w] = -alpout*Aout*rin[w]**(-alpout-1)
    w = rin < rmin
    output[w] = -alpin*Ain*rin[w]**(-alpin-1)

    return output

def masstot(rmax,alphalim,r,rho,massr,rin):
    '''Function to interpolate the cumulative mass beyond the data [consistent
       with the assumed surface density outer power law distribution]'''

    # Check rmax value:
    if rmax > amax(r):
        rmax = amax(r)

    # Find the mass match point:
    gam = alphalim + 1
    m0 = interp(rmax,r,massr)
    Ac = interp(rmax,r,rho)*rmax**gam
 
    # Interpolate cumulative mass
    output = empty_like(rin)
    mtot = interp(rin,r,massr)
    w = rin > rmax
    if -gam+3 < 0:
        mtot[w] = m0 + 4*pi*Ac/(3.-gam)*(rin[w]**(3-gam)-rmax**(3-gam))
    else:
        mtot[w] = m0 + 4*pi*Ac*(log(rin[w]/rmax))

    return mtot

def gRr(r,beta,lower):
    '''Inner integral function for sigp(r)'''

    # Error checking:
    assert amin(r)**2 - lower**2 >= 0,\
           'lower %f > min(r) %f' % (lower,amin(r))
    assert beta <= 1,\
           'anisotropy beta %f > 1' % beta

    # Treat beta=0 as a special case:
    if beta == 0:
        return sqrt(r**2 - lower**2)
    else:
        # General solution:
        vec_thetaint = vectorize(thetaint)
        return lower**(1-2*beta)*vec_thetaint(beta,arccos(lower/r))

def sphericalcumulate(r,array,integrator):
    '''Calculate the spherical cumulative "mass" of an array'''

    intpnts = len(array)
    out = zeros(intpnts,'double')
    for i in xrange(1,intpnts):
        out[i] = integrator(4*pi*r[0:i]**2*array[0:i],r[0:i])
    return out

def abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,R,sigma,mass):
    '''Solve the Abel integral to obtain rho(r)'''

    # Some asserts to check the inputs are all sensible:
    assert imagemin >= 0,\
           'Imagemin %f < 0' % imagemin
    assert imagemin < imagemax,\
           'Imagemin %f > Imagemax %f' % (imagemin, imagemax)
    assert imagemax > 0,\
           'Imagemax %f < 0' % imagemax
    assert intpnts > 0,\
           'inpnts %i <= 0' % intpnts
    assert alphalim > 0,\
           'alphalim %f < 0' % alphalim
    assert alphalim >= 2,\
           'alphalim %f < 2 (this alphalim gives > "log-infinite" mass)' % alphalim

    theta = linspace(0,pi/2-1e-6,num=intpnts)
    cth = cos(theta)
    rho = empty(len(r), 'double')
    for i in xrange(len(r)):
        y = dSigdR(imagemin,imagemax,alphalim,
                   R,sigma,mass,r[i]/cth)/cth
        rho[i] = (-1/pi)*integrator(y,theta)
    return rho

def sigpsolve(r,rho,integrator,intpnts,alphalim,Gsp,
              light,lpars,beta):
    '''Solve the integral to obtain sigp(r). See Wilkinson et al. 2004
       for details. Note typos in equation (1) of their paper. Int.
       limits for f(r) should be r-->infty and GM(r)/r should be GM(r)/r**2'''

    massr = sphericalcumulate(r,rho,integrator)
    rmax = amax(r)
    sigp2 = empty(len(r), 'double')

    theta = linspace(0,pi/2-1e-6,num=intpnts)
    cth = cos(theta)
    cth2 = cth**2
    sth = sin(theta)

    for i in xrange(len(r)):
        lower = r[i]
        rint = lower/cth
        rhostar = light.den(rint,lpars)
        mtot = masstot(rmax,alphalim,r,rho,massr,rint)
        integrand = rhostar*rint**(2*beta-2)*Gsp*\
                    mtot*gRr(rint,beta,lower)
        sigp2[i] = integrator(integrand*lower*sth/cth2,theta)
    sigp2 = sigp2 * 2/light.surf(r,lpars)
    sigp = sqrt(sigp2)

    return sigp

def sigpsingle(rin,sigp,light,lpars,aperture,integrator):
    '''Reduce the projected velocity disp profile to a single
       mean dispersion value rms averaged over some aperture.
       aperture must be passed with the same units as rin'''

    ap = 0
    while rin[ap] < aperture:
        ap=ap+1

    R = rin[:ap]
    sigp2 = sigp[:ap]**2
    IR = light.surf(R,lpars)

    return sqrt(integrator(sigp2*IR*2*pi*R,R)/\
                integrator(IR*2*pi*R,R))

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
    imagemin = 1
    imagemax = 20
    alphalim = 4

    #Integrator options [simps/trapz] + number of points to use:
    integrator = simps
    intpnts = 100 
    interpnts = 500
    
    #Light distribution parameters + vel anisotropy: 
    import massmodel.hernquist as light
    lpars = [1,15,1,intpnts]
    beta = 0
    aperture = 30

    #Input files [sigma(R) and mass(R)]:
    datadir = '../Data/'
    surffile = datadir+'surftest.txt'
    massfile = datadir+'masstesthi.txt'
    
    #Files containing correct answers:
    surftest = datadir+'surftest.txt'
    rhotest = datadir+'rhotest.txt'
    masstest = datadir+'massthreetest.txt'
    sigtest = datadir+'sigtest.txt'

    #Output directory:
    outdir = datadir
    
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
    r = logspace(log10(amin(f1['R'])/10),log10(amax(f1['R'])*10),
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
                     light,lpars,beta)/1000
    sigpsing = sigpsingle(r,sigp,light,lpars,aperture,integrator)

    print 'Final rms mean projected vel. dispersion:',sigpsing

    #-------------------------------------------------------------------------
    # Plot the results: testing the interpolants
    #-------------------------------------------------------------------------
    figure()
    Rint = linspace(0,imagemax,num=intpnts)
    dsigma = derivative(lambda x: interp(x,f1['R'],f1['sigma']), f1['R'])
    plot(f1['R'], f1['sigma'],label='Data')
    plot(f1['R'], dsigma,label='Gradient')
    plot([imagemax,imagemax],[amin(dsigma),amax(dsigma)])
    plot([imagemin,imagemin],[amin(dsigma),amax(dsigma)])
    title('Testing interpolants')
    xlabel(r'$R(\mathrm{kpc})$')
    ylabel(r'$\Sigma(R)$')
    legend()
    savefig(outdir+'interp.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: input surface density profile
    #-------------------------------------------------------------------------
    if imagemax > amax(f1['R']):
        imagemax = amax(f1['R'])
    if imagemin < amin(f1['R']):
        imagemin = f1['R'][1]
    f2 = loadtxt(surftest,
                 dtype = {'names': ('R', 'sigma'),
                          'formats': ('f8', 'f8')})
    figure()
    loglog(f1['R'], f1['sigma'],label='Python input')
    plot(f2['R'], f2['sigma'],label='IDL input')
    plot([imagemax,imagemax],[amin(f1['sigma']),amax(f1['sigma'])])
    plot([imagemin,imagemin],[amin(f1['sigma']),amax(f1['sigma'])])
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
    plot([imagemax,imagemax],[amin(rho),amax(rho)])
    plot([imagemin,imagemin],[amin(rho),amax(rho)])
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
    masst = masstot(amax(r),alphalim,r,rho,massr,rlong)

    figure()
    plot(r,massr,label='Python result')
    plot(f2['r'],f2['mass'],label='Right result')
    plot(rlong,masst,label='Interpolated cumulative mass')
    plot([imagemax,imagemax],[amin(massr),amax(massr)])
    plot([imagemin,imagemin],[amin(massr),amax(massr)])
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
    plot([imagemax,imagemax],[amin(sigp),amax(sigp)])
    plot([imagemin,imagemin],[amin(sigp),amax(sigp)])
    gca().set_xlim(0,imagemax*1.5)
    title('Projected velocity dispersion')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$\sigma_p(r)$')
    legend()
    savefig(outdir+'sigp.pdf')
    show()

    



