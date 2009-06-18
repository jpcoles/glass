'''
   #------------------------------------------------------------------------
   # spherical_deproject.py | version 0.0 | Justin Read 2009 
   #------------------------------------------------------------------------

   These routines form the Abel deprojection module for GLASS.

   abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
             R,sigma,mass)
     - like cumsolve, but calculates first rho(r) then
       integrates to get mass(r)
     - deprojects a surface density profile sigma(R) with enclosed
       surface mass profile massp(R) to produce rho(r).and mass(r)
     - imagemin/max define the range over which sigma(R) is to be believed
     - intpnts is the number of integration points
     - integrator defines the choice of integration technique (simps/trapz)
     - alphalim sets the min. outer slope steepness (ensures "log-finite" mass)

   cumsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
             R,sigma,mass)
     - like abelsolve, but calculates first mass(r) then diff.
       to get rho(r)
     - deprojects a surface density profile sigma(R) with enclosed
       surface mass profile massp(R) to produce rho(r) and mass(r).
     - imagemin/max define the range over which sigma(R) is to be believed
     - intpnts is the number of integration points
     - integrator defines the choice of integration technique (simps/trapz)
     - alphalim sets the min. outer slope steepness (ensures "log-finite" mass)

   dlnrhodlnr(r,rho)
     - Numerically differentiates the density distribution
       to obtain a non-parameteric measure of the power
       law exponent as a function of radius

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
from numpy import loadtxt, argmin
from numpy import interp
from numpy import linspace, logspace, empty, zeros, empty_like, vectorize
from numpy import pi, amin, amax
from numpy import sin, cos, exp, log10, log, sqrt, arccos, arctan
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

def dSigdR(rmin,rmax,alphalim,R,sigma,massp,rin):
    '''dSigdR function to calculate and interpolate derivative of the surface
       density.'''

    # Find array elements in R corresponding to rmin/rmax:
    # We use R[1] and not R[0] for the low range because interpolation
    # at R[0] is unreliable. 
    jl =  1 if rmin < amin(R) else argmin(abs(R-rmin))
    jr = -1 if rmax > amax(R) else argmin(abs(R-rmax))        
    rmin = R[jl]
    rmax = R[jr]

    # Calculate inner/outer power law match values:
    sigc_in   = sigma[jl]
    sigc_out  = sigma[jr]

    # Inner power [consistent with known enclosed mass & smooth simga]
    menc_in_real = massp[jl]
    alpin = 2 - sigc_in*2*pi/menc_in_real*rmin**2
    Ain = sigc_in * rmin**alpin

    # Outer power [consistent with alphalim]
    Aout = sigc_out * rmax**alphalim

    # Interpolate for r<rmax; power law for r>rmax:
    output = empty(len(rin), 'double')
    output = derivative(lambda x: interp(x,R,sigma), rin)
    w = rin > rmax
    output[w] = -alphalim*Aout*rin[w]**(-alphalim-1)
    w = rin < rmin
    output[w] = -alpin*Ain*rin[w]**(-alpin-1)

    return output

def sigmaint(rmin,rmax,alphalim,R,sigma,massp,rin):
    '''Sigma function to interpolate the surface density.
       It sews on power laws where there is no available
       data.'''
    
    # Find array elements in R corresponding to rmin/rmax:
    # We use R[1] and not R[0] for the low range because interpolation
    # at R[0] is unreliable. 
    jl =  1 if rmin < amin(R) else argmin(abs(R-rmin))
    jr = -1 if rmax > amax(R) else argmin(abs(R-rmax))        
    rmin = R[jl]
    rmax = R[jr]

    # Calculate inner/outer power law match values:
    sigc_in   = sigma[jl]
    sigc_out  = sigma[jr]

    # Inner power [consistent with known enclosed mass & smooth simga]
    menc_in_real = massp[jl]
    alpin = 2 - sigc_in*2*pi/menc_in_real*rmin**2
    Ain = sigc_in * rmin**alpin

    # Outer power consistent with alphalim
    Aout = sigc_out * rmax**alphalim

    # Interpolate for r<rmax; power law for r>rmax:
    output = empty(len(rin), 'double')
    output = interp(rin,R,sigma)
    w = rin > rmax
    output[w] = Aout*rin[w]**(-alphalim)
    w = rin < rmin
    output[w] = Ain*rin[w]**(-alpin)
    w = rin = 0
    output[w] = 0

    return output

def rhoint(rmin,rmax,alphalim,r,rho,mass,rin):
    '''Sigma function to interpolate the surface density.
       It sews on power laws where there is no available
       data.'''
    
    # Find array elements in r corresponding to rmin/rmax:
    # We use r[1] and not r[0] for the low range because interpolation
    # at r[0] is unreliable. 
    jl =  1 if rmin < amin(r) else argmin(abs(r-rmin))
    jr = -1 if rmax > amax(r) else argmin(abs(r-rmax))        
    rmin = r[jl]
    rmax = r[jr]

    # Calculate inner/outer power law match values:
    rhoc_in   = rho[jl]
    rhoc_out  = rho[jr]

    # Inner power [consistent with known enclosed mass & smooth rho]
    menc_in_real = mass[jl]
    gamin = 3 - rhoc_in*4*pi/menc_in_real*rmin**3
    Ain = rhoc_in * rmin**gamin

    # Outer power consistent with alphalim
    gamout = alphalim + 1
    Aout = rhoc_out * rmax**gamout

    # Interpolate for r<rmax; power law for r>rmax:
    output = empty(len(rin), 'double')
    output = interp(rin,r,rho)
    w = rin > rmax
    output[w] = Aout*rin[w]**(-gamout)
    w = rin < rmin
    output[w] = Ain*rin[w]**(-gamin)

    return output

def masspint(rmin,rmax,alphalim,R,sigma,massp,rin):
    '''Projected enclosed mass function interpolation
       It sews on power laws where there is no available
       data.'''

    # Find array elements in R corresponding to rmin/rmax:
    # We use R[1] and not R[0] for the low range because interpolation
    # at R[0] is unreliable. 
    jl =  1 if rmin < amin(R) else argmin(abs(R-rmin))
    jr = -1 if rmax > amax(R) else argmin(abs(R-rmax))        
    rmin = R[jl]
    rmax = R[jr]

    # Calculate inner/outer power law match values:
    sigc_in   = sigma[jl]
    sigc_out  = sigma[jr]

    # Inner power [consistent with known enclosed mass & smooth simga]
    menc_in_real = massp[jl]
    alpin = 2 - sigc_in*2*pi/menc_in_real*rmin**2
    Ain = sigc_in * rmin**alpin

    # Find the outer match point:
    m0 = massp[jr]
    Aout = sigc_out * rmax**alphalim

    # Interpolate for r<rmax; power law for r>rmax:
    output = empty(len(rin), 'double')
    output = interp(rin,R,massp)
    w = rin > rmax
    if -alphalim+2 < 0:
        output[w] = m0 + 2*pi*Aout/(2-alphalim)*\
                    (rin[w]**(2-alphalim)-rmax**(2-alphalim))
    else:
        output[w] = m0 + 2*pi*Aout*(log(rin[w]/rmax))
    w = rin < rmin
    output[w] = 2*pi*Ain*rin[w]**(-alpin+2)/(2-alpin)

    return output

def masstot(rmax,alphalim,r,rho,mass,rin):
    '''Function to interpolate the cumulative mass beyond the data [consistent
       with the assumed surface density outer power law distribution]'''

    # Find array elements in r corresponding to rmax:
    jr = -1 if rmax > amax(r) else argmin(abs(r-rmax))        
    rmax = r[jr]

    # Find the mass match point:
    gam = alphalim + 1
    m0 = mass[jr]
    Ac = rho[jr]*rmax**gam
 
    # Interpolate cumulative mass
    mtot = interp(rin,r,mass)
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
    elif beta == 1:
        p = arccos(lower/r)
        p *= 2
        return 1/lower * (p - sin(p)) / 4
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

def abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,R,sigma,massp):
    '''Solve the Abel integral to obtain rho(r) and then M(r)'''

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
           'alphalim %f < 2 (this alphalim gives > "log-infinite" mass)' % \
           alphalim

    theta = linspace(0,pi/2-1e-6,num=intpnts)
    cth = cos(theta)
    rhoout = empty(len(r), 'double')
    for i in xrange(len(r)):
        y = dSigdR(imagemin,imagemax,alphalim,
                   R,sigma,massp,r[i]/cth)/cth
        rhoout[i] = (-1/pi)*integrator(y,theta)
        
    # Calculate the cumulative mass:
    massout = sphericalcumulate(r,rhoout,integrator)

    return rhoout, massout

def cumsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,R,sigma,massp):
    '''Solve the Abel integral to obtain M(r) and then rho(r)'''

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
           'alphalim %f < 2 (this alphalim gives > "log-infinite" mass)' % \
           alphalim

    theta = linspace(0,pi/2-1e-6,num=intpnts)
    cth = cos(theta)
    sth = sin(theta)
    massout = empty(len(r), 'double')
    for i in xrange(len(r)):
        y = sigmaint(imagemin,imagemax,alphalim,R,sigma,massp,r[i]/cth)*\
            (1/cth**2-sth/cth**3*arctan(cth/sth))        
        massout[i] = -4*r[i]**2*integrator(y,theta)

    massout = massout + \
              masspint(imagemin,imagemax,alphalim,R,sigma,massp,r)

    # Calculate the density as the derivative of massout:
    rhoout =  derivative(lambda x: interp(x,r,massout,right=0),r)/\
             (4*pi*r**2)
    rhoout[-1] = 0
    
    return rhoout, massout

def dlnrhodlnr(r,rho):
    '''Numerically differentiates the density distribution
       to obtain a non-parameteric measure of the power
       law exponent as a function of radius'''
    lnr = log(r)
    lnrho = log(rho)

    f = derivative(lambda x: interp(x,lnr,lnrho),lnr)

    return f

def sigpsolve(r,rho,mass,integrator,intpnts,alphalim,Gsp,
              light,lpars,beta):
    '''Solve the integral to obtain sigp(r). See Wilkinson et al. 2004
       for details. Note typos in equation (1) of their paper. Int.
       limits for f(r) should be r-->infty and GM(r)/r should be GM(r)/r**2'''

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
        mtot = masstot(rmax,alphalim,r,rho,mass,rint)
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
    alphalim = 2

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
    surffile_true = datadir+'surftest.txt'
    rhofile_true = datadir+'rhotest.txt'
    massfile_true = datadir+'massthreetest.txt'
    sigfile_true = datadir+'sigtest.txt'

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
    massp = interp(f1['R'],f2['R'],f2['mass'])
    
    #-------------------------------------------------------------------------
    # Calculate the Abel integral to obtain rho(r) and mass(r)
    #-------------------------------------------------------------------------
    r = logspace(log10(amin(f1['R'])/10),log10(amax(f1['R'])*10),
                 num=interpnts)
    #rho, mass = abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
    #                      f1['R'],f1['sigma'],massp)
    rho, mass = cumsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
                         f1['R'],f1['sigma'],massp)

    rinterp = logspace(-1,3,num=1000)
    rhint = rhoint(imagemin,imagemax,alphalim,r,
                   rho,mass,rinterp)

    drho = dlnrhodlnr(rinterp,rhint)

    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp = 6.67e-11 * 1.989e30 / 3.086e19
    sigp = sigpsolve(r,rho,mass,integrator,intpnts,alphalim,Gsp,
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
    #savefig(outdir+'interp.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: input surface density profile
    #-------------------------------------------------------------------------
    if imagemax > amax(f1['R']):
        imagemax = amax(f1['R'])
    if imagemin < amin(f1['R']):
        imagemin = f1['R'][1]
    f2 = loadtxt(surffile_true,
                 dtype = {'names': ('R', 'sigma'),
                          'formats': ('f8', 'f8')})

    sigint = sigmaint(imagemin,imagemax,alphalim,f1['R'],
                      f1['sigma'],massp,rinterp)

    figure()
    loglog(f1['R'], f1['sigma'],label='Python input')
    plot(f2['R'], f2['sigma'],label='IDL input')
    plot(rinterp,sigint,label='Interpolant')
    plot([imagemax,imagemax],[amin(f1['sigma']),amax(f1['sigma'])])
    plot([imagemin,imagemin],[amin(f1['sigma']),amax(f1['sigma'])])
    gca().set_xlim(0,imagemax*10)
    title('Surface density')
    xlabel(r'$R(\mathrm{kpc})$')
    ylabel(r'$\Sigma(R)$')
    legend()
    #savefig(outdir+'surfden.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: Density(r)
    #-------------------------------------------------------------------------
    f2 = loadtxt(rhofile_true,
                 dtype = {'names': ('r', 'rho'),
                          'formats': ('f8', 'f8')})
    figure()
    loglog(r,rho,label='Python result')
    plot(rinterp,rhint,label='Interpolant')
    plot(f2['r'],f2['rho'],label='Right result')
    plot([imagemax,imagemax],[1e-2,amax(rho)])
    plot([imagemin,imagemin],[1e-2,amax(rho)])
    gca().set_xlim(0,imagemax*10)
    title('Deprojected data')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$\rho(r)$')
    legend()
    #savefig(outdir+'rho.pdf')
    
    #-------------------------------------------------------------------------
    # Plot the results: drho
    #-------------------------------------------------------------------------
    import massmodel.hernquist as hernquist
    rtest = logspace(-2,3,num=10000)
    rhotest = hernquist.den(rtest,[2.12e11,15,1])
    drhotest = dlnrhodlnr(rtest,rhotest)

    figure()
    plot(rinterp,drho,label='Python result')
    plot(rtest,drhotest,label='Right result')
    plot([imagemax,imagemax],[amin(drho),amax(drho)])
    plot([imagemin,imagemin],[amin(drho),amax(drho)])
    gca().set_xlim(0,imagemax*10)
    title('Power law exponent of density')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$d\ln\rho/d\lnr$')
    legend()
    #savefig(outdir+'rho.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: Mass(r)
    #-------------------------------------------------------------------------
    #Read in the correct answer to compare:
    f2 = loadtxt(massfile_true,
                 dtype = {'names': ('r', 'mass'),
                          'formats': ('f8', 'f8')})

    mpint = masspint(imagemin,imagemax,alphalim,f1['R'],
                     f1['sigma'],massp,rinterp)
    mtint = masstot(imagemax,alphalim,r,rho,mass,rinterp)
    
    figure()
    loglog(r,mass,label='Python result')
    plot(f2['r'],f2['mass'],label='Right result')
    plot(f1['R'],massp,label='Projected cumulative mass')
    plot(rinterp,mpint,label='Projected interpolant')
    plot(rinterp,mtint,label='3D interpolant')
    plot([imagemax,imagemax],[amin(mass),amax(mass)])
    plot([imagemin,imagemin],[amin(mass),amax(mass)])
    gca().set_xlim(imagemin/10,imagemax*10)
    gca().set_ylim(1e7,1e12)
    title('Cumulative mass')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$M(r)$')
    legend()
    #savefig(outdir+'mass.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: sigp(r)
    #-------------------------------------------------------------------------
    #Read in the correct answer to compare:
    f2 = loadtxt(sigfile_true,
                 dtype = {'names': ('r', 'sigp'),
                          'formats': ('f8', 'f8')})
    figure()
    plot(r,sigp,label='Python result')
    plot(f2['r'],f2['sigp'],label='Right result')
    plot([imagemax,imagemax],[amin(sigp),amax(sigp)])
    plot([imagemin,imagemin],[amin(sigp),amax(sigp)])
    gca().set_xlim(0,imagemax*10)
    title('Projected velocity dispersion')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$\sigma_p(r)$')
    legend()
    #savefig(outdir+'sigp.pdf')
    show()

    



