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
             light,beta)
     - calculates the projected velocity dispersion profile sigp(R)
     - rho(r) is calculated in abelsolve()
     - integrator/intpnts/alphalim as above
     - Gsp is the value of the gravitational const. (sets the units)
     - light set the light distribution [see massmodel folder]
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
from scipy.optimize.zeros import bisect

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def thetaintintegrand(theta,beta):
    return cos(theta)**(2*beta-2)*(1-beta*cos(theta)**2)
def thetaint(beta,a):
    '''Theta integral for inner integral in sigp(r) calc'''
    return quad(thetaintintegrand, 0, a, args=beta)[0]

def findmatchpoints(rmin,rmax,alphalim,R,sigma,massp):
    '''Calculate array match points for interpolation.
       Match points are where we sew on a power law'''

    # Find array elements in R corresponding to rmin/rmax:
    # We use R[1] and not R[0] for the low range because interpolation
    # at R[0] is unreliable. 
    jl =  1 if rmin < R[1]  else argmin(abs(R-rmin))
    jr = -1 if rmax > R[-1] else argmin(abs(R-rmax))        
    rmin = R[jl]
    rmax = R[jr]

    # Calculate inner/outer power law match values:
    sigc_in   = sigma[jl]
    sigc_out  = sigma[jr]

    # Inner power [consistent with known enclosed mass & smooth simga]
    menc_in_real = massp[jl]
    alpin = 2 - sigc_in*2*pi/menc_in_real*rmin**2
    Ain = sigc_in * rmin**alpin

    # Find the outer match point. This ensures a continous mass
    # beyond rmax and a continuous first derivative by matching
    # also the density at this point:
    m0 = massp[jr]
    Aout = sigc_out * rmax**alphalim

    return rmin, rmax, sigc_in, sigc_out, Ain, alpin, Aout, m0

def dSigdR(rmin,rmax,alphalim,R,sigma,massp,rin):
    '''dSigdR function to calculate and interpolate derivative of the surface
    density. It sews on power laws where there are no data.'''

    # Find match points:
    rmin, rmax, sigc_in, sigc_out, Ain, alpin, Aout, m0 = \
          findmatchpoints(rmin,rmax,alphalim,R,sigma,massp)

    # Interpolate for rmin<r<rmax; power law otherwise:
    output = empty(len(rin), 'double')
    output = derivative(lambda x: interp(x,R,sigma), rin)
    w = rin > rmax
    output[w] = -alphalim*Aout*rin[w]**(-alphalim-1)
    w = rin < rmin
    output[w] = -alpin*Ain*rin[w]**(-alpin-1)

    return output

def sigmaint(rmin,rmax,alphalim,R,sigma,massp,rin):
    '''Sigma function to interpolate the surface density.  It sews on power
    laws where there are no available data.'''

    # Find match points:
    rmin, rmax, sigc_in, sigc_out, Ain, alpin, Aout, m0 = \
          findmatchpoints(rmin,rmax,alphalim,R,sigma,massp)

    # Interpolate for rmin<r<rmax; power law otherwise:
    output = empty(len(rin), 'double')
    output = interp(rin,R,sigma)
    w = rin > rmax
    output[w] = Aout*rin[w]**(-alphalim)
    w = rin < rmin
    output[w] = Ain*rin[w]**(-alpin)
    w = rin = 0
    output[w] = 0

    return output

def masspint(rmin,rmax,alphalim,R,sigma,massp,rin):
    '''Projected enclosed mass function interpolation It sews on power laws
    where there are no available data.'''

    # Find match points:
    rmin, rmax, sigc_in, sigc_out, Ain, alpin, Aout, m0 = \
          findmatchpoints(rmin,rmax,alphalim,R,sigma,massp)
 
    # Interpolate for rmin<r<rmax; power law otherwise:
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

def masstot(rmin,rmax,alphalim,r,rho,mass,R,sigma,massp,rin):
    '''Function to interpolate the cumulative mass beyond the data [consistent
    with the assumed surface density outer power law distribution]'''

    # Find match points:
    rmin, rmax, sigc_in, sigc_out, Ain, alpin, Aout, m0 = \
          findmatchpoints(rmin,rmax,alphalim,R,sigma,massp)

    # Find array elements in r corresponding to rmax:
    jr = -1 if rmax > r[-1] else argmin(abs(r-rmax))        
    rmax = r[jr]

    # Find the mass match point. We use m0 to make sure
    # the mass is continuous beyond rmax. And we use
    # the density to ensure a continuous first derivative.
    # This is the only place that the density is used in
    # masstot, and the only reason why it needs to be passed
    # to sigpsolve():
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

def sphericalcumulate(r,array,integrator,intpnts):
    '''Calculate the spherical cumulative "mass" of an array'''
    out = zeros(len(r),'double')
    for i in xrange(1,len(r)):
        rint = linspace(0,r[i],num=intpnts)
        arrayint = interp(rint,r,array)
        out[i] = integrator(4*pi*rint**2*arrayint,rint)
    return out

def abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,R,sigma,massp):
    '''Solve the Abel integral to obtain rho(r) and then M(r). This routine
    does not perform as well as cumsolve() because it takes the integral of a
    numerical derivative, rather than a numerical derivative of an integral.
    The difference is subtle but important. To get the vel. disp., sigpsolve()
    requires M(r) *not* rho(r). The density is only used to smoothly
    extrapolate M(r) beyond rmax. So M(r) is the more important quantity and
    should be calculated directly from the data.  i.e. use cumsolve() not
    abelsolve().'''

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
    massout = sphericalcumulate(r,rhoout,integrator,intpnts)

    return rhoout, massout

def cumsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,R,sigma,massp):
    '''Solve the Abel integral to obtain M(r) and then rho(r). This routine
    performs better than abelsolve() and ought to be used instead where
    possible.'''

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
    rhoout = derivative(lambda x: interp(x,r,massout,right=0),r)/\
             (4*pi*r**2)
    rhoout[-1] = 0

    return rhoout, massout

def dlnrhodlnr(rmin,rmax,alphalim,r,rho,mass,R,sigma,massp,rin):
    '''Numerically differentiates the density distribution to obtain a
    non-parameteric measure of the power law exponent as a function of
    radius'''

    lnr = log(r)
    lnrho = log(rho)
    lnrin = log(rin)

    # Find match points:
    rmin, rmax, sigc_in, sigc_out, Ain, alpin, Aout, m0 = \
          findmatchpoints(rmin,rmax,alphalim,R,sigma,massp)

    # Find array elements in r corresponding to rmin/rmax: We use r[1] and not
    # r[0] for the low range because interpolation at r[0] is unreliable. 
    jl =  1 if rmin < r[1]  else argmin(abs(r-rmin)) 
    jr = -1 if rmax > r[-1] else argmin(abs(r-rmax))        

    print rmin, rmax, r[jl], r[jr]

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

    # Interpolate for rmin<r<rmax; power law otherwise:
    output = empty(len(rin), 'double')
    output = derivative(lambda x: interp(x,lnr,lnrho,left=0,right=0),lnrin)

    w = rin > rmax
    output[w] = -gamout
    w = rin < rmin
    output[w] = -gamin

    return output

def sigpsolve(r,rho,mass,R,sigma,massp,integrator,intpnts,alphalim,Gsp,
              light,beta):
    '''Solve the integral to obtain sigp(r). See Wilkinson et al. 2004 for
    details. Note typos in equation (1) of their paper. Int.  limits for f(r)
    should be r-->infty and GM(r)/r should be GM(r)/r**2'''

    rmin  = r[0]
    rmax  = r[-1]
    sigp2 = empty(len(r), 'double')

    theta = linspace(0,pi/2-1e-6,num=intpnts)
    cth = cos(theta)
    cth2 = cth**2
    sth = sin(theta)

    for i in xrange(len(r)):
        lower = r[i]
        rint = lower/cth
        rhostar = light.den(rint)
        mtot = masstot(rmin,rmax,alphalim,r,rho,mass,
                       R,sigma,massp,rint)
        integrand = rhostar*rint**(2*beta-2)*Gsp*\
                    mtot*gRr(rint,beta,lower)
        sigp2[i] = integrator(integrand*lower*sth/cth2,theta)

    w = light.surf(r) > 0
    sigp2[w] = sigp2[w] * 2/light.surf(r[w])
    sigp = sqrt(sigp2)

    return sigp

def sigpsingle(rin,sigp,light,aperture,integrator):
    '''Reduce the projected velocity disp profile to a single mean dispersion
    value rms averaged over some aperture.  aperture must be passed with the
    same units as rin'''

    ap = argmin(abs(rin-aperture))

    R = rin[:ap]
    sigp2 = sigp[:ap]**2
    IR = light.surf(R)

    return sqrt(integrator(sigp2*IR*R,R)/\
                integrator(IR*R,R))

def nfwintegrand(x,alp,bet):
    return x**(2-alp)*(1+x)**(alp-bet)
def nfwmint(alp,bet,p):
    '''Dimensionless mass integral for generalise NFW profile'''
    return quad(nfwintegrand, 0, p, args=(alp,bet))[0]

def findrs(c,delta,z,omegam,omegal,h,alp,bet,renc,Mrenc,rmin,rmax):
    '''Find the scale radius of the generalised NFW profile
    given an input concentration parameter c=r/rdelta and
    mass within some radius Mrenc(renc)'''

    rhocrit = 277.3*h**2*(omegam*(1+z)**3+omegal)
    c1 = Mrenc*nfwmint(alp,bet,c)
    c2 = delta*rhocrit*4/3*pi*c**3
    
    #print 'rhocrit = ', rhocrit, renc, Mrenc, rmin, rmax
    #r = linspace(rmin,rmax,num=1000)
    #func = zeros(len(r),'double')
    #for i in xrange(len(r)):
    #    func[i] = abs(c1-c2*r[i]**3*nfwmint(alp,renc/r[i]))
    #figure()
    #plot(log10(r),log10(func))
    #show()
    #sys.exit(0)

    return bisect(lambda x: c1-c2*x**3*nfwmint(alp,bet,renc/x),rmin,rmax)

def drhonfw(c,delta,z,omegam,omegal,h,alp,bet,renc,Mrenc,rmin,rmax,r):
    '''Calculates dlnrhodlnr for a generalised NFW profile'''

    rs = findrs(c,delta,z,omegam,omegal,h,alp,bet,renc,Mrenc,rmin,rmax)

    print 'rs found:', rs, c, alp, bet

    rho = (r/rs)**(-alp)*(1+r/rs)**(alp-bet)

    lnr = log(r)
    lnrho = log(rho)

    return derivative(lambda x: interp(x,lnr,lnrho,left=0,right=0),lnr)

#-----------------------------------------------------------------------------
# Main program
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import *

    #-------------------------------------------------------------------------
    # Parameters
    #-------------------------------------------------------------------------

    #Integrator options [simps/trapz] + number of points to use:
    integrator = simps
    intpnts = 99
    interpnts = None 
    
    #Which test suite to use [0,1,2,3 ...]:
    testsuite = 1

    #Input files [sigma(R) and mass(R)]:
    datadir = '../Data/'
    if testsuite == 0:
        #Range of believeable data + outer slope limiter:
        imagemin = 1
        imagemax = 20
        alphalim = 3

        #Light distribution parameters + vel anisotropy: 
        import massmodel.hernquist
        lpars = [1,15,1,intpnts]
        light = massmodel.hernquist.Hernquist(lpars)
        beta = 0
        aperture = 30

        surffile = datadir+'surftest.txt'
        massfile = datadir+'masstesthi.txt'
    
        #Files containing correct answers:
        trueans = 1
        surffile_true = datadir+'surftest.txt'
        rhofile_true = datadir+'rhotest.txt'
        massfile_true = datadir+'massthreetest.txt'
        sigfile_true = datadir+'sigtest.txt'

    if testsuite == 1:
        #Range of believeable data + outer slope limiter:
        imagemin = 1.62833449387 
        imagemax = 140.697487422
        alphalim = 2

        #Light distribution parameters + vel anisotropy: 
        arctokpc = 7.71687843482
        import massmodel.datafile 
        lpars = [0,arctokpc,1,1,intpnts]
        light = massmodel.datafile.DataFileModel(None,datadir+'jj_circular.dat',lpars)
        beta = 0
        aperture = 0.4*arctokpc

        surffile = datadir+'surfpixtest.txt'
        massfile = datadir+'masspixtest.txt'
        trueans = 0

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

    if testsuite == 0:
        # Interpolate mass file to surface density file. This bit is *only* for
        # code testing. It ensures an accurate value for the enclosed mass (as
        # should be the case also for real lensing data):
        massp = interp(f1['R'],f2['R'],f2['mass'])
    else:
        massp = f2['mass']
    
    #-------------------------------------------------------------------------
    # Calculate the Abel integral to obtain rho(r) and mass(r)
    #-------------------------------------------------------------------------
    if not interpnts: interpnts = len(f1['R'])*2
    r = logspace(log10(amin(f1['R'])/10),log10(amax(f1['R'])*10),
                 num=interpnts)
    print 'Using sample points in r:', interpnts
    print 'Over the range:', amin(f1['R'])/10, amax(f1['R'])*10

    # This package contains two different abel solving routines.  abelsolve()
    # calculates rho(r) first and then integrates to get M(r). This involves
    # taking a numerical derivative of the surface density. cumsolve()
    # calculates M(r) directly via an integral and then differentiates this to
    # obtain rho(r). The ordering in cumsolve() appears to give much better
    # results. 
    rhoa, massa = abelsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
                            f1['R'],f1['sigma'],massp)
    rho, mass = cumsolve(r,imagemin,imagemax,integrator,intpnts,alphalim,
                         f1['R'],f1['sigma'],massp)

    # Now we can calculate dlnrhodlnr to non-parametrically determine how the
    # density profile power law exponent varies with radius.  This is a useful
    # quantity to compare with simulations. 
    drhoa = dlnrhodlnr(imagemin,imagemax,alphalim,r,rhoa,massa,
                       f1['R'],f1['sigma'],massp,r)
    drho = dlnrhodlnr(imagemin,imagemax,alphalim,r,rho,mass,
                      f1['R'],f1['sigma'],massp,r)

    #-------------------------------------------------------------------------
    # Calculate the integral to obtain sigp(r)
    #-------------------------------------------------------------------------
    #units of M=Msun, L=kpc, V=km/s:
    Gsp = 6.67e-11 * 1.989e30 / 3.086e19
    sigpa = sigpsolve(r,rhoa,massa, f1['R'],f1['sigma'],massp,
                      integrator,intpnts,alphalim,Gsp,
                      light,beta)/1000
    sigp = sigpsolve(r,rho,mass, f1['R'],f1['sigma'],massp,
                     integrator,intpnts,alphalim,Gsp,
                     light,beta)/1000
    sigpsinga = sigpsingle(r,sigpa,light,aperture,integrator)
    sigpsing = sigpsingle(r,sigp,light,aperture,integrator)

    print 'Final rms mean projected vel. dispersion [abelsolve()]:',\
          sigpsinga
    print 'Final rms mean projected vel. dispersion [cumsolve()]:',\
          sigpsing

    #-------------------------------------------------------------------------
    # Plot the results: Surface density profile
    #-------------------------------------------------------------------------
    if trueans == 1:
        f2 = loadtxt(surffile_true,
                     dtype = {'names': ('R', 'sigma'),
                              'formats': ('f8', 'f8')})

    # Test the special interpolant:
    rinterp = logspace(-2,4,num=5000)
    sigint = sigmaint(imagemin,imagemax,alphalim,f1['R'],
                      f1['sigma'],massp,rinterp)

    figure()
    loglog(f1['R'], f1['sigma'],label='Python input')
    if trueans == 1: plot(f2['R'], f2['sigma'],label='IDL input')
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
    if trueans == 1:
        f2 = loadtxt(rhofile_true,
                     dtype = {'names': ('r', 'rho'),
                              'formats': ('f8', 'f8')})
    figure()
    loglog(r,rhoa,label='Python result from abelsolve()')
    loglog(r,rho,label='Python result from cumsolve()')
    if trueans == 1: plot(f2['r'],f2['rho'],label='Right result')
    axvline(imagemax)
    axvline(imagemin)
    #plot([imagemax,imagemax],[1e-2,amax(rho)])
    #plot([imagemin,imagemin],[1e-2,amax(rho)])
    gca().set_ylim(5e6,5e9)
    gca().set_xlim(0,imagemax*10)
    title('Deprojected data')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$\rho(r)$')
    legend()
    #savefig(outdir+'rho.pdf')
    
    #-------------------------------------------------------------------------
    # Plot the results: dlnrhodlnr
    #-------------------------------------------------------------------------

    # NFW overlay parameters:
    delta = 200
    omegam = 0.3
    omegal = 0.7
    z = 0
    h = 0.7
    alp = [1,1.5]
    bet = 3
    c = [5,10,20]    
    #renc = imagemax
    renc = 1000
    t = argmin(abs(r-imagemax))
    Mrenc = mass[t]
    print 'NFW overlay parameters:', renc, Mrenc

    figure()
    semilogx(r,drhoa,label='Python result from abelsolve()')
    plot(r,drho,label='Python result from cumsolve()')
    plot([imagemax,imagemax],[amin(drho),amax(drho)])
    plot([imagemin,imagemin],[amin(drho),amax(drho)])

    # Plot NFW overlay:
    for i in xrange(len(alp)):
        for j in xrange(len(c)):
            labelstr = 'alpha=%f, c=%f' % (alp[i],c[j])
            drhoover = drhonfw(c[j],delta,z,omegam,omegal,h,
                               alp[i],bet,renc,Mrenc,5*amin(r),amax(rinterp),
                               rinterp)
            plot(rinterp,drhoover,'--',label=labelstr)

    gca().set_xlim(imagemin/10,imagemax*10)
    gca().set_ylim(-3.5,-0.5)
    title('Power law exponent of density')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$d\ln\rho/d\ln r$')
    legend()
    #savefig(outdir+'rho.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: Mass(r)
    #-------------------------------------------------------------------------
    if trueans == 1:
        f2 = loadtxt(massfile_true,
                     dtype = {'names': ('r', 'mass'),
                              'formats': ('f8', 'f8')})

    # Test the special interpolant:
    mpint = masspint(imagemin,imagemax,alphalim,f1['R'],
                     f1['sigma'],massp,rinterp)
    mtinta = masstot(imagemin,imagemax,alphalim,r,rhoa,massa,f1['R'],
                     f1['sigma'],massp,rinterp)
    mtint = masstot(imagemin,imagemax,alphalim,r,rho,mass,f1['R'],
                    f1['sigma'],massp,rinterp)
    
    figure()
    loglog(r,massa,label='Python result from abelsolve()')
    plot(r,mass,label='Python result from cumsolve()')
    if trueans == 1: plot(f2['r'],f2['mass'],label='Right result')
    plot(f1['R'],massp,label='Projected cumulative mass')
    plot(rinterp,mpint,label='Projected interpolant')
    plot(rinterp,mtinta,label='3D interpolant from abelsolve()')
    plot(rinterp,mtint,label='3D interpolant from cumsolve()')
    plot([imagemax,imagemax],[amin(mass),amax(mass)])
    plot([imagemin,imagemin],[amin(mass),amax(mass)])
    gca().set_xlim(imagemin/10,imagemax*10)
    gca().set_ylim(amin(mass),amax(mass))
    title('Cumulative mass')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$M(r)$')
    legend()
    #savefig(outdir+'mass.pdf')

    #-------------------------------------------------------------------------
    # Plot the results: sigp(r)
    #-------------------------------------------------------------------------
    if trueans == 1:
        f2 = loadtxt(sigfile_true,
                     dtype = {'names': ('r', 'sigp'),
                              'formats': ('f8', 'f8')})
    figure()
    semilogx(r,sigpa,label='Python result from abelsolve()')
    plot(r,sigp,label='Python result from cumsolve()')
    if trueans == 1: plot(f2['r'],f2['sigp'],label='Right result')
    plot([imagemax,imagemax],[amin(sigp),amax(sigp)])
    plot([imagemin,imagemin],[amin(sigp),amax(sigp)])
    gca().set_xlim(imagemin/10,imagemax*10)
    title('Projected velocity dispersion')
    xlabel(r'$r(\mathrm{kpc})$')
    ylabel(r'$\sigma_p(r)$')
    legend()
    #savefig(outdir+'sigp.pdf')
    show()

    



