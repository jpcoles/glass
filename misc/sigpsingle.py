import sys
sys.path.append('..')

from spherical_deproject import sigpsingle
from numpy import loadtxt


files = sys.argv[1:]

if not files:
    dir  = '/smaug/data/theorie/justin/Backup/Mylaptop/Scratch/Lensing/Cuspcore/CMerger1'
    files.append(dir + '/cmerger_1_sigpx.txt')

for f in files:
    data = loadtxt(f,
                   dtype = {'names': ('R', 'sigp', 'err'),
                            'formats': ('f8', 'f8', 'f8')})

    import massmodel.hernquist as light
    from scipy.integrate.quadrature import simps as integrator

    intpnts = 100 
    lpars = [1,25,1,intpnts]
    beta = 0
    aperture = 400

    print sigpsingle(data['R'],data['sigp'],light,lpars,aperture,integrator)

