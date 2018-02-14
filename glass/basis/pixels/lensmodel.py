import numpy as np
from glass.lensmodel import LensModel, prop
from glass.scales import convert
from glass.environment import DArray
from . funcs import estimated_Rlens, arrival_time

def memoize(func):
    def f(self, *args, **kwargs):
        #print func.__name__, data.has_key(func.__name__), data.keys()
        if not self.d.has_key(func.__name__): 
            self.d[func.__name__] = func(self, *args, **kwargs)
        return self.d[func.__name__]
    return f

class PixelLensModel(LensModel):

    def __init__(self, obj, sol):
        LensModel.__init__(self, obj)
        #self.sol = sol

    def mean_kappa(self,k,x):
        if not isinstance(k, (type(0), type(0.0))):
            return np.sum(k[x] * self.obj.basis.cell_size[x]**2) / np.sum(self.obj.basis.cell_size[x]**2)
        else:
            return k

    @prop('kappa star')
    def kappa_star(self):
        stellar_mass = self.obj.stellar_mass if hasattr(self.obj, 'stellar_mass') else 0
        return stellar_mass * self['sm_error_factor']

    @prop('kappa')
    def kappa(self):
        return self['kappa DM'] + self['kappa star']

    @prop('kappa(R)')
    def kappa_R(self, component=[]):
        k = ' '.join(['kappa'] + component)
        return DArray([self.mean_kappa(self[k],r) for r in self.obj.basis.rings],
                       r'$\langle\kappa(R)\rangle$', {'$\kappa$': [1, None]})

    @prop('M(<R)')
    def M_ltR(self, component=[]):
        k = ' '.join(['kappa'] + component)
        dscale1 = convert('kappa to Msun/arcsec^2', 1, self.obj.dL, self['nu'])
        b = self.obj.basis
        return DArray(np.cumsum([np.sum(self[k][r]*b.cell_size[r]**2)*dscale1 for r in b.rings]),
                      r'$M(<R)$', {'Msun': [1, r'$M_\odot$']})

    @prop('Sigma(R)')
    def Sigma_R(self, component=[]):
        k = ' '.join(['kappa'] + component)
        dscale2 = convert('kappa to Msun/kpc^2',    1, self.obj.dL, self['nu'])
        return DArray([self.mean_kappa(self[k],r)*dscale2 for r in self.obj.basis.rings],
                       r'$\Sigma$', {'Msun/kpc^2': [1, r'$M_\odot/\mathrm{kpc}^2$']})
    @prop('kappa(<R)')
    def kappa_ltR(self, component=[]):
        k = ' '.join(['kappa'] + component)
        M = np.cumsum([np.sum(self[k][r]*self.obj.basis.cell_size[r]**2) for r in self.obj.basis.rings]) 
        V = np.cumsum([np.sum(self.obj.basis.cell_size[r]**2) for r in self.obj.basis.rings])
        return DArray(M/V, r'$\kappa(<R)$', {'kappa': [1, None]})

    @prop('R')
    def R(self):
        rscale = convert('arcsec to kpc', 1, self.obj.dL, self['nu'])
        return DArray(self.obj.basis.rs + self.obj.basis.radial_cell_size / 2,
                      r'$R$', {'arcsec': [1, r'$\mathrm{arcsec}$'],
                               'kpc':    [rscale, r'$\mathrm{kpc}$']})
    @prop('H0')
    def H0(self):
        return convert('nu to H0 in km/s/Mpc', self['nu'])

    @prop('1/H0')
    def H0inv(self):
        return convert('nu to H0^-1 in Gyr',   self['nu'])

    @prop('Rlens')
    def Rlens(self):
        rscale = convert('arcsec to kpc', 1, self.obj.dL, self['nu'])
        return DArray([estimated_Rlens(self.obj,self,i)[0] for i,src in enumerate(self.obj.sources)],
                          r'$R_e$', {'arcsec': [1, r'$\mathrm{arcsec}$'],
                                     'kpc':    [rscale, '$\mathrm{kpc}$']})

    @prop('Ktot')
    def Ktot(self):
        return np.sum(self['kappa'])

    @prop('arrival times')
    def arrival_times(self):
        return arrival_time((self.obj,self))

    @prop('time delays')
    def time_delays(self):
        D = []
        at = self['arrival times']
        for src_index,src in enumerate(self.obj.sources):
            d = []
            if at[src_index]:
                t0 = at[src_index][0]
                for i,t in enumerate(at[src_index][1:]):
                    d.append( float('%0.6f'%convert('arcsec^2 to days', t-t0, self.obj.dL, self.obj.z, self['nu'])) )
                    t0 = t
            D.append(d)
        return D

