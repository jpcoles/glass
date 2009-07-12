from __future__ import division
from numpy import array, empty_like, amin, amax, repeat, logspace, arange, ptp
from numpy import put, empty, zeros, ogrid, mgrid, atleast_2d, linspace, meshgrid, log10, log
from numpy.random import random
from pylab import show, imshow, contour, gca, scatter, xlabel, ylabel, plot, loglog, \
                  hist, hold, colorbar, legend, over, axvline, matshow, gcf, subplot, \
                  suptitle, figure, grid, gray, jet, semilogx
import matplotlib
import matplotlib.cm as cm  
from matplotlib import rc
from matplotlib.ticker import LogLocator
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scales import density_to_physical, distance_to_physical

rc('text', usetex=True)

_styles = [{'label':r'rejected', 'c':'r', 'ls':'-', 'z':-1, 'line':Line2D([],[],c='r',ls='-')},
           {'label':r'accepted', 'c':'b', 'ls':'-', 'z': 0, 'line':Line2D([],[],c='b',ls='-')},
           {'label':r'unknown',  'c':'k', 'ls':'-', 'z':+1, 'line':Line2D([],[],c='k',ls='-')}]

_system_colors = 'rgbcmykw'
_source_colors = 'c'

def system_color(i): return _system_colors[i%len(_system_colors)]
def source_color(i): return _source_colors[i%len(_source_colors)]

def img_plot(model, src_index=None):
    if src_index is not None and not isinstance(src_index, (list,tuple)):
        src_index = [src_index]

    if isinstance(model, (list, tuple)):
        obj, data = model
    else:
        obj = model

    for i,src in enumerate(obj.sources):
        if src_index is not None and i not in src_index: continue
        xs = []
        ys = []
        for img in src.images:
            xs.append(img.pos.real)
            ys.append(img.pos.imag)
        if xs and ys:
            over(scatter,xs, ys, s=80, c=source_color(i), zorder=1000)

def src_plot(model):
    obj, data = model
    for i,sys in enumerate(obj.sources):
        xs = []
        ys = []
        xs.append(data['src'][i*2+0])
        ys.append(data['src'][i*2+1])
        print 'src_plot:', xs, ys
        over(scatter,xs, ys, s=80, c=source_color(i), zorder=1000)


def mass_plot(model):
    obj, data = model

    L = obj.basis.pixrad
    R = obj.basis.mapextent
    S  = obj.basis.subdivision


    # XXX: Remove this circle. Shouldn't know about cell_size.
    circle0 = Circle((0,0), R)
    circle1 = Circle((0,0), R - obj.basis.cell_size/S/2)
    circle0.set_fill(False)
    circle1.set_fill(False)

    grid = obj.basis.mass_grid(data)
    #grid = random(grid.shape)

    #R += obj.basis.cell_size / 2

    #a = subplot(111, aspect='equal')
    matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
    #over(contour, grid, 50, extent=[-R,R,-R,R], extend='both')
    #over(contour, grid, colors='w', extent=[-R,R,-R,R], origin='upper', extend='both')
    #colorbar()
    #a.add_artist(circle0)
    #a.add_artist(circle1)
    #img_plot(obj)
    xlabel('arcsec')
    ylabel('arcsec')

def potential_plot(model, sys):
    obj, data = model
    R = obj.basis.mapextent
    grid = obj.basis.potential_grid(data)
    levs = obj.basis.potential_contour_levels(data)
#   matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
#   contour(grid, extent=[-R,R,-R,R], origin='upper')
    matshow(grid, fignum=False, cmap=cm.gray, extent=[-R,R,-R,R], interpolation='nearest')
    #print levs
    for i,lev in enumerate(levs):
        over(contour, grid, lev, colors = system_color(i), 
             extent=[-R,R,-R,R], origin='upper', extend='both')

    xlabel('arcsec')
    ylabel('arcsec')
    suptitle('Potential')

def arrival_plot(model, src_index):
    obj, data = model
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g   = obj.basis.arrival_grid(data)[src_index]
    lev = obj.basis.arrival_contour_levels(data)
    if lev: lev = lev[src_index]

#   figure()
#   hist(g.flatten())

#   figure()
    #matshow(g, fignum=False, cmap=cm.gray, extent=[-R,R,-R,R], interpolation='nearest')
    #lev = 50 if not lev else lev[src_index]
    print 'arrival_plot:', lev
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    over(contour, g, 50,  colors='w',               linewidths=1, 
         extent=[-R,R,-R,R], origin='upper', extend='both')
    if lev:
        print '***', lev, '***'
        over(contour, g, lev, colors=system_color(src_index), linewidths=3, 
             extent=[-R,R,-R,R], origin='upper')
    grid()

def srcdiff_plot(model, src_index):
    obj, data = model
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g   = obj.basis.srcdiff_grid(data)[src_index]

    matshow(g, fignum=False, cmap=cm.jet, extent=[-R,R,-R,R], interpolation='nearest')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    over(contour, g, 50,  colors='w',               linewidths=1, 
         extent=[-R,R,-R,R], origin='upper', extend='both')
    grid()

def _data_plot(models, X,Y, x_label, y_label, plotf=loglog, mark_images=False):
    with_legend = False
    use = [0,0,0]
    for m in models:
        for [obj, data] in m['obj,data']:

            si = m.get('accepted', 2)
            use[si] = 1

            s = _styles[si]
            plotf(data[X],data[Y], c=s['c'], ls=s['ls'], zorder=s['z'])

            if mark_images:
                for i,src in enumerate(obj.sources):
                    for img in src.images:
                        x = distance_to_physical([obj, data], abs(img.pos))
                        axvline(x, c=system_color(i), ls='-', zorder=-2, alpha=0.5)

    if use[0] or use[1]:
        lines = [s['line']  for s,u in zip(_styles, use) if u]
        labels = [s['label'] for s,u in zip(_styles, use) if u]
        legend(lines, labels)

    xlabel(x_label)
    ylabel(y_label)

_kappa_xlabel = r'$R$ $(\mathrm{kpc})$'
_kappa_ylabel = r'$\kappa$ $(M_\odot/\mathrm{kpc}^2)$'
def kappa_plot(models, **kwargs):
    _data_plot(models, 'R', 'enckappa', _kappa_xlabel, _kappa_ylabel, **kwargs)

_sigma_xlabel = r'$R$ $(\mathrm{kpc})$'
_sigma_ylabel = r'$\Sigma$ $(M_\odot/\mathrm{kpc}^2)$'
def sigma_plot(models, **kwargs):
    _data_plot(models, 'R_kpc', 'sigma', _sigma_xlabel, _sigma_ylabel, **kwargs)

_sigp_xlabel = r'$R$ $(\mathrm{kpc})$'
_sigp_ylabel = r'$\sigma_p$ $()$'
def sigp_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:sigp', _sigp_xlabel, _sigp_ylabel, plotf=semilogx, **kwargs)
    #_data_plot(models, 'sigp:R', 'sigp:sigp', _sigp_xlabel, _sigp_ylabel, kwargs, plotf=plot)

_mass3d_xlabel = r'$r$ $(\mathrm{kpc})$'
_mass3d_ylabel = r'$M$'
def mass3d_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:mass3d', _mass3d_xlabel, _mass3d_ylabel, plotf=loglog, **kwargs)

_rho_xlabel = r'$r$ $(\mathrm{kpc})$'
_rho_ylabel = r'$\rho$ $()$'
def rho_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:rho', _rho_xlabel, _rho_ylabel, **kwargs)

_rhoint_xlabel = r'$r$ $(\mathrm{kpc})$'
_rhoint_ylabel = r'$\rho$ $()$'
def rhoint_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:rhoint', _rhoint_xlabel, _rhoint_ylabel, **kwargs)

_drho_xlabel = r'$r$ $(\mathrm{kpc})$'
_drho_ylabel = r'$d\ln\rho/d\ln r$'
def drho_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:drho', _drho_xlabel, _drho_ylabel, plotf=semilogx, **kwargs)

_rhoa_xlabel = r'$r$ $(\mathrm{kpc})$'
_rhoa_ylabel = r'$\rho_\mathrm{abel}$ $()$'
def rhoa_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:rhoa', _rho_xlabel, _rho_ylabel, **kwargs)

_drhoa_xlabel = r'$r$ $(\mathrm{kpc})$'
_drhoa_ylabel = r'$d\ln\rho_\mathrm{abel}/d\ln r$'
def drhoa_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:drhoa', _drho_xlabel, _drho_ylabel, plotf=semilogx, **kwargs)

_encmass_xlabel = r'$R$ $(\mathrm{kpc})$'
_encmass_ylabel = r'$M$'
def encmass_plot(models, **kwargs):
    _data_plot(models, 'R_kpc', 'encmass', _encmass_xlabel, _encmass_ylabel, plotf=loglog, **kwargs)


_H0_xlabel = r'$H_0^{-1}$ (Gyr)'
def H0_plot(models, objects=None, key='accepted'):

    # select a list to append to based on the 'accepted' property.
    l = [[], [], []]
    for m in models:
        for [obj, data] in m['obj,data']:
            l[m.get(key,2)].append(data['1/H0'])

    not_accepted, accepted, notag = l

    #print 'H0_plot',H0s

    for d,s in zip(l, _styles):
        if d:
            hist(d, bins=ptp(d)//2+1, histtype='step', edgecolor=s['c'], zorder=s['z'], label=s['label'])

    if not_accepted or accepted:
        legend()

    axvline(13.7, c='k', ls=':', zorder = 2)

    xlabel(_H0_xlabel)
    ylabel(r'Number')

