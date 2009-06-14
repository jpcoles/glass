from __future__ import division
from numpy import array, empty_like, amin, amax, repeat, logspace
from numpy import put, empty, zeros, ogrid, mgrid, atleast_2d, linspace, meshgrid, log10, log
from numpy.random import random
from pylab import show, imshow, contour, gca, scatter, xlabel, ylabel, plot, loglog, \
                  hist, hold, colorbar, legend, over, axvline, matshow, gcf, subplot, \
                  suptitle, figure, grid, gray, jet
import matplotlib
import matplotlib.cm as cm  
from matplotlib.ticker import LogLocator
from matplotlib.patches import Circle
from scales import density_to_physical, distance_to_physical

_system_colors = 'rgbcmykw'
_source_colors = 'c'

def system_color(i): return _system_colors[i%len(_system_colors)]
def source_color(i): return _source_colors[i%len(_source_colors)]

def img_plot(model):
    obj, data = model
    for i,sys in enumerate(obj.systems):
        xs = []
        ys = []
        for img in sys.images:
            xs.append(img.pos.real)
            ys.append(img.pos.imag)
        over(scatter,xs, ys, s=80, c=system_color(i), zorder=1000)

def src_plot(model):
    obj, data = model
    for i,sys in enumerate(obj.systems):
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
    over(contour, grid, colors='w', extent=[-R,R,-R,R], origin='upper', extend='both')
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
    print levs
    for i,lev in enumerate(levs):
        over(contour, grid, lev, colors = system_color(i), 
             extent=[-R,R,-R,R], origin='upper', extend='both')

    xlabel('arcsec')
    ylabel('arcsec')
    suptitle('Potential')

def arrival_plot(model, sys):
    obj, data = model
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g   = obj.basis.arrival_grid(data)[sys]
    lev = obj.basis.arrival_contour_levels(data)
    if lev: lev = lev[sys]

#   figure()
#   hist(g.flatten())

#   figure()
    matshow(g, fignum=False, cmap=cm.gray, extent=[-R,R,-R,R], interpolation='nearest')
    #lev = 50 if not lev else lev[sys]
    print 'arrival_plot:', lev
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    over(contour, g, 50,  colors='w',               linewidths=1, 
         extent=[-R,R,-R,R], origin='upper', extend='both')
    if lev:
        print '***', lev, '***'
        over(contour, g, lev, colors=system_color(sys), linewidths=3, 
             extent=[-R,R,-R,R], origin='upper')
    grid()

def _data_plot(models, X,Y, x_label, y_label, plotf=loglog):
    for m in models:
        for [obj, data] in m['objs']:
            if m.has_key('tagged') and not m['tagged']:
                plotf(data[X],data[Y], 'r-', alpha=0.5)
            else:
                plotf(data[X],data[Y], 'b-')

            for i,sys in enumerate(obj.systems):
                for img in sys.images:
                    x = distance_to_physical([obj, data], abs(img.pos))
                    print x
                    axvline(x, c=system_color(i), ls=':')
    xlabel(x_label)
    ylabel(y_label)

_sigma_xlabel = r'$R$ $(\mathrm{kpc})$'
_sigma_ylabel = r'$\Sigma$ $(M_\odot/\mathrm{kpc}^2)$'
def sigma_plot(models):
    _data_plot(models, 'R_phys', 'sigma_phys', _sigma_xlabel, _sigma_ylabel)

_sigp_xlabel = r'$R$ $(\mathrm{kpc})$'
_sigp_ylabel = r'$\sigma_p^2$ $()$'
def sigp_plot(models):
    _data_plot(models, 'sigp:R', 'sigp:sigp', _sigp_xlabel, _sigp_ylabel, plotf=plot)

_rho_xlabel = r'$R$ $(\mathrm{kpc})$'
_rho_ylabel = r'$\rho$ $()$'
def rho_plot(models):
    _data_plot(models, 'sigp:R', 'sigp:rho', _rho_xlabel, _rho_ylabel)

_encmass_xlabel = r'$R$ $(\mathrm{kpc})$'
_encmass_ylabel = r'$M$'
def encmass_plot(models):
    _data_plot(models, 'R_phys', 'encmass_phys', _encmass_xlabel, _encmass_ylabel)


_H0_xlabel = r'$H_0^{-1}$ (Gyr)'
def H0_plot(models, objects=None):

    H0s    = [ data['1/H0'] for m in models if not m['tagged'] for [obj, data] in m['objs']]
    tagH0s = [ data['1/H0'] for m in models if     m['tagged'] for [obj, data] in m['objs']]

    print 'H0_plot',H0s

    if H0s:
        b = logspace(log10(min(H0s)*0.9), log10(max(H0s)*1.1), 10)
        hist(H0s, bins=b, histtype='step', edgecolor='black')
        axvline(14, c='r', ls=':')

    if tagH0s:
        b = logspace(log10(min(tagH0s)*0.9), log10(max(tagH0s)*1.1), 10)
        over(hist, tagH0s, bins=b, histtype='step', edgecolor='red')
        axvline(14, c='r', ls=':')

    xlabel(_H0_xlabel)
    ylabel('Number')

