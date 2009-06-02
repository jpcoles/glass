from __future__ import division
from numpy import array, empty_like, amin, amax, repeat
from numpy import put, empty, zeros, ogrid, mgrid, atleast_2d, linspace, meshgrid, log10, log
from numpy.random import random
from pylab import show, imshow, contour, gca, scatter, xlabel, ylabel, plot, loglog, hist, hold, colorbar, legend, over, axvline, matshow, gcf, subplot, suptitle, figure, grid
from pylab import gray, jet
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.patches import Circle
import matplotlib.cm as cm  

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
    R = obj.basis.maprad
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
    R = obj.basis.maprad
    grid = obj.basis.potential_grid(data)
    lev = obj.basis.potential_contour_levels(data)
#   matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
#   contour(grid, extent=[-R,R,-R,R], origin='upper')
    matshow(grid, fignum=False, cmap=cm.gray, extent=[-R,R,-R,R], interpolation='nearest')
    print lev
    for i,sys in enumerate(obj.systems):
        over(contour, grid, lev[i], colors = system_color(i), 
             extent=[-R,R,-R,R], origin='upper', extend='both')

    xlabel('arcsec')
    ylabel('arcsec')
    suptitle('Potential')

def arrival_plot(model, sys):
    obj, data = model
    S = obj.basis.subdivision
    R = obj.basis.maprad
    #R -= obj.basis.cell_size / 2

    g   = obj.basis.arrival_grid(data)
    lev = obj.basis.arrival_contour_levels(data)

    g = g[sys]
#   figure()
#   hist(g.flatten())

#   figure()
    matshow(g, fignum=False, cmap=cm.gray, extent=[-R,R,-R,R], interpolation='nearest')
    lev = 50 if not lev else lev[sys]
    print 'arrival_plot:', lev
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    over(contour, g, 20,  colors='w',               linewidths=1, 
         extent=[-R,R,-R,R], origin='upper', extend='both')
    over(contour, g, lev, colors=system_color(sys), linewidths=3, 
         extent=[-R,R,-R,R], origin='upper')
    grid()

_sigma_ylabel = r'$\Sigma$'
def sigma_plot(models):
    for [sol, objs] in models:
        for [obj, data] in objs:
            over(loglog, data['R'], data['sigma'])

    #loglog(data['R'], data['sigma'])
    #gca().set_xlim(0,data['R'][-1])
    xlabel('arcsec')
    ylabel(_sigma_ylabel)

_encmass_ylabel = r'$M$'
def encmass_plot(models):

    for [sol, objs] in models:
        for [obj, data] in objs:
            over(loglog, data['R'], data['encmass'])

    #gca().set_xlim(0,data['R'][-1])
    xlabel('arcsec')
    ylabel(_encmass_ylabel)

_H0_xlabel = r'$H_0^{-1}$ (Gyr)'
def H0_plot(models, objects=None):

    H0s = [ data['1/H0'] for [sol, objs] in models for [obj, data] in objs ]
            
    hist(log10(H0s), bins=100, histtype='step')
    axvline(log10(14), c='r', ls=':')

    #gca().set_xlim(0, 20)
    #gca().set_xscale('log', basex=10)
    xlabel(_H0_xlabel)
    ylabel('Number')
    return 

