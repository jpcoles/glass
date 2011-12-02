from __future__ import division
from numpy import array, mat, empty_like, amin, amax, repeat, logspace, arange, \
                  ptp, amin, amax, sqrt, sort, abs, put, empty, zeros, ogrid, \
                  mgrid, atleast_2d, linspace, meshgrid, log10, log, diff, ravel, \
                  meshgrid, vstack, argsort, logical_and, inf
from numpy.random import random
from pylab import show, imshow, contour, gca, scatter, xlabel, ylabel, plot, loglog, \
                  hist, hold, colorbar, legend, over, axvline, matshow, gcf, subplot, \
                  suptitle, figure, grid, gray, semilogx, semilogy, imread, imshow, errorbar, \
                  text, xlim, ylim
import matplotlib
import matplotlib.cm as cm  
from matplotlib import rc
from matplotlib.ticker import LogLocator
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D
from scales import density_to_physical, distance_to_physical, Arcsec_to_Kpc, convert
import math
from collections import defaultdict

#import matplotlib.axes3d as p3
import mpl_toolkits.mplot3d as p3

from environment import env, Object, command


from scipy.ndimage.filters import correlate1d
from scipy.misc import central_diff_weights

rc('text', usetex=True)
rc('text', dvipnghack=True)

_styles = [{'label':r'rejected', 'c':'r', 'ls':'-', 'z':-1, 'line':Line2D([],[],c='r',ls='-')},
           {'label':r'accepted', 'c':'b', 'ls':'-', 'z': 0, 'line':Line2D([],[],c='b',ls='-')},
           {'label':r'unknown',  'c':'k', 'ls':'-', 'z':+1, 'line':Line2D([],[],c='k',ls='-')}]

_system_colors = 'gbmykw'
_source_colors = 'c'

def system_color(i): return _system_colors[i%len(_system_colors)]
def source_color(i): return _source_colors[i%len(_source_colors)]

@command
def show_plots():
    show()

@command
def img_plot(model, obj_index=0, src_index=None, with_maximum=True, color=None, with_guide=False):
    if src_index is not None and not isinstance(src_index, (list,tuple)):
        src_index = [src_index]

    obj,_ = model['obj,data'][obj_index]

#   if isinstance(model, (list, tuple)):
#       obj,_ = model
#   else:
#       obj = model

    for i,src in enumerate(obj.sources):
        if src_index is not None and i not in src_index: continue
        xs,ys,cs = [], [], []

        for img in src.images:
            #print img.pos
            if not with_maximum and img.parity_name == 'max': continue

            xs.append(img.pos.real)
            ys.append(img.pos.imag)
            if not color:
                if img.parity_name == 'unk':
                    cs.append('red')
                else:
                    cs.append(system_color(i))
            else:
                cs.append(color)

        if xs and ys:
            over(scatter,xs, ys, s=80, c=cs, zorder=1000, alpha=0.5)
            if with_guide:
                a = gca()
                for x,y in zip(xs,ys):
                    a.add_artist(Circle((0,0),sqrt(x**2 + y**2), fill=False))

@command
def Re_plot(models=None, obj_index=0, color=None):

    if models is None:
        models = env().models
    elif not hasattr(models, '__getslice__'):
        models = [models]

    if not color: color = 'k'

    for m in models:
        obj,data = m['obj,data'][obj_index]
        print data.keys()
        if not data['Re']: continue
        Re, a,b, theta = data['Re']
        #gca().add_artist(Circle((rl.real,rl.imag), 0.1, fill=False, lw=2, color='r'))
        #gca().add_artist(Circle((rs.real,rs.imag), 0.1, fill=False, lw=2, color='r'))
        #gca().add_artist(Line2D([0,A[0]], [0,A[1]], lw=2, color=color))
        #gca().add_artist(Line2D([0,B[0]], [0,B[1]], lw=2, color=color))
        #gca().add_artist(Circle((0,0), a, fill=False, lw=2, color=color))
        #gca().add_artist(Circle((0,0), b, fill=False, lw=2, color=color))
        gca().add_artist(Ellipse((0,0), 2*a,2*b, theta, fill=False, lw=2, color=color))
        #gca().add_artist(Circle((0,0), a, fill=False, lw=2, color=color))

@command
def src_plot(models=None, obj_index=0, hilite_model=None, hilite_color='g'):

    if models is None: models = env().models

    def plot(model, hilite=False):
        obj, data = model
        xs = []
        ys = []
        cs = []
        for i,sys in enumerate(obj.sources):
            xs.append(data['src'][i].real)
            ys.append(data['src'][i].imag)
            cs.append(system_color(i))
        if hilite:
            over(scatter,xs, ys, s=80, c=hilite_color, zorder=2000, marker='d', alpha=0.5)
        else:
            over(scatter,xs, ys, s=80, c=cs, zorder=1000, marker='d', alpha=0.5)

    if isinstance(models, dict):
        plot(models['obj,data'][obj_index])
    else:
        for mi,model in enumerate(models):
            for m in model['obj,data']:
                plot(m, mi==hilite_model)

    #if isinstance(models, (list,tuple)) and len(models)>0 and isinstance(models[0], (list,tuple)):
    #else:

_src_hist_xlabel = r'$r$ $(\mathrm{arcsec})$'
_src_hist_ylabel = r'$\mathrm{Number}$'
def src_hist(models=None, hilite_model=None):
    if models is None: models = env().models

    d = []
    hilite=[]
    for mi,model in enumerate(models):
        for [_,data] in model['obj,data']:
            r = list(abs(data['src']))
            d += r
            if mi == hilite_model: hilite += r
    hist(d, histtype='step', log=False)
    for i,r in enumerate(hilite):
        print r
        axvline(r, c=system_color(i), ls='-', zorder=-2, alpha=0.5)

            
    xlabel(_src_hist_xlabel)
    ylabel(_src_hist_ylabel)
    

def image_plot(im, extent):
    R = extent
    imshow(imread(im), extent=[-R,R,-R,R])

#def kappa_avg_plot(models):
#    objs = {} 
#    for m in models:
#        for [obj, data] in m['obj,data']:
#            a = 
#
#    grid

def mass_plot(model, obj_index, with_contours=True, only_contours=False, clevels=30):
    print "WARNING: use of mass_plot is deprecated. Use kappa_plot instead."
    return kappa_plot(model, obj_index, with_contours, only_contours, clevels)

@command
def kappa_plot(model, obj_index, with_contours=False, only_contours=False, clevels=30, with_colorbar=True):
    obj, data = model['obj,data'][obj_index]

    R = obj.basis.mapextent

    grid = obj.basis.kappa_grid(data)
    grid = grid.copy() + 1e-4
#   grid2 = grid.copy() 
#   for i in xrange(grid.shape[0]):
#       for j in xrange(grid.shape[1]):
#           grid[i,j] = abs(grid2[grid.shape[0]-i-1, grid.shape[1]-j-1] - grid[i,j]) / grid[i,j]
#   grid = grid.copy() + 1e-4

    #grid[grid >= 1] = 0

    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.bone,
          'fignum': False,
          'vmin': -2,
          'vmax': 0}

    if not only_contours:
        #matshow(log10(grid), **kw)
        matshow(log10(grid), **kw)
        #imshow(grid, fignum=False, **kw)
        #matshow(grid, fignum=False, **kw)
        if with_colorbar: colorbar()

    if with_contours:
        kw.pop('cmap')
        over(contour, grid, clevels, extend='both', colors='w', alpha=0.7, **kw)

    xlabel('arcsec')
    ylabel('arcsec')

@command
def potential_plot(model, obj_index, src_index, with_colorbar=True, with_contours=False):
    obj, data = model['obj,data'][obj_index]
    R = obj.basis.mapextent
    grid = obj.basis.potential_grid(data)
    levs = obj.basis.potential_contour_levels(data)
#   matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
    matshow(grid, fignum=False, cmap=cm.bone, extent=[-R,R,-R,R], interpolation='nearest')
    if with_colorbar: colorbar()
#   contour(grid, extent=[-R,R,-R,R], origin='upper')
    #print levs
    if with_contours:
        for i,lev in enumerate(levs):
            over(contour, grid, lev, colors = system_color(i), 
                 extent=[-R,R,-R,R], origin='upper', extend='both')


    xlabel('arcsec')
    ylabel('arcsec')
#   figure();
#   xs = linspace(-R, R, grid.shape[0])
#   plot(xs, grid[grid.shape[1]//2, :], 'k-')
#   plot(xs, 5*xs, 'r-')

    #suptitle('Potential')

@command
def critical_curve_plot(model, obj_index, src_index):
    obj, data = model['obj,data'][obj_index]
    R = obj.basis.mapextent
    g = obj.basis.maginv_grid(data)[src_index]
    matshow(g, fignum=False, cmap=cm.bone, extent=[-R,R,-R,R], interpolation='nearest')
    over(contour, g, [0], colors='g', linewidths=1, extent=[-R,R,-R,R], origin='upper')

@command
def arrival_plot(model, obj_index, src_index, only_contours=False, clevels=30, with_colorbar=False):
    obj, data = model['obj,data'][obj_index]
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g   = obj.basis.arrival_grid(data)[src_index]
    lev = obj.basis.arrival_contour_levels(data)
    if lev: lev = lev[src_index]

    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.bone,
          'fignum': False,
          'vmin': -4,
          'vmax': 0}

    if not only_contours:
        matshow(g, **kw)
        if with_colorbar: colorbar()
        #lev = 50 if not lev else lev[src_index]

    print 'arrival_plot:', lev
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    print amin(g), amax(g)
    loglev = logspace(1, log(amax(g)-amin(g)), 20, base=math.e) + amin(g)
    print loglev
    kw.update({'colors':'k', 'linewidths':1, 'cmap':None})
    over(contour, g, 
         #loglev,
         #logspace(amin(log(lev), amax(lev), 50),  
         clevels, #logspace(amin(g), amax(g), 50),  
         **kw)
    if lev:
        print '***', lev, '***'
        kw.update({'colors':system_color(src_index), 'linewidths':3, 'cmap':None})
        over(contour, g, lev, **kw)
    #grid()

@command
def srcdiff_plot(model, obj_index, src_index, with_colorbar=False):
    obj, data = model['obj,data'][obj_index]
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g = obj.basis.srcdiff_grid(data)[src_index]
    vmin = log10(amin(g[g>0]))
    g = g.copy() + 1e-10
    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.gray,
          'fignum': False,
          'vmin': vmin,
          'vmax': vmin+2}

    #loglev = logspace(1, log(amax(g)-amin(g)), 20, base=math.e) + amin(g)
    matshow(log10(g), **kw)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    if with_colorbar: colorbar()
#   over(contour, g, 50,  colors='w',               linewidths=1, 
#        extent=[-R,R,-R,R], origin='upper', extend='both')
    #grid()

    xlabel('arcsec')
    ylabel('arcsec')

@command
def deflect_plot(model, obj_index, which, src_index):
    obj, data = model['obj,data'][obj_index]
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g = obj.basis.deflect_grid(data, which, src_index)

    vmin = log10(amin(g[g>0]))
    g = g.copy() + 1e-10
    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.bone,
          'fignum': False}

    matshow(g, **kw)
    #matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

@command
def grad_tau(model, obj_index, which, src_index):

    assert which in ['x','y'], "grad_tau: 'which' must be one of 'x' or 'y'"

    print "grad_tau"
    obj,ps = model['obj,data'][obj_index]
    R = obj.basis.mapextent

    #---------------------------------------------------------------------------
    # Find the derivative of the arrival time surface.
    #---------------------------------------------------------------------------
    arrival = obj.basis.arrival_grid(ps)[src_index]

    w = central_diff_weights(3)

    which = 1 if which == 'x' else 0
    d = correlate1d(arrival, w, axis=which, mode='constant')

    d = d[1:-1,1:-1]
    d[abs(d) < 1e-3] = 0
    d[d>0] = 1
    d[d<0] = -1
    matshow(d, fignum=False, extent=[-R,R,-R,R], alpha=0.5)

@command
def deriv(model, obj_index, src_index, m, axis, R):
    w = central_diff_weights(5)
    #d = correlate1d(m, w, axis=axis, mode='constant')
    d = (correlate1d(m, -w, axis=0, mode='constant')) \
      + (correlate1d(m,  w, axis=1, mode='constant'))
    d = (correlate1d(d, -w, axis=0, mode='constant')) \
      + (correlate1d(d,  w, axis=1, mode='constant'))
    d = d[2:-2,2:-2]
    d[d>.8] = .8
    d[d<-.8] = -.8
    #d = correlate1d(d, w, axis=axis, mode='constant')
    #d = diff(d, axis=axis)
    #d /= abs(d)
    #d = correlate1d(d, w, axis=axis, mode='constant')
    #d = diff(d, axis=axis)

    R -= model[0].basis.top_level_cell_size * 2
    #R -= model[0].basis.top_level_cell_size * 2
    matshow(d, extent=[-R,R,-R,R])
    colorbar()
    arrival_plot(model, obj_index, src_index, only_contours=True, clevels=200)
    #img_plot(model, src_index=src_index)
    #matshow(d)

#   d = d[1:-1,1:-1]
#   d[d>0] = 1
#   d[d<0] = -1
#   matshow(d, extent=[-R,R,-R,R])
#   img_plot(model, src_index=src_index)

@command
def inout_plot(model, obj_index, src_index):
    print "inout"
    obj,ps = model['obj,data'][obj_index]
    R = obj.basis.mapextent
    arrival = obj.basis.arrival_grid(ps)[src_index]

    deriv(model, obj_index, src_index, arrival, 0, R)
    deriv(model, obj_index, src_index, arrival, 1, R)

def _data_plot(models, X,Y, x_label, y_label, **kwargs):
    with_legend = False
    use = [0,0,0]
    if models is None:
        models = env().models
    elif not hasattr(models, '__getslice__'):
        models = [models]


    every = kwargs.get('every', 1)
    plotf = kwargs.get('plotf', loglog)
    mark_images = kwargs.get('mark_images', True)
    hilite_model = kwargs.get('hilite_model', None)
    hilite_color = kwargs.get('hilite_color', 'y')

    normal_kw   = {'zorder':0, 'drawstyle':'steps-post', 'alpha':0.5}
    hilite_kw   = {'zorder':1000, 'drawstyle':'steps-post', 'alpha':1.0, 'lw':4}
    accepted_kw = {'zorder':500,  'drawstyle':'steps-post', 'alpha':0.5}

    convert = (lambda x: x) if mark_images == 'arcsec' \
              else (lambda x: Arcsec_to_Kpc([obj,data], x))

    normal = []
    hilite = []
    accepted = []
    imgs = {}
    xmin, xmax = inf, -inf
    ymin, ymax = inf, -inf
    for mi,m in enumerate(models[::every]):
        for [obj, data] in m['obj,data']:

            if not data.has_key(X): 
                print "Missing information for object %s with key %s. Skipping plot." % (obj.name,X)
                continue
            if not data.has_key(Y): 
                print "Missing information for object %s with %s. Skipping plot." % (obj.name,Y)
                continue

            si = m.get('accepted', 2)
            use[si] = 1

            s = _styles[si]

            #xmin, xmax = min(xmin, amin(data[X])), max(xmax, amax(data[X]))
            #ymin, ymax = min(ymin, amin(data[Y])), max(ymax, amax(data[Y]))

            if hilite_model == mi:
                hilite += [data[X], data[Y], hilite_color + s['ls']]
            elif si == 1:
                accepted += [data[X], data[Y], s['c'] + s['ls']]
            else:
                normal += [data[X], data[Y], s['c'] + s['ls']]

            if mark_images:
                for i,src in enumerate(obj.sources):
                    for img in src.images:
                        imgs[convert(abs(img.pos))] = 0

    if normal:   plotf(*normal, **normal_kw)
    if hilite:   plotf(*hilite, **hilite_kw)
    if accepted: plotf(*accepted, **accepted_kw)

    for x in imgs.iterkeys():
        axvline(x, c=system_color(0), ls='-', zorder=-2, alpha=0.5)

    if use[0] or use[1]:
        lines  = [s['line']  for s,u in zip(_styles, use) if u]
        labels = [s['label'] for s,u in zip(_styles, use) if u]
        legend(lines, labels)

    #axis('scaled')
    xlabel(x_label)
    ylabel(y_label)
    #xlim(xmin, xmax)
    #ylim(0, ymax)

def _find_key(xs, key):
    if hasattr(key, '__iter__'):
        for k in key[:-1]:
            xs = xs[k]
        key = key[-1]
    return xs[key]

def _data_plot2(models, X,Y, **kwargs):
    with_legend = False
    use = [0,0,0]
    if models is None:
        models = env().models
    elif not hasattr(models, '__getslice__'):
        models = [models]

    x_label = None
    y_label = None

    every = kwargs.get('every', 1)
    upto = kwargs.get('upto', len(models))
    plotf = kwargs.get('plotf', semilogy)
    mark_images = kwargs.get('mark_images', True)
    hilite_model = kwargs.get('hilite_model', None)
    hilite_color = kwargs.get('hilite_color', 'y')

    normal_kw   = {'zorder':0,    'drawstyle':'steps-post', 'alpha':0.5}
    hilite_kw   = {'zorder':-1000, 'drawstyle':'steps-post', 'alpha':1.0, 'lw':4}
    accepted_kw = {'zorder':500,  'drawstyle':'steps-post', 'alpha':0.5}

    normal = []
    hilite = []
    accepted = []
    #imgs = set()
    imgs = defaultdict(set)
    xmin, xmax = inf, -inf
    ymin, ymax = inf, -inf
    for mi,m in enumerate(models[:upto:every]):
        for [obj, data] in m['obj,data']:

            try:
                xs = _find_key(data, X)
                ys = _find_key(data, Y)

                x_label = xs.label
                y_label = ys.label
                #xs, ys = xs.v, ys.v

            except KeyError as bad_key:
                print "Missing information for object %s with key %s. Skipping plot." % (obj.name,bad_key)
                continue

            si = m.get('accepted', 2)
            use[si] = 1

            s = _styles[si]

            #xmin, xmax = min(xmin, amin(data[X])), max(xmax, amax(data[X]))
            #ymin, ymax = min(ymin, amin(data[Y])), max(ymax, amax(data[Y]))

            if hilite_model == mi:
                hilite += [xs, ys, hilite_color + s['ls']]
            elif si == 1:
                accepted += [xs, ys, s['c'] + s['ls']]
            else:
                normal += [xs,ys, s['c'] + s['ls']]

            if mark_images:
                for i,src in enumerate(obj.sources):
                    for img in src.images:
                        imgs[i].add(convert('arcsec to %s' % xs.units, abs(img.pos), obj.dL, data['nu']))

    if normal:   plotf(*normal, **normal_kw)
    if hilite:   plotf(*hilite, **hilite_kw)
    if accepted: plotf(*accepted, **accepted_kw)

    for k,v in imgs.iteritems():
        for img_pos in v:
            axvline(img_pos, c=system_color(k), ls='-', zorder=-2, alpha=0.5)

    if use[0] or use[1]:
        lines  = [s['line']  for s,u in zip(_styles, use) if u]
        labels = [s['label'] for s,u in zip(_styles, use) if u]
        legend(lines, labels)

    #axis('scaled')
    if x_label: xlabel(x_label)
    if y_label: ylabel(y_label)
    #xlim(xmin, xmax)
    #ylim(0, ymax)

@command
def glplot(models, ptype, xkeys, ykeys=[], **kwargs):
    if not ykeys: ykeys = ptype
    _data_plot2(models, xkeys, ykeys, **kwargs)

_enckappa_xlabel = r'$R$ $(\mathrm{arcsec})$'
_enckappa_ylabel = r'$\kappa(<R)$'
@command
def enckappa_plot(models=None, **kwargs):
    #if not kwargs.has_key('mark_images'): kwargs['mark_images'] = 'arcsec'
    kwargs.setdefault('mark_images', 'arsec')
    _data_plot(models, 'R', 'kappa(<R)', _enckappa_xlabel, _enckappa_ylabel, plotf=plot,**kwargs)

_kappa_prof_xlabel = r'$R$ $(\mathrm{arcsec})$'
_kappa_prof_ylabel = r'$\langle\kappa(R)\rangle$'
@command
def kappa_prof_plot(models=None, **kwargs):
    #if not kwargs.has_key('mark_images'): kwargs['mark_images'] = 'arcsec'
    kwargs.setdefault('mark_images', 'arsec')
    _data_plot(models, 'R', 'kappa(R)', _kappa_prof_xlabel, _kappa_prof_ylabel, plotf=plot,**kwargs)

_sigma_xlabel = r'$R$ $(\mathrm{kpc})$'
_sigma_ylabel = r'$\Sigma$ $(M_\odot/\mathrm{kpc}^2)$'
@command
def sigma_plot(models, **kwargs):
    kwargs.setdefault('mark_images', 'arsec')
    xaxis  = kwargs.setdefault('xaxis',       'R_kpc')
    xlabel = kwargs.setdefault('xlabel',      _sigma_xlabel)
    kwargs.setdefault('plotf',       semilogy)
    _data_plot(models, xaxis, 'sigma', xlabel, _sigma_ylabel, **kwargs)

_sigp_xlabel = r'$R$ $(\mathrm{kpc})$'
_sigp_ylabel = r'$\sigma_p$ $()$'
@command
def sigp_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:sigp', _sigp_xlabel, _sigp_ylabel, plotf=semilogx, **kwargs)
    #_data_plot(models, 'sigp:R', 'sigp:sigp', _sigp_xlabel, _sigp_ylabel, kwargs, plotf=plot)

_mass3d_xlabel = r'$r$ $(\mathrm{kpc})$'
_mass3d_ylabel = r'$M$'
@command
def mass3d_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:mass3d', _mass3d_xlabel, _mass3d_ylabel, plotf=loglog, **kwargs)

_rho_xlabel = r'$r$ $(\mathrm{kpc})$'
_rho_ylabel = r'$\rho$ $()$'
@command
def rho_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:rho', _rho_xlabel, _rho_ylabel, **kwargs)

_rhoint_xlabel = r'$r$ $(\mathrm{kpc})$'
_rhoint_ylabel = r'$\rho$ $()$'
@command
def rhoint_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:rhoint', _rhoint_xlabel, _rhoint_ylabel, **kwargs)

_drho_xlabel = r'$r$ $(\mathrm{kpc})$'
_drho_ylabel = r'$d\ln\rho/d\ln r$'
@command
def drho_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:drho', _drho_xlabel, _drho_ylabel, plotf=semilogx, **kwargs)

_rhoa_xlabel = r'$r$ $(\mathrm{kpc})$'
_rhoa_ylabel = r'$\rho_\mathrm{abel}$ $()$'
@command
def rhoa_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:rhoa', _rho_xlabel, _rho_ylabel, **kwargs)

_drhoa_xlabel = r'$r$ $(\mathrm{kpc})$'
_drhoa_ylabel = r'$d\ln\rho_\mathrm{abel}/d\ln r$'
@command
def drhoa_plot(models, **kwargs):
    _data_plot(models, 'sigp:r', 'sigp:drhoa', _drho_xlabel, _drho_ylabel, plotf=semilogx, **kwargs)

_encmass_xlabel = r'$R$ $(\mathrm{kpc})$'
_encmass_ylabel = r'$M(<R)$ $(M_\odot)$'
@command
def encmass_plot(models, **kwargs):
    kwargs.setdefault('mark_images', 'arsec')
    xaxis  = kwargs.setdefault('xaxis',       'R_kpc')
    xlabel = kwargs.setdefault('xlabel',      _encmass_xlabel)
    kwargs.setdefault('plotf',       semilogy)
    _data_plot(models, xaxis, 'encmass', xlabel, _encmass_ylabel, **kwargs)
    #_data_plot(models, 'R_kpc', 'encmass', _encmass_xlabel, _encmass_ylabel, plotf=plot, **kwargs)


_H0_xlabel = r'$H_0^{-1}$ (Gyr)'
@command
def H0_plot(models=None, objects=None, key='accepted'):
    if models is None: models = env().models

    # select a list to append to based on the 'accepted' property.
    l = [[], [], []]
    for m in models:
        obj, data = m['obj,data'][0] # For H0 we only have to look at one model because the others are the same
        l[m.get(key,2)].append(data['1/H0'])
        #l[2].append(data['kappa'][1])

    #print amin(l[2]), amax(l[2])

    not_accepted, accepted, notag = l

    #print 'H0_plot',H0s

    for d,s in zip(l, _styles):
        if d:
            print len(d), d
            #hist(d, bins=20, histtype='step', edgecolor=s['c'], zorder=s['z'], label=s['label'])
            hist(d, bins=ptp(d)//1+1, histtype='step', edgecolor=s['c'], zorder=s['z'], label=s['label'])

    if not_accepted or accepted:
        legend()

    axvline(13.7, c='k', ls=':', zorder = 2)

    xlabel(_H0_xlabel)
    ylabel(r'$\mathrm{Number}$')

    if accepted or not not_accepted:
        if accepted:
            h = array(accepted)
        else:
            h = array(accepted + notag)

        hs = sort(h)
        l = len(hs)

        m = hs[l * 0.50]
        u = hs[l * 0.68]
        l = hs[l * 0.32]

        axvline(m, c='r', ls='-', zorder = 2)
        axvline(u, c='g', ls='-', zorder = 2)
        axvline(l, c='g', ls='-', zorder = 2)

        print 'H0_plot: ', m, u, l
        print 'H0_plot: ', m, (u-m), (m-l)
    else:
        print "H0_plot: No H0 values accepted"

_time_delays_xlabel = r'Days'
@command
def time_delays_plot(models=None, object=0, key='accepted'):
    if models is None: models = env().models

    d = defaultdict(list)
    for m in models:
        obj,data = m['obj,data'][object]
        t0 = data['arrival times'][object][0]
        for i,t in enumerate(data['arrival times'][object][1:]):
            d[i].append( float('%0.6f'%convert('arcsec^2 to days', t-t0, obj.z, data['nu'])) )
            t0 = t

    for k,v in d.iteritems():
        print 'td plot', k, len(v)
        print v
        hist(v, histtype='step', label='%s - %s' % (str(k+1),str(k+2)))

    legend()

    xlabel(_time_delays_xlabel)
    ylabel(r'$\mathrm{Number}$')


_scale_factor_xlabel = r'Scale factor'
@command
def scale_factor_plot(models=None, objects=None, key='accepted'):
    if models is None: models = env().models

    # select a list to append to based on the 'accepted' property.
    l = [[], [], []]
    for m in models:
        # For H0 we only have to look at one model because the others are the same
        obj, data = m['obj,data'][0] 
        l[m.get(key,2)].append(data['sigp:scale-factor'])

    not_accepted, accepted, notag = l

    for d,s in zip(l, _styles):
        if d:
            hist(d, bins=ptp(d)//1+1, histtype='step', edgecolor=s['c'], zorder=s['z'], label=s['label'], log=False)

    if not_accepted or accepted:
        legend()

    xlabel(_scale_factor_xlabel)
    ylabel(r'$\mathrm{Number}$')


_chisq_xlabel = r'$\chi^2$'
@command
def chisq_plot(models=None, objects=None, key='accepted'):
    if models is None: models = env().models

    # select a list to append to based on the 'accepted' property.
    l = [[], [], []]
    for m in models:
        # For H0 we only have to look at one model because the others are the same
        obj, data = m['obj,data'][0] 
        l[m.get(key,2)].append(data['sigp:chisq'])

    not_accepted, accepted, notag = l

    for d,s in zip(l, _styles):
        if d:
            hist(d, histtype='step', edgecolor=s['c'], zorder=s['z'], label=s['label'], log=False)

    if not_accepted or accepted:
        legend()

    xlabel(_chisq_xlabel)
    ylabel(r'$\mathrm{Number}$')

