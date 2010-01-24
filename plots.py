from __future__ import division
from numpy import array, mat, empty_like, amin, amax, repeat, logspace, arange, ptp, amin, amax, sqrt, sort, abs
from numpy import put, empty, zeros, ogrid, mgrid, atleast_2d, linspace, meshgrid, log10, log, diff, ravel, meshgrid, vstack, argsort, logical_and
from numpy.random import random
from pylab import show, imshow, contour, gca, scatter, xlabel, ylabel, plot, loglog, \
                  hist, hold, colorbar, legend, over, axvline, matshow, gcf, subplot, \
                  suptitle, figure, grid, gray, jet, semilogx, imread, imshow, errorbar
import matplotlib
import matplotlib.cm as cm  
from matplotlib import rc
from matplotlib.ticker import LogLocator
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scales import density_to_physical, distance_to_physical, Arcsec_to_Kpc
import math

#import matplotlib.axes3d as p3
import mpl_toolkits.mplot3d as p3



from scipy.ndimage.filters import correlate1d
from scipy.misc import central_diff_weights

rc('text', usetex=True)

_styles = [{'label':r'rejected', 'c':'r', 'ls':'-', 'z':-1, 'line':Line2D([],[],c='r',ls='-')},
           {'label':r'accepted', 'c':'b', 'ls':'-', 'z': 0, 'line':Line2D([],[],c='b',ls='-')},
           {'label':r'unknown',  'c':'k', 'ls':'-', 'z':+1, 'line':Line2D([],[],c='k',ls='-')}]

_system_colors = 'rgbmykw'
_source_colors = 'c'

def system_color(i): return _system_colors[i%len(_system_colors)]
def source_color(i): return _source_colors[i%len(_source_colors)]

def img_plot(model, src_index=None):
    if src_index is not None and not isinstance(src_index, (list,tuple)):
        src_index = [src_index]

    if isinstance(model, (list, tuple)):
        obj,_ = model
    else:
        obj = model

    for i,src in enumerate(obj.sources):
        if src_index is not None and i not in src_index: continue
        xs,ys,cs = [], [], []

        for img in src.images:
            #print img.pos
            xs.append(img.pos.real)
            ys.append(img.pos.imag)
            if img.parity_name == 'unk':
                cs.append('red')
            else:
                cs.append(source_color(i))

        if xs and ys:
            over(scatter,xs, ys, s=80, c=cs, zorder=1000)
            a = gca()
            for x,y in zip(xs,ys):
                a.add_artist(Circle((0,0),sqrt(x**2 + y**2), fill=False))

def src_plot(models, hilite_model=None):

    def plot(model, hilite=False):
        obj, data = model
        xs = []
        ys = []
        cs = []
        for i,sys in enumerate(obj.sources):
            xs.append(data['src'][i].real)
            ys.append(data['src'][i].imag)
            cs.append(source_color(i))
        if hilite:
            over(scatter,xs, ys, s=80, c='g', zorder=2000)
        else:
            over(scatter,xs, ys, s=80, c=cs, zorder=1000)

    if isinstance(models, (list,tuple)) and len(models)>0 and isinstance(models[0], (list,tuple)):
        for mi,model in enumerate(models):
            for m in model['obj,data']:
                plot(m, mi==hilite_model)
    else:
        plot(models)

_src_hist_xlabel = r'$r$ $(\mathrm{arcsec})$'
_src_hist_ylabel = r'$\mathrm{Number}$'
def src_hist(models, hilite_model=None):
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

def mass_plot(model, with_contours=True, only_contours=False, clevels=30):
    print "WARNING: use of mass_plot is deprecated. Use kappa_plot instead."
    return kappa_plot(model, with_contours, only_contours, clevels)

def kappa_plot(model, with_contours=True, only_contours=False, clevels=30, with_colorbar=True):
    obj, data = model

    R = obj.basis.mapextent

    grid = obj.basis.kappa_grid(data)

    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper'}

    if not only_contours:
        imshow(grid, **kw)
        #imshow(grid, fignum=False, **kw)
        #matshow(grid, fignum=False, **kw)
        if with_colorbar: colorbar()

    if with_contours:
        over(contour, grid, clevels, extend='both', colors='w', alpha=0.7, **kw)

    xlabel('arcsec')
    ylabel('arcsec')

def kappa_ensemble_plot(models, obj, with_contours=True, only_contours=False, clevels=30, with_colorbar=True):

    grid = None

    n = 0
    for mi,model in enumerate(models):
        for oi,od in enumerate(model['obj,data']):
            if oi == obj:
                o,d = od
                if grid is None:
                    grid = o.basis.kappa_grid(d).copy()
                    L    = o.basis.pixrad
                    R    = o.basis.mapextent
                    S    = o.basis.subdivision
                else:
                    grid += o.basis.kappa_grid(d)
                n+=1

    grid /= len(models)

    if not only_contours:
        matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
        if with_colorbar: colorbar()

    if with_contours:
        over(contour, grid, clevels, extent=[-R,R,-R,R], extend='both', colors='w', alpha=0.7, origin='upper')

    xlabel('arcsec')
    ylabel('arcsec')

def kappa_plot3d(model, with_contours=True, only_contours=False, clevels=30, with_colorbar=True):
    obj, data = model

    L = obj.basis.pixrad
    R = obj.basis.mapextent
    r = obj.maprad
    S = obj.basis.subdivision

    coords = obj.basis.xy

    X = linspace(-R, R, (2*L+1) * S)
    Y = X
    X, Y = meshgrid(X, Y)
    Z = obj.basis.kappa_grid(data)

    print Z.shape
    print (2*L+1) * S

    ax = p3.Axes3D(gcf(), rect=gca().get_position())
    #ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet)
    #ax.contour(X,Y,Z, rstride=10, cstride=10, cmap=cm.jet, levels=clevels)

def kappa_ensemble_plot3d(models, obj, with_contours=True, only_contours=False, clevels=30, with_colorbar=True):

    Z = None

    for mi,model in enumerate(models):
        for oi,od in enumerate(model['obj,data']):
            if oi == obj:
                o,d= od
                if Z is None:
                    Z = o.basis.kappa_grid(d).copy()
                    L = o.basis.pixrad
                    R = o.basis.mapextent
                    S = o.basis.subdivision
                else:
                    Z += o.basis.kappa_grid(d)

    Z /= len(models)

    X = linspace(-R, R, (2*L+1) * S)
    Y = X
    X, Y = meshgrid(X, Y)

    ax = p3.Axes3D(gcf(), rect=gca().get_position())
    #ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet)
    #ax.contour(X,Y,Z, rstride=10, cstride=10, cmap=cm.jet, levels=clevels)

def kappa_compare_plot(models, obj, base):
    obj0,data0 = base
    N  = len(models)
    Nk = len(data0['kappa'])
    kappas = empty((N, Nk))

    i=0
    for mi,model in enumerate(models):
        for oi,od in enumerate(model['obj,data']):
            if oi == obj:
                kappas[i,:] = od[1]['kappa']
                i += 1

    bs = argsort(data0['kappa'])
    ks = kappas.take(bs, axis=1)

    every=2
    ks.sort(axis=0) 
    xs = arange(Nk)[::every]
    ys = ks[int(N*0.50),:][::every]
    ds = data0['kappa'][bs][::every]

    hi = ks[int(N*1.00)-1,:][::every]
    lo = ks[int(N*0.00),:][::every]
    es = vstack((ys-lo, hi-ys))
    errorbar(xs, ys, es, ecolor="#AAAAAA", ls='None', barsabove=True)

    print sum(logical_and(lo <= ds, ds <= hi))/(Nk/every)

    hi = ks[int(N*0.84),:][::every]
    lo = ks[int(N*0.16),:][::every]
    es = vstack((ys-lo, hi-ys))
    errorbar(xs, ys, es, ecolor="#555555", ls='None', barsabove=True)

    print sum(logical_and(lo <= ds, ds <= hi))/(Nk/every)

    plot(xs, ds, "k-", lw=4)

def potential_plot(model, sys):
    obj, data = model
    R = obj.basis.mapextent
    grid = obj.basis.potential_grid(data)
    levs = obj.basis.potential_contour_levels(data)
#   matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
    matshow(grid, fignum=False, cmap=cm.jet, extent=[-R,R,-R,R], interpolation='nearest')
    colorbar()
#   contour(grid, extent=[-R,R,-R,R], origin='upper')
    #print levs
    for i,lev in enumerate(levs):
        over(contour, grid, lev, colors = system_color(i), 
             extent=[-R,R,-R,R], origin='upper', extend='both')

    xlabel('arcsec')
    ylabel('arcsec')
    #suptitle('Potential')

def critical_curve_plot(model, src_index):
    obj, data = model
    R = obj.basis.mapextent
    g = obj.basis.maginv_grid(data)[src_index]
    matshow(g, fignum=False, cmap=cm.jet, extent=[-R,R,-R,R], interpolation='nearest')
    over(contour, g, [0], colors='g', linewidths=1, extent=[-R,R,-R,R], origin='upper')

def arrival_plot(model, src_index, only_contours=False, clevels=300, with_colorbar=True):
    obj, data = model
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g   = obj.basis.arrival_grid(data)[src_index]
    lev = obj.basis.arrival_contour_levels(data)
    if lev: lev = lev[src_index]

    if not only_contours:
        matshow(g, fignum=False, cmap=cm.jet, extent=[-R,R,-R,R], interpolation='nearest')
        if with_colorbar: colorbar()
        #lev = 50 if not lev else lev[src_index]

    print 'arrival_plot:', lev
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    print amin(g), amax(g)
    loglev = logspace(1, log(amax(g)-amin(g)), 20, base=math.e) + amin(g)
    print loglev
    over(contour, g, 
         loglev,
         #clevels, #logspace(amin(g), amax(g), 50),  
         colors='w',
         linewidths=1, 
         extent=[-R,R,-R,R], 
         origin='upper')
         #extent=[-R,R,-R,R], origin='upper', extend='both')
    if lev:
        print '***', lev, '***'
        over(contour, g, lev, colors=system_color(src_index), linewidths=3, 
             extent=[-R,R,-R,R], origin='upper')
    #grid()

def srcdiff_plot(model, src_index):
    obj, data = model
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g   = obj.basis.srcdiff_grid(data)[src_index]

    matshow(g, fignum=False, cmap=cm.jet, extent=[-R,R,-R,R], interpolation='nearest')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
#   over(contour, g, 50,  colors='w',               linewidths=1, 
#        extent=[-R,R,-R,R], origin='upper', extend='both')
    #grid()

def deflect_plot(model, which, src_index):
    obj, data = model
    S = obj.basis.subdivision
    R = obj.basis.mapextent

    g = obj.basis.deflect_grid(data, which, src_index)

    matshow(g, fignum=False, cmap=cm.jet, extent=[-R,R,-R,R], interpolation='nearest')
    #matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

def grad_tau(model, which, src_index):

    assert which in ['x','y'], "grad_tau: 'which' must be one of 'x' or 'y'"

    print "grad_tau"
    obj,ps = model
    R = obj.basis.mapextent

    #---------------------------------------------------------------------------
    # Find the derivative of the arrival time surface.
    #---------------------------------------------------------------------------
    arrival = obj.basis.arrival_grid(ps)[src_index]

    w = central_diff_weights(3)

    which = 1 if which == 'x' else 0
    d = correlate1d(arrival, w, axis=which, mode='constant')

    d = d[1:-1,1:-1]
    d[d>0] = 1
    d[d<0] = -1
    matshow(d, fignum=False, extent=[-R,R,-R,R])

def deriv(model, src_index, m, axis, R):
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
    arrival_plot(model, src_index, only_contours=True, clevels=200)
    #img_plot(model, src_index=src_index)
    #matshow(d)

#   d = d[1:-1,1:-1]
#   d[d>0] = 1
#   d[d<0] = -1
#   matshow(d, extent=[-R,R,-R,R])
#   img_plot(model, src_index=src_index)

def inout_plot(model, src_index):
    print "inout"
    obj,ps = model
    R = obj.basis.mapextent
    arrival = obj.basis.arrival_grid(ps)[src_index]

    deriv(model, src_index, arrival, 0, R)
    deriv(model, src_index, arrival, 1, R)

def _data_plot(models, X,Y, x_label, y_label, **kwargs):
    with_legend = False
    use = [0,0,0]
    every = kwargs.get('every', 1)
    plotf = kwargs.get('plotf', loglog)
    mark_images = kwargs.get('mark_images', False)
    hilite_model = kwargs.get('hilite_model', None)
    normal = []
    hilite = []
    imgs = {}
    for mi,m in enumerate(models[::every]):
        for [obj, data] in m['obj,data']:

            si = m.get('accepted', 2)
            use[si] = 1

            s = _styles[si]

            if hilite_model == mi:
                hilite += [data[X], data[Y], 'g' + s['ls']]
            else:
                normal += [data[X], data[Y], s['c'] + s['ls']]

            if mark_images == 'arcsec':
                for i,src in enumerate(obj.sources):
                    for img in src.images:
                        imgs[abs(img.pos)] = 0
                        #axvline(abs(img.pos), c=system_color(i), ls='-', zorder=-2, alpha=0.5)
            elif mark_images:
                for i,src in enumerate(obj.sources):
                    for img in src.images:
                        x = Arcsec_to_Kpc([obj, data], abs(img.pos))
                        imgs[x] = 0

    if normal:
        #plotf(*normal, zorder=0)
        plotf(*normal, zorder=0, drawstyle='steps')
    if hilite:
        #plotf(*hilite, zorder=1000)
        plotf(*hilite, zorder=1000, drawstyle='steps')

    for x in imgs.keys():
        axvline(x, c=system_color(0), ls='-', zorder=-2, alpha=0.5)

    if use[0] or use[1]:
        lines = [s['line']  for s,u in zip(_styles, use) if u]
        labels = [s['label'] for s,u in zip(_styles, use) if u]
        legend(lines, labels)

    xlabel(x_label)
    ylabel(y_label)

_enckappa_xlabel = r'$R$ $(\mathrm{arcsec})$'
_enckappa_ylabel = r'$\kappa(<R)$'
def enckappa_plot(models, **kwargs):
    _data_plot(models, 'R', 'enckappa', _enckappa_xlabel, _enckappa_ylabel, plotf=plot,**kwargs)

_sigma_xlabel = r'$R$ $(\mathrm{kpc})$'
_sigma_ylabel = r'$\Sigma$ $(M_\odot/\mathrm{kpc}^2)$'
def sigma_plot(models, **kwargs):
    _data_plot(models, 'R_kpc', 'sigma', _sigma_xlabel, _sigma_ylabel, plotf=plot,**kwargs)

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
_encmass_ylabel = r'$M(<R)$ $(M_\odot)$'
def encmass_plot(models, **kwargs):
    _data_plot(models, 'R_kpc', 'encmass', _encmass_xlabel, _encmass_ylabel, plotf=plot, **kwargs)


_H0_xlabel = r'$H_0^{-1}$ (Gyr)'
def H0_plot(models, objects=None, key='accepted'):

    # select a list to append to based on the 'accepted' property.
    l = [[], [], []]
    for m in models:
        obj, data = m['obj,data'][0] # For H0 we only have to look at one model because the others are the same
        l[m.get(key,2)].append(data['1/H0'])

    not_accepted, accepted, notag = l

    #print 'H0_plot',H0s

    for d,s in zip(l, _styles):
        if d:
            print len(d), d
            hist(d, bins=ptp(d)//1+1, histtype='step', edgecolor=s['c'], zorder=s['z'], label=s['label'])

    if not_accepted or accepted:
        legend()

    axvline(13.7, c='k', ls=':', zorder = 2)

    xlabel(_H0_xlabel)
    ylabel(r'$\mathrm{Number}$')

    if accepted:
        h = array(accepted)
        hs = sort(h)
        l = len(hs)

        m = hs[l * 0.50]
        u = hs[l * 0.68]
        l = hs[l * 0.32]

        print 'H0_plot: ', m, u, l
        print 'H0_plot: ', m, (u-m), (m-l)
    else:
        print "H0_plot: No H0 values accepted"


_scale_factor_xlabel = r'Scale factor'
def scale_factor_plot(models, objects=None, key='accepted'):

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
def chisq_plot(models, objects=None, key='accepted'):

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

