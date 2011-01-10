from numpy import array, mat, empty_like, amin, amax, repeat, logspace, arange, \
                  ptp, amin, amax, sqrt, sort, abs, put, empty, zeros, ogrid, \
                  mgrid, atleast_2d, linspace, meshgrid, log10, log, diff, ravel, \
                  meshgrid, vstack, argsort, logical_and, logical_not, where, argmin
from pylab import show, imshow, contour, gca, scatter, xlabel, ylabel, plot, loglog, \
                  hist, hold, colorbar, legend, over, axvline, matshow, gcf, subplot, \
                  suptitle, figure, grid, gray, semilogx, imread, imshow, errorbar, \
                  text, axvline, axhline

import matplotlib.cm as cm  
import mpl_toolkits.mplot3d as p3

from environment import env

def kappa_plot3d(model, obj_index=0, with_contours=False, only_contours=False, clevels=30, with_colorbar=False):

    obj, data = model['obj,data'][obj_index]

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
    #ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    #ax.contour(X,Y,Z, rstride=10, cstride=10, cmap=cm.terrain, levels=clevels)

def kappa_ensemble_plot(models=None, obj_index=0, with_contours=False, only_contours=False, clevels=30, with_colorbar=False):
    if models is None: models = env().models

    grid = None

    with_colorbar=True

    n = 0
    for mi,model in enumerate(models):
        o,d = model['obj,data'][obj_index]
        if grid is None:
            grid = o.basis.kappa_grid(d).copy()
            L    = o.basis.pixrad
            R    = o.basis.mapextent
            S    = o.basis.subdivision
        else:
            grid += o.basis.kappa_grid(d)
        n+=1

    grid /= len(models)

    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.terrain,
          'fignum': False,
          'vmin': -1,
          'vmax':  1}

    if not only_contours:
        matshow(log10(grid), **kw)
        if with_colorbar: colorbar()

    if with_contours:
        over(contour, grid, clevels, extent=[-R,R,-R,R], extend='both', colors='w', alpha=0.7, **kw)

    xlabel('arcsec')
    ylabel('arcsec')

def kappa_ensemble_plot3d(models=None, obj_index=0, 
                          with_contours=False, 
                          only_contours=False, 
                          with_colorbar=False,
                          clevels=30):

    if models is None: models = env().models

    Z = None

    for mi,model in enumerate(models):
        o,d = model['obj,data'][obj_index]
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
    #ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    #ax.contour(X,Y,Z, rstride=10, cstride=10, cmap=cm.terrain, levels=clevels)

def kappa_compare_plot(models, base_model, obj_index, sort=True, normalize=False, label=True, mark=None):
    obj0,data0 = base_model['obj,data'][obj_index]
    N  = len(models)

    assert len(models) > 0

    obj,_ = models[0]['obj,data'][obj_index]
    rs = [ abs(img.pos) for src in obj.sources for img in src.images ]
    rmin, rmax = amin(rs), amax(rs)
    #print rmin, rmax
    #print abs(obj.basis.ploc)


    rmin = 0 #abs(obj.basis.ploc[argmin(abs(abs(obj.basis.ploc)-rmin))])
    rmax = abs(obj.basis.ploc[argmin(abs(abs(obj.basis.ploc)-rmax))])

    plist, = where(logical_and(abs(obj.basis.ploc) <= rmax, abs(obj.basis.ploc) >= rmin))

    Nk = len(plist)
    kappas = empty((N, Nk))

    for i,model in enumerate(models):
        _,data = model['obj,data'][obj_index]
        kappas[i,:] = data['kappa'][plist]
       
    data0 = data0['kappa'][plist]

    if sort:
        bs = argsort(data0)
        ks = kappas.take(bs, axis=1)
        bs = data0[bs]
    else:
        bs = data0
        ks = kappas

    every=1

    ds = bs[::every]
    ks.sort(axis=0) 

    if normalize:
        ks /= ds
        ds = 1

    xs = arange(Nk)[::every]
    ys = ks[int(N*0.50),:][::every]
    #ds = data0[bs][::every]


    lo100, hi100 = int(N*0.00), int(N*1.00)-1
    lo68,  hi68  = int(N*0.16), int(N*0.84)

    hi = ks[hi100,:][::every]
    lo = ks[lo100,:][::every]
    es = vstack((ys-lo, hi-ys))
    errorbar(xs, ys, es, ecolor="#AAAAAA", ls='None', barsabove=True)

    bad = []

    if mark and mark == 'all':
        w = where(logical_not(logical_and(lo <= ds, ds <= hi)))
        bad = xs[w[0]]

    hi = ks[hi68,:][::every]
    lo = ks[lo68,:][::every]
    es = vstack((ys-lo, hi-ys))
    errorbar(xs, ys, es, ecolor="#555555", ls='None', barsabove=True)

    if mark and mark == '1sigma':
        w = where(logical_not(logical_and(lo <= ds, ds <= hi)))
        bad = xs[w[0]]

    if not normalize:
        all_ds = bs
    else:
        all_ds = 1

    lo,hi = ks[lo100,:], ks[hi100,:]
    within100 = 100 * sum(logical_and(lo <= all_ds, all_ds <= hi))/Nk

    lo,hi = ks[lo68,:], ks[hi68,:]
    within68 = 100 * sum(logical_and(lo <= all_ds, all_ds <= hi))/Nk

    if mark:
        for b in bad:
            axvline(b, color='r')

    if not normalize:
        plot(xs, ds, "k-", lw=4)
    else:
        axhline(1, linewidth=4, color='k')
    
    if label:
        text(0.1, 0.65, '%3i\\%% within  $1\sigma$\n%3i\\%% within $\infty\sigma$' % (within68, within100),
             family = 'monospace', fontsize=14,
             transform = gca().transAxes)

    xlabel('Sort index')
    ylabel(r'$\kappa$')


def kappa_compare_grid_plot(models, base_model, obj_index):
    obj0,data0 = base_model['obj,data'][obj_index]
    N  = len(models)

    assert len(models) > 0

    obj,_ = models[0]['obj,data'][obj_index]
    rs = [ abs(img.pos) for src in obj.sources for img in src.images ]
    rmin, rmax = amin(rs), amax(rs)
    #print rmin, rmax
    #print abs(obj.basis.ploc)


    rmin = 0 #abs(obj.basis.ploc[argmin(abs(abs(obj.basis.ploc)-rmin))])
    rmax = abs(obj.basis.ploc[argmin(abs(abs(obj.basis.ploc)-rmax))])

    plist, = where(logical_and(abs(obj.basis.ploc) <= rmax, abs(obj.basis.ploc) >= rmin))

    Nk = len(plist)
    kappas = empty((N, Nk))

    for i,model in enumerate(models):
        _,data = model['obj,data'][obj_index]
        kappas[i,:] = data['kappa'][plist]
       
    bs = data0['kappa'][plist]
    ks = kappas

    ks.sort(axis=0) 
    xs = arange(Nk)
    ys = ks[int(N*0.50),:]
    ds = bs

    lo68,  hi68  = int(N*0.16), int(N*0.84)
    lo = ks[lo68,:]
    hi = ks[hi68,:]

    w, = where(logical_and(lo <= ds, ds <= hi))
    g = bs.copy()
    g[w] = 0

    bs = data0['kappa'].copy()
    bs[:] = 0
    bs[plist] = g

    grid = obj.basis._to_grid(bs)
    R = obj.basis.mapextent
    #matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
    #colorbar()

    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.terrain,
          'fignum': False,
          'vmin': -1,
          'vmax':  1}

    matshow(log10(grid), **kw)
    colorbar()
    return

