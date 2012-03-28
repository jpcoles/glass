import pylab as pl
import numpy as np
import matplotlib.cm as cm  
import mpl_toolkits.mplot3d as p3

from matplotlib.lines import Line2D
from itertools import izip
from numpy import logical_and, logical_not, where
from pylab import gca, gcf

from glass.command import command
from glass.plots import default_kw


def glscolorbar():
    rows,cols,_ = gca().get_geometry()
    x,y = gcf().get_size_inches()
    pars = gcf().subplotpars
    left = pars.left
    right = pars.right
    bottom = pars.bottom
    top = pars.top
    wspace = x*pars.wspace
    hspace = y*pars.hspace
    totWidth = x*(right-left)
    totHeight = y*(top-bottom)

    figH = (totHeight-(hspace*(rows>1))) / rows
    figW = (totWidth-(wspace*(cols>1))) / cols

    pl.colorbar(shrink=figW/figH)

@command
def kappa_plot3d(env, model, obj_index=0, with_contours=False, only_contours=False, clevels=30, with_colorbar=False):

    obj, data = model['obj,data'][obj_index]

    L = obj.basis.pixrad
    R = obj.basis.mapextent
    r = obj.maprad
    S = obj.basis.subdivision

    coords = obj.basis.xy

    X = np.linspace(-R, R, (2*L+1) * S)
    Y = X
    X, Y = np.meshgrid(X, Y)
    Z = obj.basis.kappa_grid(data)

    print Z.shape
    print (2*L+1) * S

    ax = p3.Axes3D(gcf(), rect=gca().get_position())
    #ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    #ax.contour(X,Y,Z, rstride=10, cstride=10, cmap=cm.terrain, levels=clevels)

@command
def srcdiff_plot3d(env, model, obj_index=0, src_index=0, with_contours=False, only_contours=False, clevels=30, with_colorbar=False):

    obj, data = model['obj,data'][obj_index]

    L = obj.basis.pixrad
    R = obj.basis.mapextent
    r = obj.maprad
    S = obj.basis.subdivision

    coords = obj.basis.xy

    X = np.linspace(-R, R, (2*L+1) * S)
    Y = X
    X, Y = np.meshgrid(X, Y)
    Z = obj.basis.srcdiff_grid(data)[src_index]

    print Z.shape
    print (2*L+1) * S

    ax = p3.Axes3D(gcf(), rect=gca().get_position())
    ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1) #, cmap=cm.terrain)
    #ax.plot_surface(X,Y,Z, rstride=1, cstride=1) #, cmap=cm.terrain)
    #ax.contour(X,Y,Z, rstride=10, cstride=10, cmap=cm.terrain, levels=clevels)

@command
def kappa_ensemble_plot(env, models=None, obj_index=0, with_contours=False, only_contours=False, clevels=30, with_colorbar=False):
    if models is None: models = env.models

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
        pl.matshow(np.log10(grid), **kw)
        if with_colorbar: glscolorbar()

    if with_contours:
        pl.contour(grid, clevels, extent=[-R,R,-R,R], extend='both', colors='w', alpha=0.7, **kw)

    pl.xlabel('arcsec')
    pl.ylabel('arcsec')

@command
def kappa_ensemble_plot3d(env, models=None, obj_index=0, 
                          with_contours=False, 
                          only_contours=False, 
                          with_colorbar=False,
                          clevels=30):

    if models is None: models = env.models

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

    X = np.linspace(-R, R, (2*L+1) * S)
    Y = X
    X, Y = np.meshgrid(X, Y)

    ax = p3.Axes3D(gcf(), rect=gca().get_position())
    #ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.terrain)
    #ax.contour(X,Y,Z, rstride=10, cstride=10, cmap=cm.terrain, levels=clevels)

@command
def kappa_compare_plot(env, models, base_model, obj_index, sort=True, normalize=False, label=True, mark=None):
    obj0,data0 = base_model['obj,data'][obj_index]
    N  = len(models)

    assert len(models) > 0

    obj,_ = models[0]['obj,data'][obj_index]
    rs = [ np.abs(img.pos) for src in obj.sources for img in src.images ]
    rmin, rmax = np.amin(rs), np.amax(rs)
    #print rmin, rmax
    #print np.abs(obj.basis.ploc)


    rmin = np.abs(obj.basis.ploc[np.argmin(np.abs(np.abs(obj.basis.ploc)-rmin))])
    rmax = np.abs(obj.basis.ploc[np.argmin(np.abs(np.abs(obj.basis.ploc)-rmax))])

    plist, = where(logical_and(np.abs(obj.basis.ploc) <= rmax, np.abs(obj.basis.ploc) >= rmin))
    #plist = np.arange(len(obj.basis.ploc))
    Nk = len(plist)

    kappas = np.empty((N, Nk))

    for i,model in enumerate(models):
        _,data = model['obj,data'][obj_index]
        kappas[i,:] = data['kappa'][plist]
       
    data0 = data0['kappa'][plist]

    if sort:
        bs = np.argsort(data0)
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

    xs = np.arange(len(obj.basis.ploc))[::every]
    ys = ks[int(N*0.50),:][::every]
    #ds = data0[bs][::every]


    lo100, hi100 = int(N*0.00), int(N*1.00)-1
    lo68,  hi68  = int(N*0.16), int(N*0.84)

    hi = ks[hi100,:][::every]
    lo = ks[lo100,:][::every]
    es = np.vstack((ys-lo, hi-ys))
    pl.errorbar(xs[plist], ys, es, ecolor="#AAAAAA", ls='None', barsabove=True)

    bad = []

    if mark and mark == 'all':
        w = where(logical_not(logical_and(lo <= ds, ds <= hi)))
        bad = xs[w[0]]

    hi = ks[hi68,:][::every]
    lo = ks[lo68,:][::every]
    es = np.vstack((ys-lo, hi-ys))
    pl.errorbar(xs[plist], ys, es, ecolor="#555555", ls='None', barsabove=True)
    #pl.plot(xs, models[0]['obj,data'][obj_index][1]['kappa'][plist] / data0, 'b-')

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
            pl.axvline(b, color='r')

    if not normalize:
        pl.plot(xs, ds, "k-", lw=4)
    else:
        pl.axhline(1, linewidth=4, color='k')

    for r in obj.basis.rings:
        R = r[0] #np.abs(obj.basis.ploc[r[0]])
        pl.axvline(R, ls=':', color='k')

    if label:
        pl.text(0.1, 0.85, '% 3i\\%% within  $1\sigma$\n%3i\\%% within $\infty\sigma$' % (within68, within100),
             family = 'monospace', fontsize=14,
             transform = gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='white', alpha=0.8))

    if sort:
        pl.xlabel('Sort index')
    else:
        pl.xlabel('Pixel index')
    pl.ylabel(r'$\kappa$')
    pl.xlim(xmin=0, xmax=obj.basis.rings[-1][-1])


@command
def kappa_compare_grid_plot(env, models, base_model, obj_index):
    obj0,data0 = base_model['obj,data'][obj_index]
    N  = len(models)

    assert len(models) > 0

    obj,_ = models[0]['obj,data'][obj_index]
    rs = [ np.abs(img.pos) for src in obj.sources for img in src.images ]
    rmin, rmax = np.amin(rs), np.amax(rs)
    #print rmin, rmax
    #print np.abs(obj.basis.ploc)


    rmin = 0 #np.abs(obj.basis.ploc[np.argmin(np.abs(np.abs(obj.basis.ploc)-rmin))])
    rmax = np.abs(obj.basis.ploc[np.argmin(np.abs(np.abs(obj.basis.ploc)-rmax))])

    plist, = where(logical_and(np.abs(obj.basis.ploc) <= rmax, np.abs(obj.basis.ploc) >= rmin))

    Nk = len(plist)
    kappas = np.empty((N, Nk))

    for i,model in enumerate(models):
        _,data = model['obj,data'][obj_index]
        kappas[i,:] = data['kappa'][plist]
       
    bs = data0['kappa'][plist]
    ks = kappas

    ks.sort(axis=0) 
    xs = np.arange(Nk)
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
    #pl.matshow(grid, fignum=False, extent=[-R,R,-R,R], interpolation='nearest')
    #colorbar()

    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.terrain,
          'fignum': False,
          'vmin': -1,
          'vmax':  1}

    pl.matshow(np.log10(grid), **kw)
    glscolorbar()
    return

@command
def kappa_residual_grid_plot(env, model, base_model, obj_index, with_contours=False, only_contours=False, with_colorbar=True):
    obj0,data0 = base_model['obj,data'][obj_index]
    obj1,data1 = model['obj,data'][obj_index]

    kappa = data1['kappa'] - data0['kappa']
       
    grid = obj1.basis._to_grid(kappa, obj1.basis.subdivision)
    R = obj1.basis.mapextent

    kw = {'extent': [-R,R,-R,R],
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          'cmap': cm.gist_stern,
          'fignum': False}
          #'vmin': -1,
          #'vmax':  1}

    if not only_contours:
        pl.matshow(grid, **kw)
    if only_contours or with_contours:
        kw.update({'colors':'k', 'linewidths':1, 'cmap':None})
        pl.contour(grid, **kw)
        kw.update({'colors':'k', 'linewidths':2, 'cmap':None})
        pl.contour(grid, [0], **kw)

    if with_colorbar:
        glscolorbar()
    return

_chi2_xlabel = r'$\ln \chi^2$'
@command
def radial_chi2_plot(env, models, model0):
    rchi2 = None
    for m in models:
        for m1,m2 in izip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            if rchi2 is None:
                rchi2 = [ np.zeros_like(data['R']['arcsec']),  np.zeros_like(data['R']['arcsec']) ]
                xs = data['R']['arcsec']
            for ri,r in enumerate(obj.basis.rings):
                v0 = data0['kappa(R)'][ri]
                v1 = data['kappa(R)'][ri]
                rchi2[0][ri] += np.sum((v1 - v0)**2)
                rchi2[1][ri] += np.sum(v0**2)
    pl.plot(xs, np.log(rchi2[0] / rchi2[1]), 'k.-')
    pl.ylabel(_chi2_xlabel)
    pl.xlabel(r'$R$ (arcsec)')
#   print '+'*80
#   print '+'*80
#   print np.log(np.sum(rchi2[0] / rchi2[1]))
#   print np.log(np.sum(rchi2[0]) / np.sum(rchi2[1]))
#   print '+'*80
#   print '+'*80


@command
def gradient_plot(env, model, obj_index):
    obj,data = model['obj,data'][obj_index]
    b = obj.basis
    kappa = data['kappa']

    #wght = lambda x: 1.0 / len(x) if len(x) else 0
    wght = lambda x: b.cell_size[x]**2 / np.sum(b.cell_size[x]**2)
    for i,r in enumerate(b.ploc):
        n,e,s,w = b.nbrs3[i][2]

        dx = np.sum(kappa[w] * wght(w)) - np.sum(kappa[e] * wght(e))
        dy = np.sum(kappa[s] * wght(s)) - np.sum(kappa[n] * wght(n))

        dx*=-1
        dy*=-1

        #print dx, dy

        dr = np.sqrt(dx**2 + dy**2)
        dx /= dr
        dy /= dr
        dx *= .9 * b.cell_size[i]/2
        dy *= .9 * b.cell_size[i]/2

        x0,y0 = r.real, r.imag
        x1,y1 = x0 + dx, y0 + dy
        gca().add_artist(Line2D([x0,x1], [y0,y1], linewidth=1))

@command
def gradient_grid_plot(env, model, obj_index):
    obj,data = model['obj,data'][obj_index]
    b = obj.basis
    kappa = data['kappa']
    grid = np.zeros_like(kappa)

    #wght = lambda x: 1.0 / len(x) if len(x) else 0
    wght = lambda x: b.cell_size[x]**2 / np.sum(b.cell_size[x]**2)
    for i,r in enumerate(b.ploc):
        n,e,s,w = b.nbrs3[i][2]

        dx = np.sum(kappa[w] * wght(w)) - np.sum(kappa[e] *  wght(e))
        dy = np.sum(kappa[s] * wght(s)) - np.sum(kappa[n] *  wght(n))

        dx*=-1
        dy*=-1

        #print dx, dy

        dr = np.sqrt(dx**2 + dy**2)
        grid[i] = dr 

    grid = grid
    kw = default_kw(b.mapextent, vmin=np.amin(grid), vmax=np.amax(grid))
    grid = b._to_grid(grid, b.subdivision)
    pl.matshow(grid, **kw)
    glscolorbar()

