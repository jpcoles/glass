from __future__ import division

glass_basis('glass.basis.pixels', solver=None)
exclude_all_priors()

import os

#try:
if 1:
     import pylab as pl
     from pylab import show, figure, ion, ioff, savefig, gcf
#except:
if 0:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as pl
    from matplotlib.pyplot import figure, ion, ioff, savefig, gcf
from math import pi, cos

def escape(s):
    s = s.replace('_', r'\_')
    return s

def style_iterator(colors='gbrcm'):
    import matplotlib.lines as mpll
    from itertools import count
    _linestyles = [k for k,v, in mpll.lineStyles.iteritems() if not v.endswith('nothing')]
    _linestyles.sort()
    for lw in count(1):
        for ls in _linestyles:
            for clr in colors:
                yield lw,ls,clr

Lscale = 2
Mscale = 1.8e10 
Rcut = 50

fig_plot_size = None # NxN inches
fig_nr,fig_nc = None, None 
fig_subplot_index= None
produce_subfiles = None

def init_plots(size, dim, with_subfiles=False):
    global fig_plot_size, fig_nr,fig_nc, fig_subplot_index, produce_subfiles
    fig_plot_size = size
    fig_nr,fig_nc = dim
    fig_subplot_index=1
    produce_subfiles = with_subfiles
    f = figure(figsize=(fig_plot_size*fig_nc, fig_plot_size*fig_nr))

def begin_plot():
    global fig_subplot_index
    if not produce_subfiles:
        gcf().add_subplot(fig_nr,fig_nc,fig_subplot_index)
    fig_subplot_index += 1

def end_plot():
    if produce_subfiles:
        tag = chr(ord('a') + (fig_subplot_index-1))
        savefig('%s%s.png' % (os.path.splitext(state_file))[0], tag)

def PlotFigures():

    for g in gls:
        g.make_ensemble_average()
        g.bw_styles = True


    init_plots(4, [2,4])
    gcf().subplots_adjust(left=0.05, right=0.98)

    gcf().suptitle('%s' % escape(os.path.splitext(os.path.basename(opts[1]))[0]))

#   for g in gls:
#       for i,o in enumerate(g.objects):
#           if hasattr(o, 'stellar_mass'):
#               g.subtract_kappa_from_models(o.stellar_mass, i, include_ensemble_average=False)

    if 1: 
        begin_plot()
        for g in gls:
#           if 'image' in g.meta_info:
#               R = 20 #g.objects[0].basis.maprad
#               #cx,cy = -1.875, 0.08
#               cx,cy=0,0
#               g.image_plot(g.meta_info['image'], R, [cx,cy])
            g.img_plot(obj_index=0)
            g.img_plot(obj_index=1)
            g.arrival_plot(g.ensemble_average, obj_index=0, only_contours=True, clevels=50, colors='r');
            g.arrival_plot(g.ensemble_average, obj_index=1, only_contours=True, clevels=50, colors='b');
            #g.arrival_plot(g.ensemble_average, only_contours=True, clevels=50, src_index=0, colors='r');
            #g.arrival_plot(g.ensemble_average, only_contours=True, clevels=50, src_index=4, colors='g');
            if 1 or 'image' in g.meta_info:
                R = 9/2.,2.7/2. #g.objects[0].basis.maprad
                #cx,cy = -1.875, 0.08
                cx,cy=-0.21,0
                #g.image_plot('brighter_red2_7_rot.png', R, center=[cx,cy])
            #g.src_plot(obj_index=0)
            #g.src_plot(g.ensemble_average, obj_index=0)
            g.external_mass_plot(0)
        end_plot()

    if 1: 
        begin_plot()
        g.H0inv_plot()
        end_plot()

    if 1: 
        begin_plot()
        g.H0_plot()
        end_plot()

    if 1: 
        begin_plot()
        si = style_iterator(colors)
        for g in gls:
            lw,ls,clr = si.next()
            g.glerrorplot('kappa(R)', ['R', 'arcsec'], yscale='linear')
        end_plot()

    if 0: 
        begin_plot()
        si = style_iterator(colors)
        for g in gls:
            lw,ls,clr = si.next()
            g.glerrorplot('kappa(<R)', ['R', 'arcsec'])
        end_plot()

    if 0:
        begin_plot()
        si = style_iterator(colors)
        for g in gls:
            lw,ls,clr = si.next()
            g.glerrorplot('M(<R)', ['R', 'kpc'], yscale='linear')
        end_plot()

    if 0: 
        begin_plot()
        si = style_iterator(colors)
        for g in gls:
            lw,ls,clr = si.next()
            g.glerrorplot('Sigma(R)', ['R', 'kpc'], yscale='linear')
        end_plot()

    if 0:
        for g in gls:
            begin_plot()
            g.kappa_plot(g.ensemble_average, 0, with_contours=True, clevels=20, vmax=1); #Re_plot(env().ensemble_average,0)
            #g.kappa_plot(g.ensemble_average, 0, with_contours=False, vmax=1); #Re_plot(env().ensemble_average,0)
            #g.gradient_plot(g.ensemble_average, 0)
            end_plot()

    if 0:
        for g in gls:
            begin_plot()
            if 'image' in g.meta_info:
                R = 20 #g.objects[0].basis.maprad
                #cx,cy = -1.875, 0.08
                cx,cy=0,0
                g.image_plot(g.meta_info['image'], R, center=[cx,cy])
            s = 0
            if hasattr(g.objects[0], 'stellar_mass'):
                s = g.objects[0].stellar_mass

            g.kappa_plot(g.ensemble_average, 0, with_contours=True, clevels=20, subtract=s) #, vmax=1, colors='r'); #Re_plot(env().ensemble_average,0)
            #g.kappa_plot(g.ensemble_average, 0, with_contours=False, vmax=1); #Re_plot(env().ensemble_average,0)
            #g.gradient_plot(g.ensemble_average, 0)
            end_plot()

    if 1:
        for g in gls:
            begin_plot()
#           if 'image' in g.meta_info:
#               R = 20 #g.objects[0].basis.maprad
#               #cx,cy = -1.875, 0.08
#               cx,cy=0,0
#               g.image_plot(g.meta_info['image'], R, [cx,cy])
            g.kappa_plot(g.ensemble_average, 0, with_contours=True, clevels=20) #, vmax=1, colors='r'); #Re_plot(env().ensemble_average,0)
            #g.kappa_plot(g.ensemble_average, 0, with_contours=False, vmax=1); #Re_plot(env().ensemble_average,0)
            #g.gradient_plot(g.ensemble_average, 0)
            end_plot()

    if 0:
        for g in gls:
            begin_plot()
            g.glhist('N1', label='N1', color='r', xlabel=r'$\theta_E$')
            #g.glhist('N3', label='N3', color='b', xlabel=r'$\theta_E$')
            #g.glhist('N4', label='N4', color='m', xlabel=r'$\theta_E$')
            g.glhist('N2', label='N2', color='g', xlabel=r'$\theta_E$')
            #g.glhist('N5', label='N5', color='c', xlabel=r'$\theta_E$')
            end_plot()

    if 1:
        for g in gls:
            begin_plot()
            g.shear_plot2d()
            end_plot()

    if 1:
        for g in gls:
            begin_plot()
            g.time_delays_plot()
            end_plot()


#-------------------------------------------------------------------------------

ion()
opts = Environment.global_opts['argv']
gls = [loadstate(f) for f in opts[1:]]

if len(gls) == 1:
    colors = 'k'
else:
    colors = 'rgbcm'

PlotFigures()
ioff()

savefig('viewstate.pdf')

show()
#try:
#except:
#    savefig('viewstate.pdf')


