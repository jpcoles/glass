from __future__ import division

glass_basis('glass.basis.pixels', solver=None)
exclude_all_priors()

import os
import pylab as pl
from pylab import show, figure, ion, savefig, gcf
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

    #gcf().suptitle('%s' % escape(os.path.splitext(os.path.basename(state_file))[0]))

    if 1: 
        begin_plot()
        for g in gls:
            #img_plot(g.ensemble_average,0,1)
            g.img_plot(obj_index=0)
            g.arrival_plot(g.ensemble_average, only_contours=True, clevels=550);
            #g.arrival_plot(g.ensemble_average, only_contours=True, clevels=20, src_index=1);
            g.src_plot(g.ensemble_average, obj_index=0)
            g.external_mass_plot(0)
            #g.arrival_plot(g.ensemble_average, only_contours=True, clevels=150, src_index=1);
        end_plot()

    if 0: 
        begin_plot()
        H0_plot         (env().models)
        end_plot()

    if 1: 
        begin_plot()
        si = style_iterator(colors)
        for g in gls:
            lw,ls,clr = si.next()
            g.glerrorplot('kappa(R)', ['R', 'arcsec'])
        end_plot()

    if 1: 
        begin_plot()
        si = style_iterator(colors)
        for g in gls:
            lw,ls,clr = si.next()
            g.glerrorplot('kappa(<R)', ['R', 'arcsec'])
        end_plot()

    if 1:
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

    if 1:
        for g in gls:
            begin_plot()
            g.kappa_plot(g.ensemble_average, 0, with_contours=True, clevels=80, vmax=1); #Re_plot(env().ensemble_average,0)
            #g.kappa_plot(g.ensemble_average, 0, with_contours=False, vmax=1); #Re_plot(env().ensemble_average,0)
            #g.gradient_plot(g.ensemble_average, 0)
            end_plot()

    if 0:
        for g in gls:
            begin_plot()
            g.glhist('N1')
            g.glhist('N2')
            g.glhist('N3')
            g.glhist('N4')
            g.glhist('N5')
            end_plot()

    if 1:
        for g in gls:
            begin_plot()
            g.shear_plot()
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
show()

