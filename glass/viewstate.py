from __future__ import division

glass_basis('glass.basis.pixels', solver=None)
#setup_log(stdout=False)
exclude_prior(
              'smooth_symmetry',
              #'PLsmoothness',
              #'smoothness',
              #'J1parity',
              #'time_delay',
              #'hubble_constant',
              #'annular_density',
              #'external_shear',
              #'profile_steepness',
              #'J2gradient',
              #'symmetry'
              )


from basis.pixels import init_model_generator
import numpy as np
import numpy.linalg as la
import os
from numpy import load
from matplotlib import rc
import pylab as pl
from pylab import over, scatter, show, figure, close, title, ion, draw, plot, savefig, gcf
#from plots import mass_plot, potential_plot, sigma_plot, H0_plot, encmass_plot, arrival_plot, src_plot, img_plot
#from plots import kappa_plot, sigp_plot
from misc.raytrace import raytrace, write_code, check_model_magnifications, magnification_filter

#from basis.pixels.funcs import estimated_Re
from environment import Image

def escape(s):
    s = s.replace('_', r'\_')
    return s

ion()

load_sim = False
#load_sim = True

opts = Environment.global_opts['argv']

if len(opts) > 1:
    state_file = opts[1]
    opts = opts[1:]

if len(opts) > 2:
    sim_file = opts[1]
    load_sim = True

gls = loadstate(state_file)

Lscale = 2
Mscale = 1.8e10 
Rcut = 50

def withinR(r, R):
    return np.sqrt(np.sum(r**2, 1)) < R

if load_sim:
    env1 = env()
    obj = env().models[0]['obj,data'][0][0]
    #pr = env().models[0]['obj,data'][0][0].basis.pixrad
    #mr = env().models[0]['obj,data'][0][0].basis.maprad
    #srcz = env().objects[0].sources[0].z
    #env().clear()

    import pynbody as pb
    from pynbody.analysis.halo import center
    print "Loading file", sim_file
    S = pb.load(sim_file)
    print "File loaded."

    def make_obj_like(obj, X,Y,M,H0inv, src_index=0):
        hubble_time(H0inv)
        o = globject('Leicester Data')

        if 0:
            zlens(0.31)
            pixrad(12)
            maprad(6, units='kpc')
        else:
            zlens(obj.z)
            pixrad(5*obj.basis.pixrad)
            maprad(obj.basis.maprad)
        #maprad(13.67)

        steepness(0, None)

        #SOURCE AT  -0.372441055614436 0.079649115439940
        #A = (-3.178988717310369, 10.034994575243555)
        #B = (-3.412743684249515,-9.857930040562394)
        #C = (-11.195866318770571,-0.676685702993670)
        #D = ( 10.629660495542433,-0.459675924352989)
        #E = ( 0.023372816146500,-0.006770151392341)
        #source(2.50, A, 'min', B, 'min', 38.125200271034608, C, 'sad', 85.619880982188278, D, 'sad', 209.078327318320220) #, E, 'max', 1364.605340662779099)


        #SOURCE AT  0.084448766188927 -0.514624450226698
        #A = ( 0.365350271754632,-7.143280956325841)
        #B = ( 0.816440372659059, 6.311702299228262)
        #C = ( 6.308858287199853, 2.003479710377767)
        #D = (-6.241124403270530, 1.823265225374992)
        #E = (-0.039000928735765, 0.071430904578305)
        #source(0.60, A, 'min', B, 'min', 330.286063666193741, C, 'sad', 67.244072948216015, D, 'sad', 52.993243712330532) #, E, 'max', 640.339251483868907)

        source(2.5)

        s1 = -0.372441055614436,  0.079649115439940
        s2 =  0.084448766188927, -0.514624450226698
        model(1, mode='particles', data=[X,Y,M, [s1], H0inv])

        #model(1, X,Y,M,H0inv, mode='particles')

        #for img in obj.sources[src_index].images:
        #    img = img.pos
        #    src.add_image(Image((img.real, img.imag), 'min'))


        #model(0)
        #mg = env().model_gen
        #mg.start()

        #print env().solutions[-1]
        #print type(env().solutions[-1])
        #print id(env().solutions[-1])
        #print '!!!', mg.in_simplex(env().solutions[-1][1:])

        #assert 0
        return o

    new_env()
    if S.has_key('mass'):
        Mboth = S['mass']
    else:
        Mboth  = np.zeros(len(S))
        Mboth[:len(S.dark)] = S.header.mass[1]
        Mboth[len(S.dark):] = S.header.mass[4]
        #print R.header.mass
    Rboth = S['pos'] * Lscale
    Mboth *= Mscale
    w = withinR(Rboth, Rcut)
    Rboth = Rboth[w]
    Mboth = Mboth[w]
    X,Y,Z = Rboth.T
    hires_obj = make_obj_like(obj, X,Y,Mboth, 13.7)
    hires_model = env().models

    new_env()
    if S.has_key('mass'):
        Mdark = S.dark['mass']
    else:
        Mdark  = np.zeros(len(S.dark))
        Mdark[:] = S.header.mass[1]
    Rdark = S.dark['pos'] * Lscale
    Mdark *= Mscale
    w = withinR(Rdark, Rcut)
    Rdark = Rdark[w]
    Mdark = Mdark[w]
    X,Y,Z = Rdark.T
    make_obj_like(obj, X,Y,Mdark, 13.7)
    dark_model = env().models

    new_env()
    Rstar = S.star['pos'] * Lscale
    if S.has_key('mass'):
        Mstar = S.star['mass']
    else:
        Mstar  = np.zeros(len(S.star))
        Mstar[:] = S.header.mass[4]
    Mstar *= Mscale
    w = withinR(Rstar, Rcut)
    Rstar = Rstar[w]
    Mstar = Mstar[w]
    X,Y,Z = Rstar.T
    make_obj_like(obj, X,Y,Mstar, 13.7)
    stellar_model = env().models

    set_env(env1)

elif 0:
    env1 = env()

    obj = env().models[0]['obj,data'][0][0]
    def make_obj(_zlens, _zsrc, Rmap):
        def _make_obj(Rpix):
            new_env()
            hubble_time(13.7)
            o = globject('Isothermal Analytic')
            zlens(_zlens)
            pixrad(Rpix)
            subdiv(1)
            maprad(obj.basis.maprad)
            steepness(0.0, 2.0)
            #smooth(factor=2, include_central_pixel=False)
            src = source(_zsrc)
            return o
        return _make_obj

    new_obj = make_obj(0.5, 2.5, 8)

    hires_obj = new_obj(obj.basis.pixrad)
    model(1, mode='isothermal', data=[1, [[0,0]], 13.7])
    hires_model = env().models
    env2 = env()

    new_obj(obj.basis.pixrad)
    model(0)
    mg = env().model_gen

    print id(env1.solutions[-1])
    print '!!!', mg.in_simplex(env1.solutions[-1][1:])

    set_env(env1)

elif 0:
    env1 = env()

    obj = gls.models[0]['obj,data'][0][0]
    def make_obj(_zlens, _zsrc, Rmap):
        def _make_obj(Rpix):
            new_env()
            hubble_time(13.7)
            o = globject('Isothermal Analytic')
            zlens(_zlens)
            pixrad(Rpix)
            subdiv(1)
            maprad(obj.basis.maprad)
            steepness(0.0, 2.0)
            #smooth(factor=2, include_central_pixel=False)
            #src = source(_zsrc)
            A = ( 0.655391830063660,-0.411106804865330)
            B = (-0.311024708127276, 0.195212314914944)
            src = source(1.50, A, 'min', B, 'sad', 71.584011478768957)

            return o
        return _make_obj

    #new_obj = make_obj(0.5, 2.5, 8)

    #hires_obj = new_obj(obj.basis.pixrad)
    #model(1, mode='isothermal', data=[1, [[0,0]], 13.7])
    #hires_model = env().models
    #env2 = env()

    new_obj = make_obj(obj.z, obj.sources[0].z, -1)
    new_obj(obj.basis.pixrad)
    model(0)
    mg = env().model_gen

    #print id(env1.solutions[-1])
    #print '!!!', mg.in_simplex(env1.solutions[-1][1:])

    set_env(env1)

if 0:
    for m in env().models:
        obj,ps = m['obj,data'][0]
        ps['Re'] = estimated_Re(obj,ps,0)

#shear_plot()
#glerrorplot(env().models, 'kappa(R)', ['R', 'arcsec'], mark_images=True)
#if load_sim: glplot(hires_model, 'kappa(R)', ['R', 'arcsec'], hilite_model=0, mark_images=False)
#savefig('xxx.png')
#sys.exit(0)

if 0:
    env().models = env().models[::10]

    #post_process(magnification_filter)
    #apply_filters()
    bad=0
    for i,m in enumerate(env().models[7:]):
        print 'model', i
        m['accepted'] = check_model_magnifications(m)
        if not m['accepted']:
            print '** Rejected **'
            bad+=1
            env().models = [m]
            #break

    print '%i/%i Rejected' % (bad, len(env().models))

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

def TestFigures():

    gls.make_ensemble_average()

    if 0:
        name = os.path.split(state_file)[1]
        with open('chi2_data.py', 'a+') as f:
            print >>f, "'%s': [%e,%e,%e,%e,%e]," % (name, 
                                         0, #ensemble_mass_rms(env().models, hires_model[0]),
                                         kappa_chi2(env().models, hires_model[0]),
                                         kappa_profile_chi2(env().models, hires_model[0]),
                                         kappa_chi2([env().ensemble_average], hires_model[0]),
                                         kappa_profile_chi2([env().ensemble_average], hires_model[0]),
                                         #time_delay_chi2([env().ensemble_average], hires_model[0]),
                                         )

#   print '*'*80
#   print '*'*80
#   Q= kappa_profile_chi2([env().ensemble_average], hires_model[0])
#   print Q, np.log(Q)
#   print '*'*80
#   print '*'*80

    init_plots(4, [2,4])
    gcf().subplots_adjust(left=0.05, right=0.98)

    gcf().suptitle('%s' % escape(os.path.splitext(os.path.basename(state_file))[0]))

    env().bw_styles = True

    if 1: 
        begin_plot()
        img_plot(gls.ensemble_average,0)
        #Re_plot(env().models,0)
        gls.arrival_plot(gls.ensemble_average, only_contours=True, clevels=50);
        for m in gls.models:
            src_plot(m,0,0)
        end_plot()

    if 0: 
        begin_plot()
        H0_plot         (env().models)
        end_plot()

#assert 0

    if 1: 
        begin_plot()
        gls.glerrorplot('kappa(R)', ['R', 'arcsec'], mark_images=True)
        end_plot()

    if 1: 
        begin_plot()
        gls.glerrorplot('kappa(<R)', ['R', 'arcsec'], mark_images=True)
        end_plot()

    if 1:
        begin_plot()
        gls.glerrorplot('M(<R)', ['R', 'kpc'], mark_images=True, every=1, yscale='linear')
        end_plot()

    if 0: 
        begin_plot()
        gls.glerrorplot('Sigma(R)', ['R', 'kpc'], yscale='linear')
        end_plot()

    if 0: 
        begin_plot()
        time_delays_plot(src_index=0)
        end_plot()

    if 0 and load_sim: 
        begin_plot()
        radial_chi2_plot([env().ensemble_average], hires_model[0])
        #radial_chi2_plot(env().models, hires_model[0])
        pl.ylim(-10,0)
        end_plot()


    if 0: figure(); sigma_plot      (env().models, xaxis='R', mark_images='arcsec')
#if load_sim: figure(); kappa_prof_plot (env().models, mark_images='arcsec')

    if 1:
        begin_plot()
        gls.kappa_plot(gls.ensemble_average, 0, with_contours=False); #Re_plot(env().ensemble_average,0)
        end_plot()

    if 0 and load_sim:
        begin_plot()
        grad_kappa_plot(env().ensemble_average, 0, which='x', with_contours=False)
        end_plot()
        begin_plot()
        grad_kappa_plot(env().ensemble_average, 0, which='y', with_contours=False)
        end_plot()

    if 0 and load_sim:
        begin_plot()
        #kappa_compare_plot(env().models, env().models[10], 0, sort=False)
        kappa_residual_grid_plot(env().ensemble_average, hires_model[0], 0, only_contours = False, with_colorbar=True)
        end_plot()

    if 0 and load_sim:
        begin_plot()
        #kappa_compare_plot(env().models, env().models[10], 0, sort=False)
        kappa_compare_plot(env().models, hires_model[0], 0, sort=False, normalize=True)
        end_plot()


#if 1: figure(); encmass_plot    (env().models, mark_images=False)
#if 1: figure(); H0_plot         (env().models, obj_index=1)

    if 0:
        figure()
        plot(hi_ps2['R'], hi_ps['sigma'] / hi_ps2['sigma'], 'k-')
        plot(hi_ps2['R'], hi_ps['sigma'] / dm_ps['sigma'], 'r-')
        #plot(hi_ps2['R'], hi_ps['sigma'] / gs_ps['sigma'], 'g-')

    if 0:
        f=figure()
        f.add_subplot(121, yscale="log")
        #kappa_compare_plot(env().models, env().ensemble_average, 0)
        kappa_compare_plot(env().models, hires_model, 0, mark='all')
        f.add_subplot(122)
        kappa_compare_grid_plot(env().models, hires_model, 0)
        img_plot(hires_model,0,0)

if 1:
    TestFigures()
    if not produce_subfiles:
        savefig('%s.png' % (os.path.splitext(state_file))[0])
    show()
    sys.exit(0)


f = figure(); 

for i,m in enumerate(env().models):
    src_index = 0

    #obj,ps = m['obj,data'][0]

    #print '&' * 80
    #print check_model_magnifications([m,0])
    #print '&' * 80

    #print ps['kappa']

    #d = raytrace(obj,ps,0)
#   images = [x.pos for x in obj.sources[0].images]
#   print images
#   #images = [-0.22305148599874872-0.55762871499687183j]
#   images = None
#   d = raytrace(obj,ps,0, initial_guess = images)
#   write_code(obj, ps, 0, d)

#   del obj.sources[0].images[:]
#   for img,t,_ in d:
#       obj.sources[0].add_image(Image((img.real, img.imag), 'unk'))

    print "Plotting model %i/%i" % (i,len(env().models))
    #print 'H0inv =', m['obj,data'][0][1]['1/H0']
    f.clear()
    f.add_subplot(121); kappa_plot(m,0, with_colorbar=False); 
    arrival_plot(m); 
    img_plot(m,0,0); 
    src_plot(m,0,0); 

    if 0:
        img_plot(hires_model,0,0, color='g')
        Re_plot(m,0)

    if 0:
        f.add_subplot(122); kappa_plot(hires_model,0); 
        img_plot(hires_model, 0,0, color='g')
    #figure(); kappa_plot(hires_model,0)
    #figure(); arrival_plot(m, 0, 0); img_plot(m, 0,0)
    #arrival_plot(hires_model, 0, 0)

    f.suptitle('Model %i/%i - %s' % (i,len(env().models), ['Rejected', 'Accepted', 'No filter'][m.get('accepted',2)]))
    draw() #show()
    raw_input()

    continue
    
    good = True
    for src_index,src in enumerate(obj.sources):
        for img_index, img in enumerate(src.images):
            xx,yy,xy,e = obj.basis.maginv(m['obj,data'][0][1])[src_index][img_index]
            A = e[0]

            g = False
            if img.parity_name == 'sad':
                g= A[0,0] * A[1,1] > A[0,1]**2
            elif img.parity_name == 'min':
                g= A[0,0] * A[1,1] < A[0,1]**2 and A[0,0] > 0
            elif img.parity_name == 'max':
                g= A[0,0] * A[1,1] < A[0,1]**2 and A[0,0] < 0

            #good = good and g
            #if not good: break
        
    if good:
        for src_index,src in enumerate(obj.sources):
            for img_index, img in enumerate(src.images):
                xx,yy,xy,e = obj.basis.maginv(m['obj,data'][0][1])[src_index][img_index]
                A = e[0]
                print src_index, img_index, img.parity_name, img.pos
                #print src_index, img.pos, xx,yy,xy
                #print e[0]
                #print e[1]
                print 'eval', e[5]
                print 'evec', e[6]
                print 'P', e[7]
                print 'D', e[1]
                print 'A', e[0]
                print 'alpha, beta,delta,ratio', e[2:5], e[8]
                print "----------------"

#       figure(); 
#       arrival_plot(m['obj,data'][0], src_index); 
#       img_plot(m['obj,data'][0], src_index); 
#       title('Source %i' % src_index)
#       show()

    show()
    raw_input()
    close('all')

