glass_basis('glass.basis.pixels')
include_prior('external_shear', 'external_mass')
import sys
import numpy as np
import pylab as pl
from matplotlib.widgets import Slider, Button, RadioButtons
from glass.misc.raytrace import raytrace, write_code, write_glass_code
from glass.environment import Image, env
from glass.scales import convert

#----------------------------------------------------------------------------
# USER SETTINGS HERE --------------------------------------------------------
#----------------------------------------------------------------------------

# Choose the input file. If 'isothermal' is given, then create an analytic 
# isothermal lens model.

#sim_file='Data/Triaxial/StarsAandHaloA.snap'
sim_file='Data/Triaxial/StarsAandHaloC.snap'
#sim_file='Data/Triaxial/StarsBandHaloB.snap'
#sim_file='Data/Triaxial/StarsBandHaloC.snap'

params = {
    'isothermal'                       : dict(Lscale=None, Mscale=None, Rcut=8, Rmap=8),
    'Data/Triaxial/StarsAandHaloA.snap': dict(Lscale=2, Mscale=1.8e10, Rcut=50, Rmap=50),
    'Data/Triaxial/StarsBandHaloB.snap': dict(Lscale=2, Mscale=1.8e10, Rcut=50, Rmap=50),
    'Data/Triaxial/StarsAandHaloC.snap': dict(Lscale=2, Mscale=1.8e10, Rcut=50, Rmap=10),
    'Data/Triaxial/StarsBandHaloC.snap': dict(Lscale=2, Mscale=1.8e10, Rcut=50, Rmap=10)
}

# The default set of parameters if the simulation name does not appear above in 'params'
default_params = dict(
      Lscale=2         # Length scale. How many kpc in 1 simulation file unit.
    , Mscale=1.8e10    # Mass scale. How many Msun in 1 simulation mass unit.
    , Rcut=50          # Physical 3D radius in kpc to extract from the input.
    , Rmap=50          # Physical 2D radius in kpc of the projected glass pixel map.
)

zL = 0.31       # Default redshift of the lens.
zS = 1.72       # Default redshift of the source.
H0inv = 13.7    # Default Hubble Time.

#----------------------------------------------------------------------------
# NO MORE SETTING BELOW HERE ------------------------------------------------
#----------------------------------------------------------------------------

def make_obj(_zlens, _zsrc, Rmap, units='arcsec'):
    def _make_obj(Rpix):
        env = Environment()
        env.hubble_time(H0inv)
        obj = env.globject('Leicester Data')
        env.zlens(_zlens)
        env.pixrad(Rpix)
        env.subdiv(1)
        env.external_mass(Isothermal(0,0), 4)
        #env.external_mass(PointMass(1,1), 10)
        if Rpix < 0:
            env.hires(1,5)
        if units == 'arcsec':
            env.maprad(Rmap)
        else:
            env.maprad(convert('%s to arcsec' % units, Rmap, obj.dL, convert('H0^-1 in Gyr to nu', H0inv)))
        src = env.source(_zsrc)
        return env

    return _make_obj

def rotate3d(R, angle, axis):
    axes = {'x':0, 'y':1, 'z':2}
    del axes[axis]
    ind = axes.values()
    rot_matrix = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    R[ind] = np.tensordot(rot_matrix(angle), R[ind], axes=1)

def plot_exclude():
    global exclude
    #c = '#FFFFFF'
    c = '#DDDDDD'
    #c = 'g'
    for l,u in exclude:
        pl.axvspan(l,u, fc=c, ec=c)

def withinR(r, R):
    return np.sqrt(np.sum(r**2, 1)) < R

def make_particle_lens(zL, zS, lo, hi, R, units='arcsec'):
    global lores, hires
    global lores_models, hires_models

    sx,sy = 0.0, 0.0

    new_obj = make_obj(zL, zS, R, units=units)

    lores = new_obj(lo)
    lores.model(1, mode='particles', data=[X,Y,M, [[sx,sy]], H0inv])
    lores_models = lores.models

    hires = new_obj(hi)
    hires.model(1, mode='particles', data=[X,Y,M, [[sx,sy]], H0inv])
    hires_models = hires.models

def make_iso_lens(zL, zS, lo, hi, R, units='arcsec'):
    global lores, hires
    global lores_models, hires_models

    sx,sy = 0.0, 0.0
    new_obj = make_obj(zL, zS, R)

    lores = new_obj(lo)
    lores.model(1, mode='isothermal', data=[4, [[sx,sy]], H0inv])
    lores_models = lores.models

    hires = new_obj(hi)
    hires.model(1, mode='isothermal', data=[4, [[sx,sy]], H0inv])
    hires_models = hires.models


def load_data(sim_file):
    if sim_file == 'isothermal':
        make_lens = make_iso_lens
        X,Y,Z,M = [None] * 4
    else:

        try:
            import pynbody as pb
            from pynbody.analysis.halo import center
        except:
            print 'ERROR:'
            print 'lenspick requires the Python library "pynbody" in order to load simulation files.'
            print 'pynbody is available at https://code.google.com/p/pynbody/'
            sys.exit(1)

        print "Loading file", sim_file
        S = pb.load(sim_file)

        if S.has_key('mass'):
            M = S['mass']
        else:
            M  = np.zeros(len(S))
            M[:len(S.dark)] = S.header.mass[1]
            M[len(S.dark):] = S.header.mass[4]
            print 'DM   unit mass', S.header.mass[1]
            print 'Star unit mass', S.header.mass[4]

        M *= sim_params['Mscale']
        R = S['pos'] * sim_params['Lscale']
        w = withinR(R, sim_params['Rcut'])
        R = R[w] #; rotate3d(R, np.pi/4, 'z')
        M = M[w]
        X,Y,Z = R.T

        print 'Total Mass %g Msun' % np.sum(M)
        print '%i particles' % len(M)
        print '%i DM particles' % len(S.dark['pos'])

        make_lens = make_particle_lens

    return X,Y,Z,M,make_lens


lores = None
hires = None
lores_models = None
hires_models = None


_env = None
_last_env = None
_last_model = None
last_nimages = 0

def redraw(env=None, last_model=None):
    global draw_first_time

    fig1 = pl.figure(1)

    if env and last_model:
        global _env,_last_model
        _env,_last_model = env, last_model


    if draw_first_time and _env and _last_model:
        fig1.sca(ax1)
        pl.cla()
        _env.kappa_plot(_last_model[0], 0, with_colorbar=False)
        #hires.kappa_plot(hires_models[0], 0, with_colorbar=False)
        draw_first_time = False


    if ptsx and ptsy:
        fig1.sca(ax1)
        x1,x2=pl.gca().get_xlim()
        y1,y2=pl.gca().get_ylim()
        if ptsx[:-1]:
            pl.scatter(ptsx[:-1], ptsy[:-1], s=10)
        pl.scatter([ptsx[-1]], [ptsy[-1]], s=10, c='r')
        pl.gca().set_xlim(x1,x2)
        pl.gca().set_ylim(y1,y2) 


    if _env and _last_model:
        fig1.sca(ax3)
        pl.cla()
        #_env.grad_tau(_last_model[0], 0, 'x', 0)
        #_env.grad_tau(_last_model[0], 0, 'y', 0)
        _env.glerrorplot('kappa(<R)', ['R', 'kpc'], mark_images=True, yscale='linear')

        fig1.sca(ax2)
        pl.cla()
        #_env.srcdiff_plot(_last_model[0], with_colorbar=False)
        #_env.glerrorplot('kappa(<R)', ['R', 'kpc'], mark_images=True, yscale='linear')
        #potential_plot(last_model[0], 0, 0, with_colorbar=False)
        #kappa_plot(last_model[0], 0, 0, with_colorbar=False)
        _env.arrival_plot(_last_model[0], only_contours=True) #, with_colorbar=True) #, clevels=100)
        _env.img_plot(with_guide=True, tight=True)
        fig1.sca(ax6)
        pl.cla()
        _env.glerrorplot('Sigma(R)', ['R', 'kpc'], mark_images=True, yscale='linear')
        #_env.glerrorplot('kappa(R)', ['R', 'kpc'], mark_images=True, yscale='linear')

    pl.draw()

ptsx = []
ptsy = []

initial_guess = None
use_fixed = True
def onclick(event):
    global initial_guess, last_model, last_nimages

    if event.inaxes != ax1.axes: return
    if event.key not in ['shift', 'control']: return

    if not use_fixed or not (ptsx or ptsy):
        ptsx.append(event.xdata)
        ptsy.append(event.ydata)

    if 1:
        if event.key == 'shift':
            env = hires
            models = hires_models

        if event.key == 'control':
            initial_guess = None
            env = lores
            models = lores_models

        x,y = event.xdata, event.ydata

        if use_fixed:
            x,y = fixed_xy

        env.change_source([[complex(x,y)]])

    else:

        if event.key == 'shift':   
            obj = make_obj(35)

        if event.key == 'control': 
            initial_guess = None
            obj = make_obj(8); 

        ps = f(obj, src=[], H0inv=H0inv)

    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            event.button, event.x, event.y, x, y)

    src = models[0]['obj,data'][0][0].sources[0]
    obj,ps = models[0]['obj,data'][0]

    d = raytrace([obj,ps,0], initial_guess = initial_guess, nimgs=5)
    print d
    print '='*80
    print 'GLASS Input'
    print '='*80
    print "SOURCE AT  %.2f, [%9.15f, %9.15f]" % (obj.sources[0].z, x,y)
    write_code(models[0], 0,0, d) #, simple=True)
    write_glass_code(models[0], 0,0, d) #, simple=True)
    print

    s = complex(x, y)
    iso_mass = 4
    i1 = s + iso_mass * s/abs(s) / src.zcap / np.pi
    i2 = s - iso_mass * s/abs(s) / src.zcap / np.pi
    at1 = src.zcap * abs(i1 - s)**2 / 2 - iso_mass*abs(i1) / np.pi
    at2 = src.zcap * abs(i2 - s)**2 / 2 - iso_mass*abs(i2) / np.pi
    print 'isothermal images at', i1, i2
    print 'arrival times', at1, at2
    print 'time delay', (at2 - at1) * obj.dL * (1+slider_zL.val) / env.nu * 365.25
    #print 'time delay', (at2 - at1) * slider_zL.val * (1+slider_zL.val) * 80 / H0inv * 10

    src.images = []
    last_nimages = 0
    for img,t,_,parity in d:
        src.add_image(Image((img.real, img.imag), parity))
        last_nimages += 1

    initial_guess = None
    redraw(env, models)

def onpress(event):
    global ptsx, ptsy

    if event.key == 'x':
        ptsx = []
        ptsy = []
        redraw()

    if event.key == 'c':
        for models in [lores_models, hires_models]:
            src = models[0]['obj,data'][0][0].sources[0]
            src.images = []
        redraw()

    if event.key == 'u':
        if ptsx: del ptsx[-1]
        if ptsy: del ptsy[-1]
        for models in [lores_models, hires_models]:
            src = models[0]['obj,data'][0][0].sources[0]
            if src.images: src.images = src.images[:-last_nimages]
        redraw()

if __name__ == '__main__':

    help_text =\
    """
    lenspick - Create gravitational lenses from projected simulation data.

    Lenspick loads particle data from a simulation file and projects it onto a
    mass map that is used to calculate the image positions of a lensed source.

    The parameters for the simulation to load must be specified in lenspick.py
    Some parameters such as the source/object redshift, rendering resolutions,
    and size of the mass map are adjustable from the interface.

    Usage:
        1) Modify lenspick.py to select the simulation to load and the scaling
           parameters.
        2) Run lenspick using GLASS: "run_glass -t Nthreads lenspick.py", where
           Nthreads is the number of CPU threads to use.  Specifying more
           threads than available CPU cores will slow down the calculation time.
        3) Adjust the available lens parameters using the interface.
        4) Click "Make lens" to create a lens model using the selected
           parameters. The plots will not update.
        5) Click on the mass map using one of the modifier keys listed below to
           place a source and calculate the image positions.

    Commands:
        x               Clear the current set of source points
        u               Undo
        Control-click   Places a source at the clicked position on the mass map.
                        Calculates the images using a low resolution mass map.
        Shift-click     Places a source at the clicked position on the mass map.
                        Calculates the images using a high resolution mass map.
    """

    print help_text

    fixed_xy = .001,0
    sim_params = params.get(sim_file, default_params)
    X,Y,Z,M,make_lens = load_data(sim_file)

    pl.ion()
    fig1 = pl.figure(figsize=(16,8))
    pl.suptitle(sim_file)
    pl.subplots_adjust(left=0.05, top=0.95, right=0.95)
    ax1 = fig1.add_subplot(241)
    ax2 = fig1.add_subplot(242)
    ax3 = fig1.add_subplot(243)
    #ax4 = fig1.add_subplot(264)
    #ax5 = fig1.add_subplot(265)
    ax6 = fig1.add_subplot(244)

    draw_first_time = True

    #kappa_plot3d(hires_model, 0)

    #fig1.sca(ax4)
    #lores.deflect_plot(lores_models[0],0, 'x', 0)
    #fig1.sca(ax5)
    #lores.deflect_plot(lores_models[0],0, 'y', 0)

    cid = fig1.canvas.mpl_connect('button_press_event', onclick)
    cid = fig1.canvas.mpl_connect('key_press_event', onpress)

    fig1.sca(ax6)
    #hires.glplot('Sigma(R)', ['R', 'arcsec'])
    #xs = np.linspace(0.1714/2, 7, 100)
    #ys = 0.5*4/xs 
    #semilogy(xs,ys, 'r-')

    rax = pl.axes([0.025, 0.2, 0.15, 0.10])
    radio = RadioButtons(rax, ('Use fixed point', 'Free selection'), active=0)
    def onradio(label):
        global use_fixed
        use_fixed = (label == 'Use fixed point')
    radio.on_clicked(onradio)

    rax = pl.axes([0.025, 0.1, 0.15, 0.05])
    slider_zL = Slider(rax, 'zL', 0, 2, zL)
    rax = pl.axes([0.025, 0.05, 0.15, 0.05])
    slider_zS = Slider(rax, 'zS', 0, 2, zS)

    rax = pl.axes([0.4, 0.1, 0.15, 0.05])
    slider_lo = Slider(rax, 'lo res', 3, 130, 10, valfmt='%i')
    rax = pl.axes([0.4, 0.05, 0.15, 0.05])
    slider_hi = Slider(rax, 'hi res', 3, 130, 90, valfmt='%i')

    rax = pl.axes([0.4, 0.2, 0.15, 0.05])
    slider_R = Slider(rax, 'Rmap', 0, sim_params['Rcut'], sim_params['Rmap'])
    #slider_R = Slider(rax, 'R', 0, Rcut, 6.7578)

    def on_mklens(event):
        global draw_first_time
        if slider_zL.val >= slider_zS.val:
            print 'zL must be strictly less than zS'
        else:
            make_lens(slider_zL.val, slider_zS.val, int(slider_lo.val), int(slider_hi.val), slider_R.val, units='kpc')
            draw_first_time = True
            redraw()
            print 
            print 'Model ready for lensing.'
            print 

    rax = pl.axes([0.20, 0.05, 0.15, 0.05])
    button_mklens = Button(rax, 'Make lens')
    button_mklens.on_clicked(on_mklens)

    def on_setfixed(event):
        global fixed_xy
        if ptsx and ptsy:
            fixed_xy = ptsx[-1], ptsy[-1]
            print 'Set fixed point to', fixed_xy

    rax = pl.axes([0.20, 0.2, 0.15, 0.05])
    button_setfixed = Button(rax, 'Set fixed point')
    button_setfixed.on_clicked(on_setfixed)
    pl.show()
