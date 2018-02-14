from __future__ import division
from glass.command import command
#from solvers.samplex.samplex import Samplex
#from solvers.samplexsimple.samplex import Samplex
#from solvers.lpsolve.samplex import Samplex
import numpy as np
from itertools import izip
from basis import PixelBasis as basis_class
from glass.scales import convert
from glass.exceptions import GLInputError
from glass.utils import dist_range
from . funcs import default_post_process

@command
def minsteep(*args, **kwargs): raise GLInputError("minsteep not supported. Use steepness().")
@command
def maxsteep(*args, **kwargs): raise GLInputError("maxsteep not supported. Use steepness().")
@command
def dgcone(*args, **kwargs): raise GLInputError("dgcone not supported. Use global_gradient().")

#def _foo(options):
   #return Samplex(**options)
   #return Samplex(nvars, nthreads=env.ncpus) 


@command
def globject(env, name):
    co = env.new_object(name)
    co.basis = basis_class()
    #env.model_gen_factory = _foo
    return co

@command
def pixrad(env, r):
    env.current_object().basis.pixrad = r

@command
def priors(env, *ps):
    env.current_object().basis.prior_list = ps

@command
def subdiv(env, n):
    n = int(n)
    if (n%2!=1): raise GLInputError("subdiv: n must be odd")
    env.current_object().basis.subdivision = n

@command
def hires(env, r, refine=1):
    if not (r > 0 and refine>=3 and refine%2==1): raise GLInputError('hires: Minimum refinement value is 3. Must be odd too.')
    env.current_object().basis.hiresR       = r
    env.current_object().basis.hires_levels = refine
    
@command
def smooth(env, factor=2, L=None, include_central_pixel=None):
    #if not prior_included('PLsmoothness3'): raise GLInputError("The 'PLsmoothness3' prior must be included to enable the 'smooth()' command.")
    o = env.current_object()
    #o.prior_options['smoothness'] = {}
    o.prior_options['smoothness']['factor'] = factor
    if include_central_pixel is not None:
        o.prior_options['smoothness']['include_central_pixel'] = include_central_pixel
    if L is not None:
        o.prior_options['smoothness']['L'] = L

@command
def steepness(env, lb, ub):
    if not prior_included('profile_steepness'): raise GLInputError("The 'profile_steepness' prior must be included to enable the 'steepness()' command.")
    o = env.current_object()
    o.prior_options['steepness'] = [lb, ub]

@command
def kann(env, theta):
    if not prior_included('annular_density'): raise GLInputError("The 'annular_density' prior must be included to enable the 'kann()' command.")
    o = env.current_object()
    o.prior_options['annular_density'] = theta

@command
def global_gradient(env, theta):
    if not (0 < theta <= 90): raise GLInputError("dgcone: need 0 < theta <= 90")
    o = env.current_object()
    o.prior_options['gradient'] = np.radians(90-theta)

@command
def local_gradient(env, theta=None, L=None):
    o = env.current_object()
    #if not prior_included('J3gradient'): raise GLInputError("The 'J3gradient' prior must be included to enable the 'local_gradient()' command.")

    if theta is not None: 
        if not (0 < theta <= 90): raise GLInputError("local_gradient: need 0 < theta <= 90")
        o.prior_options['J3gradient']['theta'] = theta
        o.prior_options['J2Gradient']['theta'] = theta

    if L is not None: o.prior_options['J3gradient']['size']  = L
    if L is not None: o.prior_options['J2Gradient']['size']  = L

@command
def min_kappa(env, v):
    o = env.current_object()
    o.prior_options['min_kappa']['kappa'] = v

@command
def min_annular_density(env, v):
    o = env.current_object()
    o.prior_options['min_annular_density']['v'] = v

@command
def min_kappa_particles(env, X,Y,M,H0inv):

    o = env.current_object()
    o.prior_options['min_kappa_particles']['particles'] = [X,Y,M]
    o.prior_options['min_kappa_particles']['H0inv'] = H0inv
    o.prior_options['min_kappa_particles']['nu'] = convert('H0^-1 in Gyr to nu', H0inv)

@command
def minkappa_from_model(env, model, obj_index):
    #assert len(env.g) == 1

    env.current_object().basis.min_kappa_model = model['obj,data'][obj_index][1]['kappa']

@command
def central_pixel_maximum(env, M,H0inv):

    o = env.current_object()
    o.prior_options['central_pixel_maximum']['M'] = M
    o.prior_options['central_pixel_maximum']['H0inv'] = H0inv
    o.prior_options['central_pixel_maximum']['nu'] = convert('H0^-1 in Gyr to nu', H0inv)

@command
def savestate_PixeLens(env, fname):
    obj0 = env.objects[0]
    pr = obj0.basis.pixrad
    w = pr*2 + 1
    pmap = obj0.basis._to_grid(range(1,len(obj0.basis.pmap)+1))

    with open(fname, 'w') as f:
        print >>f, '#BEGIN INPUT'

        for obj in env.objects:
            print >>f, '''\
object %(objname)s
pixrad %(pixrad)i
maprad %(maprad)f
zlens %(zlens).2f
models %(models)i
g %(g).2f
cosm %(om).2f %(ol).2f''' % { \
            'objname': obj0.name,
            'pixrad':obj0.basis.pixrad, 
            'maprad':obj0.basis.maprad,
            'zlens': obj0.z,
            'models': len(env.models),
            'g': convert('nu to H0^-1 in Gyr', env.nu[0]),
            'om':env.omega_matter, 
            'ol':env.omega_lambda,
             }

            for src in obj0.sources:
                print >>f, 'multi %i %.2f' % (len(src.images), src.z)
                for img in src.images:
                    print >>f, '% 12.12e % 12.12e %i' % (img.pos.real, img.pos.imag, img.parity+1)

        #print >>f, env.input_file
        print >>f, '#END INPUT'

        print >>f, '#BEGIN PMAP'
        print >>f, '%i %i' % (pr,pr)
        for i in range(w):
            for j in range(w):
                print >>f, '%3i' % pmap[i,j], 
            print >>f
        print >>f, '#END PMAP'

        print >>f, '#BEGIN ENSEM'
        for m in env.models:
            print >>f, '#BEGIN MODEL'
            for d in m['sol'][1:]:
                print >>f, '%.15g' % d
            print >>f, '#END MODEL'

        print >>f, '#END ENSEM'


@command
def savestate_misc(env, fname):
    with open(fname, 'w') as f:
        for m in env.models:
            for d in m['sol'][1:]:
                print >>f, '%.15g' % d,
            print >>f

@command
def leier_grid(env, fname, size, **kwargs):
    units=kwargs.get('units', 'arcsec')
    scale=kwargs.get('scale', 1.0)
    error=kwargs.get('error', 0)
    o = env.current_object()
    o.prior_options['minkappa Leier grid']['filename'] = fname
    o.prior_options['minkappa Leier grid']['grid size'] = size
    o.prior_options['minkappa Leier grid']['grid size units'] = units
    o.prior_options['minkappa Leier grid']['scale'] = scale
    o.stellar_mass_error = error


@command
def extended_source_size(env, size):
    o = env.current_object()
    o.prior_options['extended source size'] = size


@command
def subtract_kappa_from_models(env, a, obj_index=0, include_ensemble_average=True):
    for m in env.models:
        m['obj,data'][obj_index][1]['kappa'] -= a * m['obj,data'][obj_index][1]['sm_error_factor']
        default_post_process(m['obj,data'][obj_index])

    if include_ensemble_average and hasattr(env, 'ensemble_average'):
        env.ensemble_average['obj,data'][obj_index][1]['kappa'] -= a * env.ensemble_average['obj,data'][obj_index][1]['sm_error_factor']
        default_post_process(env.ensemble_average['obj,data'][obj_index])


def _shape(kappa,r):
    from scipy.linalg import eig, inv

    I = np.zeros((2,2))
    I[0,0] =  np.sum(kappa * r.imag**2)
    I[1,1] =  np.sum(kappa * r.real**2)
    I[0,1] = -np.sum(kappa * r.imag * r.real)
    I[1,0] =  I[0,1]

    V,D = eig(inv(I))
    if V[0] < V[1]:
        np.roll(V,1)
        np.roll(D,1,0)

    # flip vectors to between on the right half-plane
    if D[0,0] < 0: D[:,0] *= -1
    if D[0,1] < 0: D[:,1] *= -1

    #import sys
    theta = np.arctan2(D[1,0], D[0,0])
    #print >>sys.stderr, "THETA", D[0,0], D[1,0], theta
    assert -np.pi/2 <= theta <= np.pi/2
    phi   = np.arctan2(D[1,1], D[0,1])

    return theta,phi


@command
def shape_chi(env, models, model0, frac='1sigma'):
    ns, ds = [[],[]], [[],[]]

    for m in models:
        nT,dT = 0,0
        nP,dP = 0,0
        for m1,m2 in izip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            rs = [ abs(img.pos) for src in obj.sources for img in src.images]
            rmin, rmax = np.amin(rs), np.amax(rs)
            w = (abs(obj.basis.ploc) >= obj.basis.top_level_cell_size * 0.9) * (abs(obj.basis.ploc) <= (rmax+ obj.basis.top_level_cell_size * 0.5))

            t0,p0 = _shape(data0['kappa'][w], obj0.basis.ploc[w])
            t,p   = _shape(data['kappa'][w],  obj.basis.ploc[w])

            #dTheta = lambda t0,t1 = (t0-t1) if np.abs(t0-t1) < np.pi else 2*np.pi-(t0-t1)

            nT += (t-t0)**2
            nT = (t-t0)
            dT += t0**2

            nP += (p-p0)**2
            nP = (p-p0)
            dP += p0**2

        ns[0].append(nT)
        ds[0].append(dT)
        ns[1].append(nP)
        ds[1].append(dP)

    return dist_range(np.array(ns[0]), frac), dist_range(np.array(ns[1]), frac)
    #return dist_range(np.array(ns[0]) / np.array(ds[0]), frac), dist_range(np.array(ns[1]) / np.array(ds[1]), frac)





