
import time
import numpy as np
from numpy import arctan2, pi
from itertools import count, repeat

import glass.cosmo
from  glass.report import report

from glass.environment import Image, Arc, Source, Environment
from glass.command import command, Commands
from glass.shear import Shear
from glass.scales import convert
from glass.log import log as Log, setup_log
from glass.exceptions import GLInputError
from glass.utils import dist_range

#@command('Load a glass basis set')
#def glass_basis(env, name, **kwargs):
#    env.basis_options = kwargs
#    f = __import__(name, globals(), locals())
#    for name,[f,g,help_text] in Commands.glass_command_list.items():
#        #if name in __builtins__.__dict__:
#        if name in builtins.__dict__:
#            print('WARNING: Glass command %s (%s) overrides previous function %s' % (name, f, builtins.__dict__[name]))
#        builtins.__dict__[name] = g

@command
def ptmass(xc, yc, mmin, mmax): raise GLInputError("ptmass not supported. Use external_mass().")
@command
def redshifts(*args):           raise GLInputError("redshifts not implemented.")
@command
def double(*args):              raise GLInputError("double() not supported. Use source().")
@command
def quad(*args):                raise GLInputError("quad() not supported. Use source().")
@command
def multi(*args):               raise GLInputError("multi() not supported. Use source().")
@command
def g(*args):                   raise GLInputError('g() no longer supported, use hubble_time().')
@command
def t0(*args):                  raise GLInputError('t0() no longer supported, use hubble_time().')

@command
def new_env(env):
    Commands.set_env(Environment())

@command
def globject(env, name):
    return env.new_object(name)

@command
def shear(env, strength=0.1):
    if not prior_included('external_shear'): raise GLInputError("The 'external_shear' prior must be included to enable the shear() command.")
    env.current_object().extra_potentials.append(Shear(shift=strength))
    env.current_object().prior_options['shear']['strength'] = strength

@command
def external_mass(env, mass_obj, mass_range=None):
    if not prior_included('external_mass'): raise GLInputError("The 'external_mass' prior must be included to enable the external_mass() command.")
    if mass_range is None:
        min, max = None, None
    else:
        if isinstance(mass_range, (int, float)):
            min = max = mass_range
        else:
            min, max = mass_range
            if min is not None and min < 0: raise GLInputError("external_mass: Lower bound 'min' must be at least zero.")
            if max is not None and max < 0: raise GLInputError("external_mass: Upper bound 'max' must be at least zero.")
            if min > max: raise GLInputError("external_mass: Lower bound 'min' must be less than upper bound 'max'.")
    env.current_object().prior_options['external mass'][mass_obj] = min, max
    #env.current_object().add_external_mass(mass_obj)
    env.current_object().extra_potentials.append(mass_obj)

@command
def zlens(env, z):
    assert z is not None
    o = env.current_object()
    if o.z is not None: raise GLInputError('zlens() can only be called once per object.')
    o.z = z
    o.dL = glass.cosmo.angdist(env,0,o.z)

@command
def omega(env, om, ol):
    if len(env.objects) != 0: raise GLInputError('omega() must be used before any objects are created.')
    env.omega_matter = om
    env.omega_lambda = ol

#def lens(zsrc, img0, img0parity, *imgs):
#    print 'lens() is now deprecated. Use source() instead.'
#    source(zsrc, img0, img0parity, imgs)

@command
def source(env, zsrc, img0=None, img0parity=None, *imgs, **kwargs):

    o = env.current_object()

    if o.z is None: raise GLInputError("zlens() must first be specified.")
    if zsrc < o.z: raise GLInputError("Source is not behind lens.")

    src = Source(env, zsrc, o.z, kwargs.get('zcap', None))

    if 'position' in kwargs and kwargs['position'] is not None:
        if not prior_included('source_position'): raise GLInputError("The 'source_position' prior must be included when using the position keyword.")
        src.pos = complex(kwargs['position'][0], kwargs['position'][1])
        if len(kwargs['position']) == 3:
            src.pos_tol = kwargs['position'][2]


    if img0 is not None and img0parity is not None:
#       if isinstance(img0[0], (list,tuple)):
#           image0 = Arc(img0, img0parity)
#           src.add_arc(image0)
#       else:
        image0 = Image(img0, img0parity)
        src.add_image(image0)

        prev = image0
        for i in range(0, len(imgs), 3):
            img,parity,time_delay = imgs[i:i+3]
            if time_delay == 0: raise GLInputError('Cannot set a time delay of 0. Use None instead.')
            if prev.parity_name == 'sad' and parity == 'max':
                prev.angle = arctan2(prev.pos.imag-img[1], 
                                     prev.pos.real-img[0]) * 180/pi
            image = Image(img, parity)
#           if isinstance(img[0], (list,tuple)):
#               image = Arc(img, parity)
#               assert len(src.arcs) == len(img)
#               src.add_arc(image)
#           else:
            image = Image(img, parity)
            src.add_image(image)
            src.add_time_delay(prev,image, time_delay)
            prev = image

    o.add_source(src)
    return src

@command
def link_sources(env, *srcs, **kwargs):
    pos = kwargs.pop('position', srcs[0].pos)
    tol = kwargs.pop('tol', None)
    #assert tol is not None and tol > 0
    w = env.current_object().prior_options['link sources']
    w[len(w)] = [srcs,tol]
    #w['position'] = pos

@command
def delay(env, A,B, delay):
    """ Add a time delay between images A and B such that B arrives 'delay' days after A. """
    src = env.current_object().current_source()
    a = src.images[src.images.index(A)]
    b = src.images[src.images.index(B)]
    src.add_time_delay(a,b,delay)

@command
def symm(env, v=True):
    """Turn lens symmetry on or off. Default is off."""
    #assert False, "Symmetry not yet supported."
    env.current_object().symm = v

@command
def universe_age(env, *args):
    """Set age of the Universe in Gyr"""
    if len(env.objects) != 0: raise GLInputError('universe_age() must be used before any objects are created.')
    nu       = convert('age in Gyr to nu', np.array(args), glass.cosmo.age_factor(env))
    env.nu = np.array([nu[-1], nu[0]])
    env.hubble_args = args

@command
def hubble_time(env, *args):
    """Set H0^-1 (or a range) in Gyr"""
    #print env, args
    if len(env.objects) != 0: raise GLInputError('hubble_time() must be used before any objects are created.')
    nu       = convert('H0^-1 in Gyr to nu', np.array(args))
    env.nu = np.array([nu[-1], nu[0]])
    env.hubble_args = args

@command
def hubble_constant(env, *args):
    """Set H0 (or a range) in km/s/Mpc"""
    if len(env.objects) != 0: raise GLInputError('hubble_constant() must be used before any objects are created.')
    env.nu      = convert('H0 in km/s/Mpc to nu', np.array(args))
    env.hubble_args = args

@command
def maprad(env, r,units='arcsec'):
    if units != 'arcsec':
        assert env.nu[0] == env.nu[1]
        r = convert('%s to arcsec' % units, r, env.current_object().dL, env.nu[0])
    env.current_object().maprad = r

@command
def clear(env):
    env.clear()

@command
def savestate(env, fname):

    Log('Saving state to %s' % fname)

    env.meta_info['glheader'] = '\n'.join([
        'GLASS version 1.1',
        'CREATED ON: %s' % time.asctime()
    ])

    #ppf = env.post_process_funcs
    #pff = env.post_filter_funcs
    #env.post_process_funcs = []
    #env.post_filter_funcs = []

    import numpy as np
    with open(fname, 'wb') as f:
        np.savez(f, env)

    #env.post_process_funcs = ppf
    #env.post_filter_funcs = pff

@command
def loadstate(env, fname, setenv=True):
    """ Load the state stored in fname and replace the current environment. If
    setenv is False the environment will not be replaced. Return the loaded
    environment.  
    """
    x = np.load(fname, allow_pickle=True)['arr_0'].item()
    for o in x.objects:
        for i,s in enumerate(o.sources):
            if not hasattr(s,'index'):
                s.index = i

    #if setenv: set_env(x)
    return x

@command
def post_process(env, f, *args, **kwargs):
    env.current_object().post_process_funcs.append([f, args, kwargs])

@command
def post_filter(env, f, *args, **kwargs):
    env.current_object().post_filter_funcs.append([f, args, kwargs])

def _filter(models):
    for m in models: m['accepted'] = False           # Reject all
    models = list(filter(_filter_one, zip(models, count(), repeat(len(models)))))      # Run each filter, keeping those that survive
    #models = filter(parallel_map(_filter, models, threads=10))                 # Run each filter, keeping those that survive
    for m,_,_ in models: m['accepted'] = True            # Those that make it to the end are accepted
    return models

def _filter_one(arg):
    model,i,nmodels = arg
    for obj,data in model['obj,data']:
        if obj.post_filter_funcs:
            Log( 'Post filtering ... Model %i/%i Object %s' % (i+1, nmodels, obj.name) )
            for f,args,kwargs in obj.post_filter_funcs:
                if not f([obj,data], *args, **kwargs): return False
    return True

@command
def apply_filters(env):
    env.accepted_models = _filter(env.models)

@command
def model(env, nmodels=None, *args, **kwargs):

    for o in env.objects:
        o.init()

    report(env)

    #Log( '=' * 80 )
    #Log( 'OBJECT BASIS REPORT' )
    #Log( '=' * 80 )
    #for o in env.objects:
    #    o.basis.report()

    #init_model_generator(nmodels)

    if env.models is None:
        env.models = []
        env.solutions = []

    models = []
    solutions = []

    if nmodels is None:
        m = {'sol':  None,
             'obj,data': [ [o, {}] for o in env.objects ],
             'tagged':  False}
        models.append(m)
    else:
        t0 = time.perf_counter()
        for i,m in enumerate(generate_models(env.objects, nmodels, *args, **kwargs)):
            t1 = time.perf_counter()
            if (t1-t0) > 5:
                Log( 'Model %i/%i complete.' % (i+1, nmodels), overwritable=True)
                t0 = time.perf_counter()

            models.append(m)
            solutions.append(m['sol'])
            #print 'glcmds.py:model ???', id(m['sol'])

        Log( 'Generated %i model(s).' % len(models) )
        _post_process(models)

    env.models.extend(models)
    env.solutions.extend(solutions)
    env.accepted_models = _filter(env.models)

def _post_process(models):
    nmodels = len(models)
    nProcessed = 0
    for i,m in enumerate(models):
        has_ppfs = False
        for o,data in m['obj,data']:
            if o.post_process_funcs:
                has_ppfs = True
                #print 'Post processing ... Model %i/%i Object %s' % (i+1, nmodels, o.name)
                for f,args,kwargs in o.post_process_funcs:
                    f((o,data), *args, **kwargs)
        nProcessed += has_ppfs
    Log('Post processed %i model(s), %i had post processing functions applied.' % (nmodels, nProcessed) )

@command
def apply_function(env, f, *args, **kwargs):
    for i,m in enumerate(env.models):
        for o,data in m['obj,data']:
            f((o,data), *args, **kwargs)

@command
def apply_filter_function(env, f, *args, **kwargs):
    for i,m in enumerate(env.models):
        m['accepted'] = True
        for o,data in m['obj,data']:
            if not f((o,data), *args, **kwargs):
                m['accepted'] = False
                break

@command
def reprocess(env, state_file):
    for o in env.objects:
        Log( o.name )
        o.init()

    e = loadstate(state_file, setenv=False)
    env.solutions = e.solutions

    #init_model_generator(len(env.solutions))

    env.models = [ m for m in regenerate_models(env.objects) ]
    _post_process()

    #env.models = parallel_map(_f, regenerate_models(env.objects), threads=10)
    env.accepted_models = _filter(env.models)

def XXXreprocess(state_file):
    for o in env.objects:
        Log( o.name )
        o.init()

    env.solutions = loadstate(state_file, setenv=False).solutions

    init_model_generator(len(env.solutions))

    env.models = []
    for i,m in enumerate(regenerate_models(env.objects)):
        for o,data in m['obj,data']:
            for f,args,kwargs in o.post_process_funcs:
                f((o,data), *args, **kwargs)

        env.models.append(m)

    env.accepted_models = env.models


@command
def ensemble_mass_rms(env, models, model0):
    total_rms2 = 0
    for m in models:
        for m1,m2 in zip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            mass0 = data0['kappa'] * convert('kappa to Msun/arcsec^2', 1, obj0.dL, data0['nu'])
            mass1 = data['kappa'] * convert('kappa to Msun/arcsec^2', 1, obj.dL, data['nu'])
            total_rms2 += np.mean((mass1 - mass0)**2)
    return np.sqrt(total_rms2)


#@command
#def kappa_chi2(env, models, model0):
#    n_chi2 = 0
#    d_chi2 = 0
#    for m in models:
#        for m1,m2 in izip(m['obj,data'], model0['obj,data']):
#            obj,data = m1
#            obj0,data0 = m2
#            v0 = data0['kappa']
#            v1 = data['kappa']
#            n_chi2 += np.sum((v1 - v0)**2)
#            d_chi2 += np.sum(v0**2)
#    return n_chi2 / d_chi2

@command
def kappa_chi2(env, models, model0, frac='1sigma'):
    n_max,d_max=0,0
    n_min,d_min=np.inf,np.inf
    ns, ds = [], []
    for m in models:
        n,d = 0,0
        for m1,m2 in zip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            rs = [ abs(img.pos) for src in obj.sources for img in src.images if img.parity_name != 'max']
            rmin, rmax = np.amin(rs), np.amax(rs)

            #w = (abs(obj.basis.rs) >= rmin) * (abs(obj.basis.rs) <= rmax)
            #w = abs(obj.basis.rs) <= rmax
            w = (abs(obj.basis.ploc) >= obj.basis.top_level_cell_size * 0.9) * (abs(obj.basis.ploc) <= (rmax+ obj.basis.top_level_cell_size * 0.5))

            #b = np.argmin(abs(data['R'] - rmin))
            #e = np.argmin(abs(data['R'] - rmax))
            #v0 = data0['kappa'][b:e+1]
            #v1 = data['kappa'][b:e+1]

            v0 = data0['kappa'][w]
            v1 = data['kappa'][w]
            n += np.sum((v1 - v0)**2)
            d += np.sum(v0**2)
        ns.append(n)
        ds.append(d)

        #n_max,d_max = np.amax([n,n_max]), np.amax([d,d_max])
        #n_min,d_min = np.amin([n,n_min]), np.amin([d,d_min])
    nd = np.array(ns) / np.array(ds)
    return dist_range(nd, frac)
#   nd.sort()
#   N = len(nd)
#   if len(nd) % 2 == 0:
#       M = (nd[(N-1)//2] + nd[(N-1)//2+1]) / 2
#       L = nd[(N-1)//2   - int(0.32*N)]
#       R = nd[(N-1)//2+1 + int(0.32*N)]
#   else:
#       M = nd[(N-1)//2]
#       L = nd[(N-1)//2 - int(0.32*N)]
#       R = nd[(N-1)//2 + int(0.32*N)]
#   return M, R, L


@command
def kappa_profile_chi2(env, models, model0, frac='1sigma'):
    n_max,d_max=0,0
    n_min,d_min=np.inf,np.inf
    ns, ds = [], []
    for m in models:
        n,d = 0,0
        for m1,m2 in zip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            rs = [ abs(img.pos) for src in obj.sources for img in src.images]
            #rs = [ abs(img.pos) for src in obj.sources for img in src.images if img.parity_name != 'max']
            rmin, rmax = np.amin(rs), np.amax(rs)
            if 0:
                b = 0
            else:
                rmin = obj.basis.top_level_cell_size * 1.6
                b = np.argmin(abs(data['R'] - rmin))

            e = np.argmin(abs(data['R'] - rmax))

            v0 = data0['kappa(R)'][b:e+1]
            v1 = data['kappa(R)'][b:e+1]
            n += np.sum((v1 - v0)**2)
            d += np.sum(v0**2)
            #d += len(v0) #np.sum(v0**2)
        ns.append(n)
        ds.append(d)

    nd = np.array(ns) / np.array(ds)
    return dist_range(nd, frac)
#   nd.sort()
#   N = len(nd)
#   frac = {'1sigma': 0.6827,
#           '2sigma': 0.9545,
#           '3sigma': 0.9973}.get(frac, frac)
#   n = int(frac/2. * N)
#   mid = (N-1) // 2
#   if N % 2 == 0:
#       M = (nd[mid] + nd[mid+1]) / 2
#       L = nd[mid   - n]
#       R = nd[mid+1 + n]
#   else:
#       M = nd[mid]
#       L = nd[mid - n]
#       R = nd[mid + n]
#   return M, R, L
    #return n/d, n_max/d_max, n_min/d_min



@command
def time_delay_chi2(env, models, model0):
    n_chi2 = 0
    d_chi2 = 0
    for m in models:
        for m1,m2 in zip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            v0 = np.array([ td for tds in data0['time delays'] for td in tds ])
            v1 = np.array([ td for tds in data['time delays'] for td in tds ])
            n_chi2 += np.sum((v1 - v0)**2)
            d_chi2 += np.sum(v0**2)
    return n_chi2 / d_chi2

@command
def meta(env, *args, **kwargs):
    if len(args) != 0: raise GLInputError('meta() only takes named arguments')
    env.meta_info = kwargs

#@command
#def load_leier_grid(env, fname, grid_size, units='arcsec'):
