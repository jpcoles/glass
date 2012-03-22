from __future__ import division, with_statement
import time
import numpy as np
from numpy import arctan2, savez, load, array, pi
from itertools import izip, count, repeat

import cosmo

from environment import set_env, Image, Source, command
from shear import Shear
from scales import convert
from glass import report
from log import log as Log, setup_log

@command
def ptmass(xc, yc, mmin, mmax): assert False, "ptmass not implemented"
@command
def redshifts(*args):           assert False, "redshifts not implemented"
@command
def double(*args):              assert False, "double() not supported. Use source()."
@command
def quad(*args):                assert False, "quad() not supported. Use source()."
@command
def multi(*args):               assert False, "multi() not supported. Use source()."
@command
def g(*args):                   assert False, 'g() no longer supported, use hubble_time'
@command
def t0(*args):                  assert False, 't0() no longer supported, use hubble_time'

@command
def globject(env, name):
    return env.new_object(name)

@command
def shear(env, phi, strength=0.1):
    env.current_object().shear = Shear(phi)
    env.current_object().prior_options['shear']['strength'] = strength

@command
def zlens(env, z):
    assert z is not None
    o = env.current_object()
    assert o.z is None, 'zlens() can only be called once per object.'
    o.z = z
    o.dL = cosmo.angdist(0,o.z)

@command
def omega(env, om, ol):
    assert len(env.objects) == 0, 'omega() must be used before any objects are created.'
    env.omega_matter = om
    env.omega_lambda = ol

#def lens(zsrc, img0, img0parity, *imgs):
#    print 'lens() is now deprecated. Use source() instead.'
#    source(zsrc, img0, img0parity, imgs)

@command
def source(env, zsrc, img0=None, img0parity=None, *imgs, **kwargs):

    o = env.current_object()

    assert o.z is not None, "zlens() must first be specified."
    assert zsrc >= o.z, "Source is not behind lens."

    src = Source(zsrc, o.z)

    if kwargs.has_key('loc'):
        src.pos = complex(kwargs['loc'][0], kwargs['loc'][1])

    if img0 is not None and img0parity is not None:
        image0 = Image(img0, img0parity)
        src.add_image(image0)

        prev = image0
        for i in xrange(0, len(imgs), 3):
            img,parity,time_delay = imgs[i:i+3]
            assert time_delay != 0, 'Cannot set a time delay of 0. Use (None,None) instead.'
            if prev.parity_name == 'sad' and parity == 'max':
                prev.angle = arctan2(prev.pos.imag-img[1], 
                                     prev.pos.real-img[0]) * 180/pi
            image = Image(img, parity)
            src.add_image(image)
            src.add_time_delay(prev,image, time_delay)
            prev = image

    o.add_source(src)
    return src

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
    assert len(env.objects) == 0, 'universe_age() must be used before any objects are created.'
    nu       = convert('age in Gyr to nu', array(args), cosmo.age_factor())
    env.nu = array([nu[-1], nu[0]])

@command
def hubble_time(env, *args):
    """Set H0^-1 (or a range) in Gyr"""
    print env, args
    assert len(env.objects) == 0, 'hubble_time() must be used before any objects are created.'
    nu       = convert('H0^-1 in Gyr to nu', array(args))
    env.nu = array([nu[-1], nu[0]])

@command
def hubble_constant(env, *args):
    """Set H0 (or a range) in km/s/Mpc"""
    assert len(env.objects) == 0, 'hubble_constant() must be used before any objects are created.'
    env.nu      = convert('H0 in km/s/Mpc to nu', array(args))

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

    header = '''\
GLASS version 0.1
CREATED ON: %s''' % time.asctime()

    env.creation_info = [header]
    #ppf = env.post_process_funcs
    #pff = env.post_filter_funcs
    #env.post_process_funcs = []
    #env.post_filter_funcs = []

    import numpy as np
    with open(fname, 'w') as f:
        print np.asanyarray(env)
        savez(f, env)

    #env.post_process_funcs = ppf
    #env.post_filter_funcs = pff

@command
def loadstate(env, fname, setenv=True):
    """ Load the state stored in fname and replace the current environment. If
    setenv is False the environment will not be replaced. Return the loaded
    environment.  
    """
    x = load(fname)['arr_0'].item()
    if setenv: set_env(x)
    return x

@command
def post_process(env, f, *args, **kwargs):
    env.current_object().post_process_funcs.append([f, args, kwargs])

@command
def post_filter(env, f, *args, **kwargs):
    env.current_object().post_filter_funcs.append([f, args, kwargs])

def _filter(arg):
    model,i,nmodels = arg
    for obj,data in model['obj,data']:
        if obj.post_filter_funcs:
            Log( 'Post filtering ... Model %i/%i Object %s' % (i+1, nmodels, obj.name) )
            for f,args,kwargs in obj.post_filter_funcs:
                if not f([obj,data], *args, **kwargs): return False
    return True

@command
def apply_filters(env):
    models = env.models
    for m in models: m['accepted'] = False           # Reject all
    models = filter(_filter, izip(models, count(), repeat(len(models))))      # Run each filter, keeping those that survive
    #models = filter(parallel_map(_filter, models, threads=10))                 # Run each filter, keeping those that survive
    for m,_,_ in models: m['accepted'] = True            # Those that make it to the end are accepted

    env.accepted_models = models

@command
def model(env, nmodels=None, *args, **kwargs):

    Log( '=' * 80 )
    Log('GLASS version 0.1  %s' % time.asctime())
    Log( '=' * 80 )

    for o in env.objects:
        o.init()

    report()

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
        for i,m in enumerate(generate_models(env.objects, nmodels, *args, **kwargs)):
            Log( 'Model %i/%i complete.' % (i+1, nmodels) )
            models.append(m)
            solutions.append(m['sol'])
            #print 'glcmds.py:model ???', id(m['sol'])

        _post_process(models)

    env.accepted_models = env.models

    env.models.extend(models)
    env.solutions.extend(solutions)

def _post_process(models):
    nmodels = len(models)
    for i,m in enumerate(models):
        for o,data in m['obj,data']:
            if o.post_process_funcs:
                Log( 'Post processing ... Model %i/%i Object %s' % (i+1, nmodels, o.name) )
                for f,args,kwargs in o.post_process_funcs:
                    f((o,data), *args, **kwargs)

# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
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
    env.accepted_models = env.models

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
        for m1,m2 in izip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            mass0 = data0['kappa'] * convert('kappa to Msun/arcsec^2', 1, obj0.dL, data0['nu'])
            mass1 = data['kappa'] * convert('kappa to Msun/arcsec^2', 1, obj.dL, data['nu'])
            total_rms2 += np.mean((mass1 - mass0)**2)
    return np.sqrt(total_rms2)


@command
def kappa_chi2(env, models, model0):
    n_chi2 = 0
    d_chi2 = 0
    for m in models:
        for m1,m2 in izip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            v0 = data0['kappa']
            v1 = data['kappa']
            n_chi2 += np.sum((v1 - v0)**2)
            d_chi2 += np.sum(v0**2)
    return n_chi2 / d_chi2


@command
def kappa_profile_chi2(env, models, model0):
    n_chi2 = 0
    d_chi2 = 0
    for m in models:
        for m1,m2 in izip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            v0 = data0['kappa(R)'][1:-2]
            v1 = data['kappa(R)'][1:-2]
            n_chi2 += np.sum((v1 - v0)**2)
            d_chi2 += np.sum(v0**2)
    return n_chi2 / d_chi2


@command
def time_delay_chi2(env, models, model0):
    n_chi2 = 0
    d_chi2 = 0
    for m in models:
        for m1,m2 in izip(m['obj,data'], model0['obj,data']):
            obj,data = m1
            obj0,data0 = m2
            v0 = np.array([ td for tds in data0['time delays'] for td in tds ])
            v1 = np.array([ td for tds in data['time delays'] for td in tds ])
            n_chi2 += np.sum((v1 - v0)**2)
            d_chi2 += np.sum(v0**2)
    return n_chi2 / d_chi2

