from __future__ import division, with_statement
import cosmo
from itertools import izip, count, repeat
from numpy import arctan2, savez, load, array, pi
from environment import env, set_env, Image, Source, command
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
def g(*args):
    assert False, 'g() no longer supported, use hubble_time'
    env().g      = array(args)
    env().h_spec = 1 / array(args)
    #h = args if len(args) > 1 else [args[0], args[0]]
    #env().h_spec = (1/h[0], 1/h[1])
@command
def t0(*args):
    assert False, 't0() no longer supported, use hubble_time'
    env().g      = array(args)
    env().h_spec = 1 / array(args)
    #h = args if len(args) > 1 else [args[0], args[0]]
    #env().h_spec = (1/h[0], 1/h[1])

@command
def globject(name):
    return env().new_object(name)

@command
def shear(phi):
    env().current_object().shear = Shear(phi)

@command
def zlens(z):
    assert z is not None
    o = env().current_object()
    assert o.z is None, 'zlens() can only be called once per object.'
    o.z = z
    o.dL = cosmo.angdist(0,o.z)
    o.scales = cosmo.scales(o.z, 0)

@command
def omega(om, ol):
    assert len(env().objects) == 0, 'omega() must be used before any objects are created.'
    env().omega_matter = om
    env().omega_lambda = ol

#def lens(zsrc, img0, img0parity, *imgs):
#    print 'lens() is now deprecated. Use source() instead.'
#    source(zsrc, img0, img0parity, imgs)

@command
def source(zsrc, img0=None, img0parity=None, *imgs, **kwargs):

    o = env().current_object()

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
            if time_delay: time_delay = convert('days to years', time_delay)
            src.add_time_delay(prev,image, time_delay)
            prev = image

    o.add_source(src)
    return src

@command
def delay(A,B, delay):
    """ Add a time delay between images A and B such that B arrives 'delay' days after A. """
    src = env().current_object().current_source()
    a = src.images[src.images.index(A)]
    b = src.images[src.images.index(B)]
    src.add_time_delay(a,b,delay)

@command
def symm(v=True):
    """Turn lens symmetry on or off. Default is off."""
    #assert False, "Symmetry not yet supported."
    env().current_object().symm = v

@command
def universe_age(*args):
    """Set age of the Universe in Gyr"""
    assert len(env().objects) == 0, 'universe_age() must be used before any objects are created.'
    nu       = convert('age in Gyr to nu', array(args), cosmo.age_factor())
    env().nu = array([nu[-1], nu[0]])

@command
def hubble_time(*args):
    """Set H0^-1 (or a range) in Gyr"""
    assert len(env().objects) == 0, 'hubble_time() must be used before any objects are created.'
    nu       = convert('H0^-1 in Gyr to nu', array(args))
    env().nu = array([nu[-1], nu[0]])

@command
def hubble_constant(*args):
    """Set H0 (or a range) in km/s/Mpc"""
    assert len(env().objects) == 0, 'hubble_constant() must be used before any objects are created.'
    env().nu      = convert('H0 in km/s/Mpc to nu', array(args))

@command
def maprad(r):
    env().current_object().maprad = r

@command
def clear():
    env().clear()

@command
def savestate(fname):

    header = '''\
GLASS version 0.1
CREATED ON: %s'''

    env().creation_info = [header]
    #ppf = env().post_process_funcs
    #pff = env().post_filter_funcs
    #env().post_process_funcs = []
    #env().post_filter_funcs = []

    import numpy as np
    with open(fname, 'w') as f:
        print np.asanyarray(env())
        savez(f, env())

    #env().post_process_funcs = ppf
    #env().post_filter_funcs = pff

@command
def loadstate(fname, setenv=True):
    """ Load the state stored in fname and replace the current environment. If
    setenv is False the environment will not be replaced. Return the loaded
    environment.  
    """
    x = load(fname)['arr_0'].item()
    if setenv: set_env(x)
    return x

@command
def post_process(f, *args, **kwargs):
    env().current_object().post_process_funcs.append([f, args, kwargs])

@command
def post_filter(f, *args, **kwargs):
    env().current_object().post_filter_funcs.append([f, args, kwargs])

def _filter(arg):
    model,i,nmodels = arg
    for obj,data in model['obj,data']:
        if obj.post_filter_funcs:
            Log( 'Post filtering ... Model %i/%i Object %s' % (i+1, nmodels, obj.name) )
            for f,args,kwargs in obj.post_filter_funcs:
                if not f([obj,data], *args, **kwargs): return False
    return True

@command
def apply_filters():
    models = env().models
    for m in models: m['accepted'] = False           # Reject all
    models = filter(_filter, izip(models, count(), repeat(len(models))))      # Run each filter, keeping those that survive
    #models = filter(parallel_map(_filter, models, threads=10))                 # Run each filter, keeping those that survive
    for m,_,_ in models: m['accepted'] = True            # Those that make it to the end are accepted

    env().accepted_models = models

# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
@command
def model(nmodels, *args, **kwargs):

    for o in env().objects:
        o.init()

    report()

    #init_model_generator(nmodels)

    env().models = []
    env().solutions = []
    for i,m in enumerate(generate_models(env().objects, nmodels, *args, **kwargs)):
        Log( 'Model %i/%i complete.' % (i+1, nmodels) )
        env().models.append(m)
        env().solutions.append(m['sol'])

    _post_process()

    env().accepted_models = env().models

def _post_process():
    nmodels = len(env().models)
    for i,m in enumerate(env().models):
        for o,data in m['obj,data']:
            if o.post_process_funcs:
                Log( 'Post processing ... Model %i/%i Object %s' % (i+1, nmodels, o.name) )
                for f,args,kwargs in o.post_process_funcs:
                    f((o,data), *args, **kwargs)

# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
@command
def reprocess(state_file):
    for o in env().objects:
        Log( o.name )
        o.init()

    e = loadstate(state_file, setenv=False)
    env().solutions = e.solutions

    #init_model_generator(len(env().solutions))

    env().models = [ m for m in regenerate_models(env().objects) ]
    _post_process()

    #env().models = parallel_map(_f, regenerate_models(env().objects), threads=10)
    env().accepted_models = env().models

def XXXreprocess(state_file):
    for o in env().objects:
        Log( o.name )
        o.init()

    env().solutions = loadstate(state_file, setenv=False).solutions

    init_model_generator(len(env().solutions))

    env().models = []
    for i,m in enumerate(regenerate_models(env().objects)):
        for o,data in m['obj,data']:
            for f,args,kwargs in o.post_process_funcs:
                f((o,data), *args, **kwargs)

        env().models.append(m)

    env().accepted_models = env().models


