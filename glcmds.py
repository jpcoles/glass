from __future__ import division, with_statement
import math
from numpy import arctan2, savez, load, array
import environment
from environment import env, set_env, Image, Source
from shear import Shear
import cosmo
#from handythread import parallel_map
from scales import convert
from itertools import izip, count, repeat

from log import log as Log, setup_log

def ptmass(xc, yc, mmin, mmax): assert False, "ptmass not implemented"
def redshifts(*args):           assert False, "redshifts not implemented"
def double(*args):              assert False, "double() not supported. Use source()."
def quad(*args):                assert False, "quad() not supported. Use source()."
def multi(*args):               assert False, "multi() not supported. Use source()."


def globject(name):
    return env().new_object(name)

def shear(phi):
    env().current_object().shear = Shear(phi)

def zlens(z):
    assert z is not None
    o = env().current_object()
    assert o.z is None, 'zlens() can only be called once per object.'
    o.z = z
    o.dL = cosmo.angdist(0,o.z)
    o.scales = cosmo.scales(o.z, 0)

def omega(om, ol):
    assert len(env().objects) == 0, 'omega() must be used before any objects are created.'
    env().omega_matter = om
    env().omega_lambda = ol

#def lens(zsrc, img0, img0parity, *imgs):
#    print 'lens() is now deprecated. Use source() instead.'
#    source(zsrc, img0, img0parity, imgs)

def source(zsrc, img0=None, img0parity=None, *imgs):

    o = env().current_object()

    assert o.z is not None, "zlens() must first be specified."
    assert zsrc >= o.z, "Source is not behind lens."

    src = Source(zsrc, o.z)

    if img0 is not None and img0parity is not None:
        image0 = Image(img0, img0parity)
        src.add_image(image0)

        prev = image0
        for i in xrange(0, len(imgs), 3):
            img,parity,time_delay = imgs[i:i+3]
            assert time_delay != 0, 'Cannot set a time delay of 0. Use (None,None) instead.'
            if prev.parity_name == 'sad' and parity == 'max':
                prev.angle = arctan2(prev.pos.imag-img[1], 
                                     prev.pos.real-img[0]) * 180/math.pi
            image = Image(img, parity)
            src.add_image(image)
            if time_delay: time_delay = convert('days to years', time_delay)
            src.add_time_delay(prev,image, time_delay)
            prev = image

    o.add_source(src)
    return src

def delay(A,B, delay):
    """ Add a time delay between images A and B such that B arrives 'delay' days after A. """
    src = env().current_object().current_source()
    a = src.images[src.images.index(A)]
    b = src.images[src.images.index(B)]
    src.add_time_delay(a,b,delay)

def symm(v=False):
    """Turn lens symmetry on or off. Default is off."""
    assert False, "Symmetry not yet supported."
    env().current_object().symm = v

def g(*args):
    env().g      = array(args)
    env().h_spec = 1 / array(args)
    #h = args if len(args) > 1 else [args[0], args[0]]
    #env().h_spec = (1/h[0], 1/h[1])

def t0(*args):
    """Set H0^-1 (or a range) in Gyr"""

    #env().H0inv   = args
    env().nu      = convert('H0^-1 in Gyr to nu', array(args))
    #env().h_spec = 1 / array(args)

def maprad(r):
    env().current_object().maprad = r

def clear():
    env().clear()

def savestate(fname):

    header = '''\
GLASS version 0.1
CREATED ON: %s'''

    env().creation_info = [header]
    #ppf = env().post_process_funcs
    #pff = env().post_filter_funcs
    #env().post_process_funcs = []
    #env().post_filter_funcs = []

    with open(fname, 'w') as f:
        savez(f, env())

    #env().post_process_funcs = ppf
    #env().post_filter_funcs = pff

def loadstate(fname, setenv=True):
    """ Load the state stored in fname and replace the current environment. If
    setenv is False the environment will not be replaced. Return the loaded
    environment.  
    """
    x = load(fname)['arr_0'].item()
    if setenv: set_env(x)
    return x

def post_process(f, *args, **kwargs):
    env().current_object().post_process_funcs.append([f, args, kwargs])

def post_filter(f, *args, **kwargs):
    env().current_object().post_filter_funcs.append([f, args, kwargs])

def _filter(arg):
    model,i,nmodels = arg
    for obj,data in model['obj,data']:
        Log( 'Post filtering ... Model %i/%i Object %s' % (i+1, nmodels, obj.name) )
        for f,args,kwargs in obj.post_filter_funcs:
            if not f([obj,data], *args, **kwargs): return False
    return True

def apply_filters():
    models = env().models
    for m in models: m['accepted'] = False           # Reject all
    models = filter(_filter, izip(models, count(), repeat(len(models))))      # Run each filter, keeping those that survive
    #models = filter(parallel_map(_filter, models, threads=10))                 # Run each filter, keeping those that survive
    for m,_,_ in models: m['accepted'] = True            # Those that make it to the end are accepted

    env().accepted_models = models

