from __future__ import division
import math
from environment import env, Image, System
from shear import Shear
import cosmo

def ptmass(xc, yc, mmin, mmax): assert False, "ptmass not implemented"
def redshifts(*args):           assert False, "redshifts not implemented"
def double(*args):              assert False, "double() not supported. Use lens()."
def quad(*args):                assert False, "quad() not supported. Use lens()."

def globject(name):
    env.new_object(name)

def subdiv(n):
    n = int(n)
    assert (n > 2 and n%2==0), "subdiv: n must be odd and > 2"
    env.current_object().S = s

def shear(phi):
    env.current_object().shear = Shear(phi)

def zlens(z):
    o = env.current_object()
    o.zlens = z

def cosm(om, ol):
    cosmo.omega_matter = om
    cosmo.omega_lambda = ol

def lens(zsrc, img0, img0parity, *imgs):

    o = env.current_object()
    assert o.zlens < zsrc, "Lens is not infront of source."

    o.zlens, o.tscale, o.tscalebg, o.dlscale, o.cdscale = cosmo.scales(o.zlens,zsrc)
    sys = System(cosmo.angdist(0,zsrc) / cosmo.angdist(o.zlens,zsrc))

    image0 = Image(img0, img0parity)
    sys.add_image(image0)

    prev = image0
    for i in xrange(0, len(imgs), 3):
        img,parity,time_delay = imgs[i:i+3]
        image = Image(img, parity)
        sys.add_image(image)
        sys.add_time_delay(prev,image, time_delay)
        prev = image

    o.add_system(sys)


def symm(v=True):
    """Turn lens symmetry on or off. Default is on."""
    assert False, "Symmetry not yet supported."
    env.current_object().symm = v

def dgcone(theta):
    assert (0 < theta <= 90), "dgcone: need 0 < theta <= 90"
    env.current_object().cen_ang = (90-theta) * math.pi/180

def minsteep(a):
    env.current_object().minsteep = a

def maxsteep(a):
    env.current_object().maxsteep = a

def g(h):
    env.current_object().h_spec = 1.0/h

def kann(theta):
    env.current_object().kann_spec = theta

def maprad(r):
    env.current_object().maprad = r


def postfilter(*fs):
    models = env.models
    for f in fs:
        models = filter(f, models)

    env.accepted_models = models
