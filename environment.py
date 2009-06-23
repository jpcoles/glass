from __future__ import division
import numpy, os
from numpy import arctan2, pi, linspace, atleast_2d, abs
from potential import poten2d

def _detect_cpus():
    """
    Detects the number of CPUs on a system.
    From http://codeliberates.blogspot.com/2008/05/detecting-cpuscores-in-python.html
    From http://www.artima.com/weblogs/viewpost.jsp?thread=230001
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
        if ncpus > 0:
            return ncpus
    return 1 # Default

class Object:

    def __init__(self, name):
        self.name = name
        self._current_system = None
        self.systems = []

        self.S          = 0
        self.shear      = None
        self.scales     = None
        self.zlens      = 0.0       # [redshift]
        self.kann_spec  = 0.0   
        self.h_spec     = 0.0       # [Gyr]
        self.minsteep   = None      
        self.maxsteep   = None
        #self.maxsteep   = self.minsteep # TODO: This should be right, but setting to 0 skips a test in priors
        self.cen_ang    = pi/4
        self.symm       = False

        self.maprad     = None      # [arcsec]

        self.basis = None

        self.post_process_funcs = []
        self.post_filter_funcs = []

    def current_system(self):
        return self._current_system

    def add_system(self, system):
        self._current_system = system
        self.systems.append(system)

    def init(self):
        self.basis.init(self)

        #assert(self.maprad is not None)


class Image:
    def __init__(self, r, parity):
        assert parity in ['min', 'sad', 'max', 'unk']

        self._pos = r;
        self.pos = complex(r[0], r[1])
        self.angle = arctan2(self.pos.imag, self.pos.real) * 180/pi
        #self.angle = numpy.angle(self.pos, deg=True)
        self.elongation = [0.1, 10, 0.9]
        self.parity_name = parity
        self.parity = ['min', 'sad', 'max', 'unk'].index(parity)

    def __eq__(self, a):
        return a is self or a is self._pos 
        

class System:
    def __init__(self, zsrc, zlens):
        self.zcap = cosmo.angdist(0,zsrc) / cosmo.angdist(zlens,zsrc)
        print "zcap =", self.zcap
        self.images = []
        self.time_delays = []
        self.zsrc = zsrc

    def add_image(self, A):
        assert A not in self.images
        self.images.append(A)

    def add_time_delay(self, A,B, delay):
        assert A in self.images
        assert B in self.images
        if isinstance(delay, (int, float)):
            delay = (delay, delay)
        else:
            delay = tuple(delay)
            assert delay != (None,None), "Time delays can't have infinite range."
        self.time_delays.append((A,B,delay))


class Environment:

    def __init__(self):
        self.objects = []
        self._current_object = None
        self.nmodels = 0
        self.model_gen_factory = None #model_generator
        self.model_gen = None
        self.models = None
        self.accepted_models = None

        # For use in cosmo.py
        self.omega_matter = 0.26
        self.omega_lambda = 0.74
        self.h_spec       = None
        self.filled_beam = True

        self.ncpus = _detect_cpus()
        #self.ncpus = 3

        print "%i CPUs detected. Using %i threads" % (self.ncpus, self.ncpus)

    def current_object(self):
        return self._current_object

    def new_object(self, name):
        self._current_object = Object(name)
        self.objects.append(self._current_object)
        return self._current_object

    def clear(self):
        self.__init__()
        
        

_env = Environment()
def env():
    return _env

def set_env(env):
    global _env
    _env = env

import cosmo
