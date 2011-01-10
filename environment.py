from __future__ import division
import numpy, os, subprocess
from numpy import arctan2, pi, linspace, atleast_2d, abs
from potential import poten2d
from collections import defaultdict 

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
            #return int(os.popen2("sysctl -n hw.ncpu")[1].read())
            return int(subprocess.Popen("sysctl -n hw.ncpu",shell=True,stdout=subprocess.PIPE).communicate()[0])
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
        if ncpus > 0:
            return ncpus
    return 1 # Default

class Object:

    def __init__(self, name):
        self.name = name
        self._current_source = None
        self.sources = []

        self.S          = 0
        self.shear      = None
        self.scales     = None
        self.z          = None       # [redshift]
        self.symm       = False

        self.maprad     = None      # [arcsec]

        self.basis = None
        self.prior_options = defaultdict(dict)

        self.post_process_funcs = []
        self.post_filter_funcs = []

    def current_source(self):
        return self._current_source

    def add_source(self, source):
        self._current_source = source
        self.sources.append(source)

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
        

class Source:
    def __init__(self, zsrc, zlens):
        self.zcap = cosmo.angdist(0,zsrc) / cosmo.angdist(zlens,zsrc)
        self.images = []
        self.time_delays = []
        self.z = zsrc

    def add_image(self, A):
        assert A not in self.images
        self.images.append(A)

    def add_time_delay(self, A,B, delay):
        assert A in self.images
        assert B in self.images
        assert delay != (None,None), "Time delays can't have infinite range."

        if isinstance(delay, (list, tuple)):
            delay = tuple(delay)
        else:
            delay = [delay]
        self.time_delays.append((A,B,delay))


class Environment:

    def __init__(self):
        self.objects = []
        self._current_object = None
        self.model_gen_factory = None #model_generator
        self.model_gen = None
        self.solutions = None
        self.models = None
        self.accepted_models = None

        # For use in cosmo.py
        self.omega_matter = 0.26
        self.omega_lambda = 0.74

        self.omega_matter = 0.3
        self.omega_lambda = 0.7

        self.h_spec       = [None, None]
        self.g            = None
        self.nu           = None
        self.filled_beam  = True

        self.ncpus_detected = _detect_cpus()
        self.ncpus          = self.ncpus_detected
        #self.ncpus = 3
        self.argv = []

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

def new_env():
    inp = env().input_file
    ncpus = env().ncpus
    set_env(Environment())
    env().ncpus = ncpus
    env().input_file = inp

#FIXME: Resetting the environment discards options set on the commandline (e.g., ncpus)
def set_env(env):
    global _env
    _env = env

import cosmo
