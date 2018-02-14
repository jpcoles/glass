from __future__ import division
import numpy, os, subprocess
from numpy import arctan2, pi, linspace, atleast_2d, abs, ndarray, asarray
from collections import defaultdict 
import traceback
import glass.cosmo
from glass.command import Commands
from glass.scales import convert


class Object:

    def __init__(self, name):
        self.name = name
        self._current_source = None
        self.sources = []

        self.S          = 0
        #self.shear      = None
        #self.external_masses = []
        self.extra_potentials = []
        self.z          = None       # [redshift]
        self.symm       = False

        self.maprad     = None      # [arcsec]

        self.basis = None
        self.prior_options = defaultdict(dict)

        self.post_process_funcs = []
        self.post_filter_funcs = []

        self.stellar_mass_error = 0
        self.stellar_mass       = 0

    def current_source(self):
        return self._current_source

    def add_source(self, source):
        self._current_source = source
        source.index = len(self.sources)
        self.sources.append(source)

    def add_external_mass(self, m):
        self.external_masses.append(m)

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

    def __str__(self):
        return '%s(%s)' % (self.pos, self.parity_name)

    def __eq__(self, a):
        return a is self or a is self._pos 

class Arc(Image):
    def __init__(self, rs, parity):
        Image.__init__(self, rs[0], parity)
        self.rs = rs

class Source:
    def __init__(self, env, zsrc, zlens, zcap=None):
        if zcap is not None:
            self.zcap = zcap
        else:
            self.zcap = glass.cosmo.angdist(env, 0,zsrc) / glass.cosmo.angdist(env, zlens,zsrc)
        self.images = []
        self.arcs = []
        self.time_delays = []
        self.z = zsrc
        self.index = 0
        self.pos = None
        self.pos_tol = 0

    def add_image(self, A):
        assert A not in self.images
        self.images.append(A)

    def add_arc(self, A):
        assert A not in self.arc
        self.arcs.append(A)

    def add_time_delay(self, A,B, delay):
        assert A in self.images
        assert B in self.images
        assert delay != (None,None), "Time delays can't have infinite range."

        if not isinstance(delay, (list, tuple)):
            delay = [delay]

        delay = [ convert('days to years', td) if td else td for td in delay ]

        self.time_delays.append((A,B,delay))


class Environment:

    H0inv_ref    = 13.7
    H0inv_ref    = 13.57
    global_opts = {}

    def __init__(self):
        self.objects = []
        self._current_object = None
        self.model_gen_factory = None #model_generator
        self.model_gen = None
        self.model_gen_options = {}
        self.solutions = None
        self.models = None
        self.accepted_models = None
        self.basis_options = {}
        self.meta_info = {}

        # For use in cosmo.py. Based on WMAP7+BAO
        self.omega_matter = 0.28
        self.omega_lambda = 0.72

        self.nu           = None
        self.filled_beam  = True

        #self.ncpus_detected = _detect_cpus()
        #self.ncpus          = self.ncpus_detected
        #self.omp_opts       = _detect_omp()
        #self.ncpus = 40
        #self.argv = []

        #self.withgfx = True
        self.bw_styles = False

#   def __getattr__(self, name):
#       f = glass_command_list.get(name, None)
#       assert f is not None, 'Glass command %s not found.' % name

    def current_object(self, o=None):
        if o is not None:
            assert o in self.objects
            self._current_object = o
        return self._current_object

    def new_object(self, name):
        self._current_object = Object(name)
        self.objects.append(self._current_object)
        return self._current_object

    def clear(self):
        self.__init__()

    def __getattr__(self, name):
        try:
            f = Commands.glass_command_list[name][0]
            return lambda *args, **kwargs: f(self, *args, **kwargs)
        except:
            raise AttributeError(name)

class DArray(ndarray):
    def __new__(cls, input_array, symbol, units=None):
        obj = asarray(input_array).view(cls)
        obj.units = units
        obj.symbol = symbol
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.units = getattr(obj, 'units', None)
        self.symbol = getattr(obj, 'symbol', None)

    def __array_wrap__(self, out_arr, context=None):
        return ndarray.__array_wrap__(self, out_arr, context)

    def __reduce_ex__(self, protocol):
        p = ndarray.__reduce_ex__(self, protocol)
        return (p[0], p[1], (p[2], self.units, self.symbol))

    def __setstate__(self, p):
        ndarray.__setstate__(self, p[0])
        self.units = p[1]
        self.symbol = p[2]

    def __getitem__(self, i):
        if isinstance(i, basestring):
            scale_factor = self.units[i][0]
            return self * scale_factor
        if i is None: return self
        return ndarray.__getitem__(self, i)

    def label(self, units):
        try:
            return self.units[units][1]
        except:
            return ''

#_env = Environment()
def env():
    #return _env
    return Commands.get_env()

#def new_env():
#    inp = env().input_file
#    ncpus = env().ncpus
#    set_env(Environment())
#    env().ncpus = ncpus
#    env().input_file = inp

#FIXME: Resetting the environment discards options set on the commandline (e.g., ncpus)
#def set_env(env):
#    global _env
#    _env = env

