# -*- coding: utf-8 -*-
'''

does all the imports and setups required for glass
use `set_ncpu(4)` to set the number of threads after start up
and then run any glass commands

Will assume a quad core machine by default

It's not very usefull if just run on it's own, so:

-- EITHER: --
use this for interactive sessions with ipython
%run setup_glass.py

make sure to run ./setup_env before starting ipython

-- OR: -----
best to use with the shellscript:
./interactive_glass
(will call this file automatically and drop you into an interactive ipython
shell)


Modified version of `glass.py`
-- RK 2015.07.08
'''

from __future__ import division, with_statement, absolute_import
import sys, getopt, os, traceback

from glass.environment import env, Environment
from glass.command import command, Commands
from glass.exmass import * #PointMass
from glass.exceptions import GLInputError

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


@command('Load a glass basis set')
def glass_basis(env, name, **kwargs):
    env.basis_options = kwargs
    f = __import__(name, globals(), locals())
    for name,[f,g,help_text] in Commands.glass_command_list.iteritems():
        if __builtins__.__dict__.has_key(name):
            print 'WARNING: Glass command %s (%s) overrides previous function %s' % (name, f, __builtins__.__dict__[name])
        __builtins__.__dict__[name] = g



def _detect_cpus():
    """
    Detects the number of CPUs on a system.
    From http://codeliberates.blogspot.com/2008/05/detecting-cpuscores-in-python.html
    From http://www.artima.com/weblogs/viewpost.jsp?thread=230001
    """
    import subprocess
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



# some nice help and info on stdout
def print_starup_status():
    print ""
    print '-'*80
    print "\nsetup of glass complete"
    print "ncpus_detected:", Environment.global_opts['ncpus_detected']
    print "ncpus_used    :", Environment.global_opts['ncpus']
    print "\n- use `set_ncpus(X)` to change the number of cpus used"
    print "- use glass commands like `loadstate(filename)`"
    print "- run an (analysis)script with `%run script.py`"
    print "- run a spagetti config file with `%run configfile.cfg`"
    print ''
    print 'current directory:', os.getcwd()


# small helper to set the number of procs on the fly
def set_ncpus(nc):
    Environment.global_opts['ncpus'] = nc
    print_starup_status()


Commands.set_env(Environment())

ncpus = _detect_cpus()
Environment.global_opts['ncpus_detected'] = ncpus
Environment.global_opts['ncpus'] = ncpus
Environment.global_opts['omp_opts'] = {} #_detect_omp()
Environment.global_opts['withgfx'] = True


import glass.plots 
import glass.glcmds
import glass.scales

glass_basis('glass.basis.pixels', solver='rwalk')


print_starup_status()





