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


# small helper to set the number of procs on the fly
def set_ncpus(nc):
    Environment.global_opts['ncpus'] = nc
    Environment.global_opts['ncpus_detected'] = nc
    

Commands.set_env(Environment())

Environment.global_opts['ncpus_detected'] = 4
Environment.global_opts['ncpus'] = 4
Environment.global_opts['omp_opts'] = {} #_detect_omp()
Environment.global_opts['withgfx'] = True


import glass.plots 

import glass.glcmds
import glass.scales


glass_basis('glass.basis.pixels', solver='rwalk')

