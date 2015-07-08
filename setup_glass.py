'''
does all the imports and setups required for glass

use this for interactive sessions with ipython
%run setup_glass.py

make sure to run ./setup_env before starting ipython
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


Commands.set_env(Environment())

Environment.global_opts['ncpus_detected'] = 4
Environment.global_opts['ncpus'] = 4
Environment.global_opts['omp_opts'] = {} #_detect_omp()
Environment.global_opts['withgfx'] = True


import glass.plots 

import glass.glcmds
import glass.scales


glass_basis('glass.basis.pixels', solver='rwalk')

