from __future__ import division, with_statement, absolute_import
import sys, getopt, os, traceback
import numpy

from glass.environment import env, DArray, Environment

import glass.report 
from glass.log import log as Log, setup_log
from glass.scales import convert
from numpy import abs
import glass.basis as basis
from glass.command import command, Commands

#GlassEnvironment = Environment


#if not globals().has_key('glass_command_list'):
#    print 'Resetting command list!'
#    glass_command_list = {}


def Ximport_functions(pkg):
    f = __import__(pkg, globals(), locals())
    #print f.__dict__
    print f
    g = globals()
    for name,func in f.__dict__.iteritems():
        if name.startswith('glcmd__'):
            print '*' * 80
            print name
            g[name.split('glcmd__')[1]] = func

@command
def glass_basis(env, name, **kwargs):
    env.basis_options = kwargs
    f = __import__(name, globals(), locals())
    for name,[f,g] in Commands.glass_command_list.iteritems():
        if __builtins__.__dict__.has_key(name):
            print 'WARNING: Glass command %s (%s) overrides previous function %s' % (name, f, __builtins__.__dict__[name])
        __builtins__.__dict__[name] = g


def help():
    print >>sys.stderr, "Usage: glass.py <input>"
    sys.exit(2)

if __name__ == "__main__":

    if len(sys.argv) < 2: help()

    Commands.set_env(Environment())

    optlist, arglist = getopt.getopt(sys.argv[1:], 't:h', ['nw'])
    for opt in optlist:
        if   opt[0] == '-h':
            help()
        elif opt[0] == '-t':
            ncpus = int(opt[1])
            assert ncpus > 0
            Commands.get_env().ncpus = ncpus
        elif opt[0] == '--nw':
            Commands.get_env().withgfx = False

    if Commands.get_env().withgfx:
        import glass.plots 

    import glass.glcmds
    import glass.scales
    #import pytipsy 

    with open(arglist[0], 'r') as f:
        Commands.get_env().input_file = f.read()

    Commands.get_env().argv = arglist


    execfile(arglist[0]) #, globals(), globals())
    try:
#    if 1:
        #-----------------------------------------------------------------------
        # We exec the original file, not the text we stored in input_file
        # because if there is an exception the stack trace will print the
        # correct filename instead of <string>.
        #-----------------------------------------------------------------------
        pass
    except (SyntaxError, TypeError, KeyError, NameError, ValueError, KeyboardInterrupt):
        traceback.print_exc(file=sys.stderr)
        #traceback.print_exc(file=sys.stderr, limit=0)
    except: #(Exception, Error):
        fname = 'glass-crash.%i' % os.getpid()
        savestate(fname)
        traceback.print_exc(file=sys.stderr)
        print >>sys.stderr
        print >>sys.stderr, "********************************************************************************"
        print >>sys.stderr, "* GLASS EXCEPTION CAUGHT. State automatically saved to %s." % fname
        print >>sys.stderr, "********************************************************************************"
        print >>sys.stderr

