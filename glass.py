from __future__ import division, with_statement
import sys, getopt, os, traceback
import numpy, pylab
from environment import env

from glcmds import *
from funcs import *
from plots import *
from scales import *
from potential import *
import filters

# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
def model(nmodels):
    for o in env().objects:
        print o.name
        o.init()

    init_model_generator()

    env().models = []
    for i,m in enumerate(generate_models(env().objects, nmodels)):
        if i == 0: continue;
        for o,data in m['obj,data']:
            for f,args,kwargs in o.post_process_funcs:
                f((o,data), *args, **kwargs)

        env().models.append(m)

    env().accepted_models = env().models

def help():
    print >>sys.stderr, "Usage: glass.py <input>"
    sys.exit(2)

if __name__ == "__main__":

    if len(sys.argv) < 2: help()

    optlist, list = getopt.getopt(sys.argv[1:], 't:h')
    for opt in optlist:
        if   opt[0] == '-h':
            help()
        elif opt[0] == '-t':
            ncpus = int(opt[1])
            assert ncpus > 0
            env().ncpus = ncpus

    with open(list[0], 'r') as f:
        env().input_file = f.read()

    try:
        #-----------------------------------------------------------------------
        # We exec the original file, not the text we stored in input_file
        # because if there is an exception the stack trace will print the
        # correct filename instead of <string>.
        #-----------------------------------------------------------------------
        execfile(list[0])
    except (SyntaxError, TypeError, KeyError, NameError, ValueError):
        traceback.print_exc(file=sys.stderr, limit=0)
    except Exception:
        fname = 'glass-crash.%i' % os.getpid()
        savestate(fname)
        traceback.print_exc(file=sys.stderr)
        print >>sys.stderr
        print >>sys.stderr, "********************************************************************************"
        print >>sys.stderr, "* GLASS EXCEPTION CAUGHT. State automatically saved to %s." % fname
        print >>sys.stderr, "********************************************************************************"
        print >>sys.stderr

