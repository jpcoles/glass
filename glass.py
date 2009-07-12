from __future__ import division, with_statement
import sys, getopt, os, traceback
import numpy, pylab
from environment import env
import cosmo
from handythread import parallel_map

from glcmds import *
#from funcs import *
from plots import *
from scales import *
from potential import *
#import filters

def str_range(v, fmt):
    if v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return fmt % v
    elif len(v) == 1:
        return fmt % v[0]
    elif len(v) > 1:
        return '(' + ', '.join(map(lambda x: fmt%x, v)) + ')'
    else:
        return str(v)

def pp(str, units, width=80):
    slen = len(str)
    ulen = len(units)
    return str + (' '*(width-ulen-slen)) + units

def report():
    _env = env()
    print '=' * 80
    print 'COSMOLOGY'
    print '=' * 80
    print pp('Omega Matter = %.4f' % _env.omega_matter, '')
    print pp('Omega Lambda = %.4f' % _env.omega_lambda, '')
    print pp('g            = %s'   % str_range(_env.g, '%.4f'), '[Gyr]')
    print pp('1/g          = %s'   % str_range(_env.h_spec, '%.4f'), '[km/s/Mpc]')
    print 
    print '=' * 80
    print 'OBJECTS'
    print '=' * 80
    for i,o in enumerate(_env.objects):
        print pp('%i. %s at z=%.4f' % (i+1, o.name, o.z), '')
        print pp('    Map radius            = %.4f' % o.maprad, '[arcsec]')
        print pp('    Map radius g=14       = %.4f' % Arcsec_to_Kpc(obj,o.maprad,14), '[kpc]')
        print pp('    Time scale            = %.4f' % o.scales['time'],    '[g days/arcsec^2]')
        print pp('    Angular distance      = %.4f' % o.scales['angdist'], '[g kpc/arcsec]')
        print pp('    Critical density      = %.4e' % o.scales['critden'], '[g Msun/arcsec^2]')
        print pp('    Critical density g=14 = %.4e' \
            % KappaArcsec2_to_MsunKpc2(obj,1,14), '[Msun/kpc^2]')
        print pp('    Shear                 = %s' % str_range(o.shear, '%.4f'), '')
        print pp('    Steepness             = %s' % str_range(o.steep, '%.4f'), '')
        print
        for src in o.sources:
            print '    Source at z=%.4f' % src.z,
            print '[NO IMAGES]' if len(src.images) == 0 else ''
            print pp('        Distance (Obs->Src)  = %.4f' % cosmo.angdist(0,src.z), '[arcsec]')
            print pp('        Distance (Lens->Src) = %.4f' % cosmo.angdist(o.z,src.z), '[arcsec]')
            print pp('        Dos/Dls              = %.4f' % src.zcap, '')
            for img in src.images:
                print '        Image at (%.4f,%.4f) : angle=%.4f parity=%s elongation=[%.4f,%.4f,%.4f]' \
                    % (img.pos[0], img.pos[1], img.parity_name, img.elongation[0], img.elongation[1], img.elongation[2])

    print 
    print '=' * 80
    print 'MISCELLANEOUS'
    print '=' * 80
    print 'filled_beam = %s' % _env.filled_beam
    print 
    print '=' * 80
    print 'SYSTEM'
    print '=' * 80
    print 'Number of CPUs detected = %i' % _env.ncpus_detected
    print 'Number of CPUS used     = %i' % _env.ncpus
    print 


# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
def model(nmodels):

    for o in env().objects:
        o.init()

    report()

    init_model_generator()

    env().models = []
    env().solutions = []
    for i,m in enumerate(generate_models(env().objects, nmodels)):
        print 'Model %i/%i complete.' % (i+1, nmodels)
        env().models.append(m)
        env().solutions.append(m['sol'])

    _post_process()

    env().accepted_models = env().models

def _post_process():
    nmodels = len(env().models)
    for i,m in enumerate(env().models):
        for o,data in m['obj,data']:
            print 'Post processing ... Model %i/%i Object %s' % (i+1, nmodels, o.name)
            for f,args,kwargs in o.post_process_funcs:
                f((o,data), *args, **kwargs)

# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
def reprocess(state_file):
    for o in env().objects:
        print o.name
        o.init()

    env().solutions = loadstate(state_file, setenv=False).solutions

    init_model_generator()

    env().models = [ m for m in regenerate_models(env().objects) ]
    _post_process()

    #env().models = parallel_map(_f, regenerate_models(env().objects), threads=10)
    env().accepted_models = env().models

def XXXreprocess(state_file):
    for o in env().objects:
        print o.name
        o.init()

    env().solutions = loadstate(state_file, setenv=False).solutions

    init_model_generator()

    env().models = []
    for i,m in enumerate(regenerate_models(env().objects)):
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

#    try:
    if 1:
        #-----------------------------------------------------------------------
        # We exec the original file, not the text we stored in input_file
        # because if there is an exception the stack trace will print the
        # correct filename instead of <string>.
        #-----------------------------------------------------------------------
        execfile(list[0])
#   except (SyntaxError, TypeError, KeyError, NameError, ValueError):
#       traceback.print_exc(file=sys.stderr, limit=0)
#   except Exception:
#       fname = 'glass-crash.%i' % os.getpid()
#       savestate(fname)
#       traceback.print_exc(file=sys.stderr)
#       print >>sys.stderr
#       print >>sys.stderr, "********************************************************************************"
#       print >>sys.stderr, "* GLASS EXCEPTION CAUGHT. State automatically saved to %s." % fname
#       print >>sys.stderr, "********************************************************************************"
#       print >>sys.stderr

