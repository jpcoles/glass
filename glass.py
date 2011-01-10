from __future__ import division, with_statement
import sys, getopt, os, traceback
import numpy
from environment import env, set_env, new_env
import cosmo
from handythread import parallel_map
from log import log as Log, setup_log


#import filters

def str_range(v, fmt):
    def tostr(v):
        return str(v) if v is None else fmt%v

    if v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return fmt % v
    elif len(v) == 1:
        return fmt % v[0]
    elif len(v) > 1:
        return '(' + ', '.join(map(tostr, v)) + ')'
    else:
        return str(v)

def pp(str, units, width=80):
    slen = len(str)
    ulen = len(units)
    return str + (' '*(width-ulen-slen)) + units

def report():
    _env = env()
    Log( '=' * 80 )
    Log( 'COSMOLOGY' )
    Log( '=' * 80 )
    Log( pp('Omega Matter = %.4g' % _env.omega_matter, '') )
    Log( pp('Omega Lambda = %.4g' % _env.omega_lambda, '') )
    Log( pp('g            = %s'   % str_range(_env.g, '%.4g'), '[Gyr]') )
    Log( pp('1/g          = %s'   % str_range(_env.h_spec, '%.4g'), '[km/s/Mpc]') )
    Log(  )
    Log( '=' * 80 )
    Log( 'OBJECTS' )
    Log( '=' * 80 )
    for i,o in enumerate(_env.objects):
        Log( pp('%i. %s at z=%.4g  Distance(Obs->Lens) = %.4f' % (i+1, o.name, o.z, cosmo.angdist(0,o.z)), '') )
        if o.maprad:
            Log( pp('    Map radius            = %.4g' % o.maprad, '[arcsec]') )
            Log( pp('    Map radius g=14       = %.4g' % Arcsec_to_Kpc(o,o.maprad,14), '[kpc]') )
        else:
            Log( pp('    Map radius            = Not specified', '') )
            Log( pp('    Map radius g=14       = Not specified', '') )
        Log( pp('    Time scale            = %.4g' % o.scales['time'],    '[g days/arcsec^2]') )
        Log( pp('    Angular distance      = %.4g' % o.scales['angdist'], '[g kpc/arcsec]') )
        Log( pp('    Critical density      = %.4e' % o.scales['critden'], '[g Msun/arcsec^2]') )
        Log( pp('    Critical density g=14 = %.4e' \
            % Kappa_to_MsunKpc2(o,1,14), '[Msun/kpc^2]') )
        if o.shear:
            pass
            #Log( pp('    Shear                 = %.4g' % o.shear.phi, '') )
        else:
            Log( pp('    NO SHEAR', '') )
            #Log( pp('    Shear                 = Not specified', '') )
#       Log( pp('    Steepness             = %s' % str_range(o.steep, '%.4g'), '') )
        Log( )
        for src in o.sources:
            Log( '    Source at z=%.4f %s' % (src.z, '[NO IMAGES]' if len(src.images) == 0 else '' ))
            Log( pp('        Distance (Obs->Src)  = %.4f' % cosmo.angdist(0,src.z), '[arcsec]') )
            Log( pp('        Distance (Lens->Src) = %.4f' % cosmo.angdist(o.z,src.z), '[arcsec]') )
            Log( pp('        Dos/Dls              = %.4f' % src.zcap, '') )
            for img in src.images: 
                Log( '        Image at (% .3f,% .3f) : angle=% 8.3f parity=%s elongation=[%.4g,%.4g,%.4g]' 
                    % (img.pos.real, img.pos.imag, img.angle, img.parity_name, img.elongation[0], img.elongation[1], img.elongation[2]) )

    Log(  )
    Log( '=' * 80 )
    Log( 'MISCELLANEOUS' )
    Log( '=' * 80 )
    Log( 'filled_beam = %s' % _env.filled_beam )
    Log(  )
    Log( '=' * 80 )
    Log( 'SYSTEM' )
    Log( '=' * 80 )
    Log( 'Number of CPUs detected = %i' % _env.ncpus_detected )
    Log( 'Number of CPUs used     = %i' % _env.ncpus )
    Log( )


# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
def model(nmodels):

    for o in env().objects:
        o.init()

    report()

    init_model_generator(nmodels)

    env().models = []
    env().solutions = []
    for i,m in enumerate(generate_models(env().objects, nmodels)):
        Log( 'Model %i/%i complete.' % (i+1, nmodels) )
        env().models.append(m)
        env().solutions.append(m['sol'])

    _post_process()

    env().accepted_models = env().models

def _post_process():
    nmodels = len(env().models)
    for i,m in enumerate(env().models):
        for o,data in m['obj,data']:
            Log( 'Post processing ... Model %i/%i Object %s' % (i+1, nmodels, o.name) )
            for f,args,kwargs in o.post_process_funcs:
                f((o,data), *args, **kwargs)

# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
def reprocess(state_file):
    for o in env().objects:
        Log( o.name )
        o.init()

    e = loadstate(state_file, setenv=False)
    env().solutions = e.solutions

    init_model_generator(len(env().solutions))

    env().models = [ m for m in regenerate_models(env().objects) ]
    _post_process()

    #env().models = parallel_map(_f, regenerate_models(env().objects), threads=10)
    env().accepted_models = env().models

def XXXreprocess(state_file):
    for o in env().objects:
        Log( o.name )
        o.init()

    env().solutions = loadstate(state_file, setenv=False).solutions

    init_model_generator(len(env().solutions))

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

    optlist, list = getopt.getopt(sys.argv[1:], 't:h', ['nw'])
    for opt in optlist:
        if   opt[0] == '-h':
            help()
        elif opt[0] == '-t':
            ncpus = int(opt[1])
            assert ncpus > 0
            env().ncpus = ncpus
        elif opt[0] == '--nw':
            env().withgfx = False

    if env().withgfx:
        from plots import *

    from glcmds import *
    #from funcs import *
    from scales import *
    from potential import *
    from pytipsy import load_tipsy

    with open(list[0], 'r') as f:
        env().input_file = f.read()

    env().argv = list[1:]

    try:
#    if 1:
        #-----------------------------------------------------------------------
        # We exec the original file, not the text we stored in input_file
        # because if there is an exception the stack trace will print the
        # correct filename instead of <string>.
        #-----------------------------------------------------------------------
        execfile(list[0])
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

