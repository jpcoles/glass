from __future__ import division, with_statement, absolute_import
import sys, getopt, os, traceback

from glass.environment import env, Environment
from glass.command import command, Commands
from glass.exmass import * #PointMass
from glass.exceptions import GLInputError

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

_omp_opts = None
def _detect_omp():
    global _omp_opts
    if _omp_opts is not None: return _omp_opts
    try:
        import weave
        kw = dict( extra_compile_args = ['-O3','-fopenmp','-DWITH_OMP','-Wall','-Wno-unused-variable'], 
                   extra_link_args = ['-lgomp'], 
                   headers = ['<omp.h>'] )
        #weave.inline(' ', **kw)
    except:
        kw = {}
    _omp_opts = kw
    return kw

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

@command('Load a glass basis set')
def glass_basis(env, name, **kwargs):
    env.basis_options = kwargs
    f = __import__(name, globals(), locals())
    for name,[f,g,help_text] in Commands.glass_command_list.iteritems():
        if __builtins__.__dict__.has_key(name):
            print 'WARNING: Glass command %s (%s) overrides previous function %s' % (name, f, __builtins__.__dict__[name])
        __builtins__.__dict__[name] = g


def help():
    print >>sys.stderr, "Usage: glass.py <input>"
    sys.exit(2)

if __name__ == "__main__":

    if len(sys.argv) < 2: help()

    Environment.global_opts['ncpus_detected'] = _detect_cpus()
    Environment.global_opts['ncpus'] = 1
    Environment.global_opts['omp_opts'] = _detect_omp()
    Environment.global_opts['withgfx'] = True

    Commands.set_env(Environment())

    optlist, arglist = getopt.getopt(sys.argv[1:], 't:h', ['nw'])

    for opt in optlist:
        if   opt[0] == '-h':
            help()
        elif opt[0] == '-t':
            ncpus = int(opt[1])
            assert ncpus > 0
            #Commands.get_env().ncpus = ncpus
            Environment.global_opts['ncpus'] = ncpus
        elif opt[0] == '--nw':
            #Commands.get_env().withgfx = False
            Environment.global_opts['withgfx'] = False


    if Environment.global_opts['withgfx']:
        import glass.plots 

    import glass.glcmds
    import glass.scales
    #import pytipsy 

    with open(arglist[0], 'r') as f:
        Commands.get_env().input_file = f.read()

    Environment.global_opts['argv'] = arglist
    #Commands.get_env().argv = arglist


    try:
        execfile(arglist[0]) #, globals(), globals())
    except GLInputError as e:
        tb = traceback.extract_tb(sys.exc_traceback, 2)[1]
        #traceback.print_tb(sys.exc_traceback, 2)
        print >>sys.stderr, "Input error on line %i of file '%s':" % (tb[1], tb[0])
        print >>sys.stderr, "> %s" % tb[3]
        print >>sys.stderr
        print >>sys.stderr, e
        print >>sys.stderr


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

