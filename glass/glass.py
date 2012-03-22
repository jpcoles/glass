from __future__ import division, with_statement
import sys, getopt, os, traceback
import numpy
from environment import env, set_env, new_env, glass_command_list, DArray, Environment
import cosmo
from handythread import parallel_map
from log import log as Log, setup_log
from scales import convert
from numpy import abs

GlassEnvironment = Environment


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

def glass_basis(name, **kwargs):
    env().basis_options = kwargs
    f = __import__(name, globals(), locals())
    for name,func in glass_command_list.iteritems():
        if __builtins__.__dict__.has_key(name):
            print 'WARNING: Glass command %s (%s) overrides previous function %s' % (name, func, __builtins__.__dict__[name])
        __builtins__.__dict__[name] = func


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
    Log( pp('H0inv        = %s'   % str_range(convert('nu to H0^-1 in Gyr',_env.nu), '%.4g'), '[Gyr]') )
    Log( pp('H0           = %s'   % str_range(convert('nu to H0 in km/s/Mpc',_env.nu), '%.4g'), '[km/s/Mpc]') )
    Log( pp('H0inv ref    = %s'   % str_range(_env.H0inv_ref, '%.4g'), '[Gyr]') )
    Log( pp('filled_beam  = %s' % _env.filled_beam, '') )
    Log(  )
    Log( '=' * 80 )
    Log( 'OBJECTS' )
    Log( '=' * 80 )
    H0inv_ref_as_nu = convert('H0^-1 in Gyr to nu', _env.H0inv_ref)
    for i,o in enumerate(_env.objects):
        Log( pp('%i. %s at z=%.4g  Distance(Obs->Lens) = %.4f' % (i+1, o.name, o.z, cosmo.angdist(0,o.z)), '') )
        if o.maprad:
            Log( pp('    Map radius       = %.4g' % o.maprad, '[arcsec]') )
            Log( pp('    Map radius       = %.4g (H0inv=%4.1f)' % (convert('arcsec to kpc', o.maprad, o.dL, H0inv_ref_as_nu), _env.H0inv_ref), '[kpc]') )
        else:
            Log( pp('    Map radius       = Not specified', '') )
        #Log( pp('    Time scale            = %.4g' % o.scales['time'],    '[g days/arcsec^2]') )
        #Log( pp('    Angular distance      = %.4g' % o.scales['angdist'], '[g kpc/arcsec]') )
        #Log( pp('    Critical density      = %.4e' % convert('kappa to Msun/arcsec^2', 1, o.dL, '[Msun/arcsec^2]') )
        Log( pp('    Critical density = %.4e (H0inv=%.1f)' \
            % (convert('kappa to Msun/kpc^2', 1, o.dL, H0inv_ref_as_nu), _env.H0inv_ref), '[Msun/kpc^2]') )
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
                Log( '        Image at (% .3f,% .3f) : |*|=% 5.3f angle=% 8.3f parity=%s ' 
                    % (img.pos.real, img.pos.imag, abs(img.pos), img.angle, img.parity_name) )
            #for img in src.images: 
            #    Log( '        Image at (% .3f,% .3f) : angle=% 8.3f parity=%s elongation=[%.4g,%.4g,%.4g]' 
            #        % (img.pos.real, img.pos.imag, img.angle, img.parity_name, img.elongation[0], img.elongation[1], img.elongation[2]) )

    Log(  )
    Log( '=' * 80 )
    Log( 'MISCELLANEOUS' )
    Log( '=' * 80 )
    Log( 'Graphics %s' % ('enabled' if _env.withgfx else 'disabled') )
    Log(  )
    Log( '=' * 80 )
    Log( 'SYSTEM' )
    Log( '=' * 80 )
    Log( 'Number of CPUs detected = %i' % _env.ncpus_detected )
    Log( 'Number of CPUs used     = %i' % _env.ncpus )
    Log( )


def help():
    print >>sys.stderr, "Usage: glass.py <input>"
    sys.exit(2)

if __name__ == "__main__":

    if len(sys.argv) < 2: help()

    optlist, arglist = getopt.getopt(sys.argv[1:], 't:h', ['nw'])
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
        import plots 

    import glcmds
    import scales
    import pytipsy 

    with open(arglist[0], 'r') as f:
        env().input_file = f.read()

    env().argv = arglist

    execfile(arglist[0])
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

