import os, platform, time
import textwrap
from glass.log import log as Log
from glass.cosmo import angdist, cosmo_params
from glass.scales import convert
from glass.environment import Environment


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

#def pp(str, units, width=80):
#    slen = len(str)
#    ulen = len(units)
#    return str + (' '*(width-ulen-slen)) + units

def pp(str, width=80):
    s = str.split('&')
    if len(s) <= 1: return ' '.join(s)
    t = [s[0].rstrip()]
    t += list(map(lambda x: x.strip(), s[1:-2]))
    t += [s[-1].lstrip()] if len(s) > 1 else []
    if len(t) <= 1: return ' '.join(t)
    f = ''.join(t)
    fill = width-len(f)
    if fill <= 0: return ' '.join(s)
    f = ''.join(t)
    q,r = divmod(fill, len(t)-1)
    for i in range(r):
        t[i] += ' '
    return (' '*q).join(t)

def section(text):
    Log( '=' * 80 )
    Log( text.center(80) )
    Log( '=' * 80 )

def report(env):
    banner=r'''  
                 ██████╗ ██╗      █████╗ ███████╗███████╗
                ██╔════╝ ██║     ██╔══██╗██╔════╝██╔════╝
                ██║  ███╗██║     ███████║███████╗███████╗
                ██║   ██║██║     ██╔══██║╚════██║╚════██║
                ╚██████╔╝███████╗██║  ██║███████║███████║
                 ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝
                                                   v1.1

                 Gravitational Lensing AnalysiS Software 

                        Written by Jonathan Coles

                   See AUTHORS for additional credits.
         '''
    Log( textwrap.indent(textwrap.dedent(banner), ' '*20) )
    Log(  )

    section( 'CONFIGURATION' )
    Log( 'Date              : %s' % time.asctime() )
    Log( 'Input file        : %s' % Environment.global_opts['argv'][0] )
    Log( 'Hostname          : %s' % platform.node())
    Log( 'SLURM job name    : %s' % os.environ['SLURM_JOB_NAME'] ) if 'SLURM_JOB_NAME' in os.environ else 0
    Log( 'SLURM job id      : %s' % os.environ['SLURM_JOB_ID']   ) if 'SLURM_JOB_ID'   in os.environ else 0
    Log( 'GLASS version     : 1.1' )
    Log( 'CPUs detected     : %i' % Environment.global_opts['ncpus_detected'] )
    Log( 'CPUs used         : %i' % Environment.global_opts['ncpus'] )
    Log( 'Threads used (-t) : %i' % Environment.global_opts['nthreads'] )
    Log( 'Graphics %s' % ('enabled' if Environment.global_opts['withgfx'] else 'disabled') )
    oo = Environment.global_opts['omp_opts']
    if oo:
        Log( 'OpenMP supported. Compiling with "%s"' % ' '.join(oo['extra_compile_args'] + oo['extra_link_args']) )
    else:
        Log( 'OpenMP not supported.' )
    Log(  )

    section( 'COSMOLOGY' )
    cp = cosmo_params(env)
    Log( pp('Omega Matter = %.4g' % cp['M']) )
    Log( pp('Omega Lambda = %.4g' % cp['L']) )
    Log( pp('%s universe (k=%i)' % ({0: 'Flat', 1: 'Open', '-1': 'Closed'}[cp['k']], cp['k']) ))
    Log( pp('filled_beam  = %s' % cp['filled_beam']) )
    Log( pp('1/H0         = %s & [Gyr]'      % str_range(convert('nu to H0^-1 in Gyr',env.nu), '%.4g')) ) if env.nu is not None else 0 
    Log( pp('H0           = %s & [km/s/Mpc]' % str_range(convert('nu to H0 in km/s/Mpc',env.nu), '%.4g')) ) if env.nu is not None else 0 
    Log( pp('1/H0 ref     = %s & [Gyr]'      % str_range(env.H0inv_ref, '%.4g')) )
    Log( pp('H0 ref       = %s & [km/s/Mpc]' % str_range(1000./env.H0inv_ref, '%.4g')) )
    Log(  )

    section( 'OBJECTS' )
    ref_nu = convert('H0^-1 in Gyr to nu', env.H0inv_ref)
    for i,o in enumerate(env.objects):
        Log( pp('%i. %s at z=%.4g  Distance(Obs->Lens) = %.4f' % (i+1, o.name, o.z, angdist(env, 0,o.z))) )
        if o.maprad:
            Log( pp('    Map radius       = %.4g & [arcsec]' % o.maprad) )
            Log( pp('    Map radius       = %.4g (for 1/H0=%4.1f [Gyr]) & [kpc]' % (convert('arcsec to kpc', o.maprad, o.dL, ref_nu), env.H0inv_ref)) )
        else:
            Log( pp('    Map radius       = Not specified') )
        #Log( pp('    Time scale            = %.4g' % o.scales['time'],    '[g days/arcsec^2]') )
        #Log( pp('    Angular distance      = %.4g' % o.scales['angdist'], '[g kpc/arcsec]') )
        #Log( pp('    Critical density      = %.4e' % convert('kappa to Msun/arcsec^2', 1, o.dL, '[Msun/arcsec^2]') )
        Log( pp('    Critical density = %.4e (for 1/H0=%.1f [Gyr]) & [Msun/kpc^2]' \
            % (convert('kappa to Msun/kpc^2', 1, o.dL, ref_nu), env.H0inv_ref)) )
#       if o.shear:
#           pass
#           #Log( pp('    Shear                 = %.4g' % o.shear.phi, '') )
#       else:
#           Log( pp('    NO SHEAR', '') )
#           #Log( pp('    Shear                 = Not specified', '') )
#       Log( pp('    Steepness             = %s' % str_range(o.steep, '%.4g'), '') )
        Log( )
        for src in o.sources:
            Log( '    Source at z=%.4f %s' % (src.z, '[NO IMAGES]' if len(src.images) == 0 else '' ))
            Log( pp('        Distance (Obs->Src)  = %.4f & [arcsec]' % angdist(env, 0,src.z)) )
            Log( pp('        Distance (Lens->Src) = %.4f & [arcsec]' % angdist(env, o.z,src.z)) )
            Log( pp('        Dos/Dls              = %.4f' % src.zcap) )
            for img in src.images: 
                Log( '        Image at (% .3f,% .3f) : |*|=% 5.3f angle=% 8.3f parity=%s ' 
                    % (img.pos.real, img.pos.imag, abs(img.pos), img.angle, img.parity_name) )
            #for img in src.images: 
            #    Log( '        Image at (% .3f,% .3f) : angle=% 8.3f parity=%s elongation=[%.4g,%.4g,%.4g]' 
            #        % (img.pos.real, img.pos.imag, img.angle, img.parity_name, img.elongation[0], img.elongation[1], img.elongation[2]) )

        Log( )
        o.basis.report()

    Log( )


