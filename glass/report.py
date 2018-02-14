from glass.log import log as Log
from  glass.cosmo import angdist
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

def pp(str, units, width=80):
    slen = len(str)
    ulen = len(units)
    return str + (' '*(width-ulen-slen)) + units

def report(env):
    Log( '=' * 80 )
    Log( 'COSMOLOGY' )
    Log( '=' * 80 )
    Log( pp('Omega Matter = %.4g' % env.omega_matter, '') )
    Log( pp('Omega Lambda = %.4g' % env.omega_lambda, '') )
    if env.nu: Log( pp('H0inv        = %s'   % str_range(convert('nu to H0^-1 in Gyr',env.nu), '%.4g'), '[Gyr]') )
    if env.nu: Log( pp('H0           = %s'   % str_range(convert('nu to H0 in km/s/Mpc',env.nu), '%.4g'), '[km/s/Mpc]') )
    Log( pp('H0inv ref    = %s'   % str_range(env.H0inv_ref, '%.4g'), '[Gyr]') )
    Log( pp('filled_beam  = %s' % env.filled_beam, '') )
    Log(  )
    Log( '=' * 80 )
    Log( 'OBJECTS' )
    Log( '=' * 80 )
    H0inv_ref_as_nu = convert('H0^-1 in Gyr to nu', env.H0inv_ref)
    for i,o in enumerate(env.objects):
        Log( pp('%i. %s at z=%.4g  Distance(Obs->Lens) = %.4f' % (i+1, o.name, o.z, angdist(env, 0,o.z)), '') )
        if o.maprad:
            Log( pp('    Map radius       = %.4g' % o.maprad, '[arcsec]') )
            Log( pp('    Map radius       = %.4g (H0inv=%4.1f)' % (convert('arcsec to kpc', o.maprad, o.dL, H0inv_ref_as_nu), env.H0inv_ref), '[kpc]') )
        else:
            Log( pp('    Map radius       = Not specified', '') )
        #Log( pp('    Time scale            = %.4g' % o.scales['time'],    '[g days/arcsec^2]') )
        #Log( pp('    Angular distance      = %.4g' % o.scales['angdist'], '[g kpc/arcsec]') )
        #Log( pp('    Critical density      = %.4e' % convert('kappa to Msun/arcsec^2', 1, o.dL, '[Msun/arcsec^2]') )
        Log( pp('    Critical density = %.4e (H0inv=%.1f)' \
            % (convert('kappa to Msun/kpc^2', 1, o.dL, H0inv_ref_as_nu), env.H0inv_ref), '[Msun/kpc^2]') )
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
            Log( pp('        Distance (Obs->Src)  = %.4f' % angdist(env, 0,src.z), '[arcsec]') )
            Log( pp('        Distance (Lens->Src) = %.4f' % angdist(env, o.z,src.z), '[arcsec]') )
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
    Log( 'Graphics %s' % ('enabled' if Environment.global_opts['withgfx'] else 'disabled') )
    Log(  )
    Log( '=' * 80 )
    Log( 'SYSTEM' )
    Log( '=' * 80 )
    Log( 'Number of CPUs detected = %i' % Environment.global_opts['ncpus_detected'] )
    Log( 'Number of CPUs used     = %i' % Environment.global_opts['ncpus'] )
    oo = Environment.global_opts['omp_opts']
    if oo:
        Log( 'OpenMP supported. Compiling with "%s"' % ' '.join(oo['extra_compile_args'] + oo['extra_link_args']) )
    else:
        Log( 'OpenMP not supported.' )
    Log( )


