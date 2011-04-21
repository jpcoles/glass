from __future__ import division
from environment import command
from cosmo import scales

N = 206265
Munit = 11.988 # Msun

def get_args(args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return obj,v,H0inv

@command
def convert(type, *args):

    s,d = type.split('to')
    if s.strip() == d.strip(): return args[0]
    
    if 'ly to rad' == type:
        v, dL, nu = args; return nu * v / dL / N**2
    elif 'rad to ly' == type:
        v, dL, nu = args; return nu / v * dL * N**2

    elif 'ly to arcsec' == type:
        v, dL, nu = args; return v * nu / dL / N
    elif 'arcsec to ly' == type:
        v, dL, nu = args; return v / nu * dL * N

    elif 'arcsec^2 to days' == type:
        v, zL, nu = args; 
        return convert('years to days', v / nu * (1+zL))

    elif 'kpc to ly' == type:
        return args[0] * 3.26e3
    elif 'ly to kpc' == type:
        return args[0] / 3.26e3

    elif 'arcsec to kpc' == type:
        return convert('ly to kpc', convert('arcsec to ly', *args))
        
    elif 'kpc to arcsec' == type:
        v, dL, nu = args; 
        return convert('ly to arcsec', convert('kpc to ly', v), dL, nu)

    elif 'days to years' == type:
        return args[0] / 365.25

    elif 'years to days' == type:
        return args[0] * 365.25

    elif 'age in Gyr to nu' == type:
        v, age_factor = args
        return (N**2 / 1e9) / args[0] * age_factor

    elif 'H0 in km/s/Mpc to nu' == type:
        return N**2 * args[0]

    elif 'H0^-1 in Gyr to nu' == type:
        return (N**2 / 1e9) / args[0]

    elif 'nu to H0^-1 in Gyr' == type:
        return (N**2 / 1e9) / args[0]

    elif 'Msun/ly^2 to kappa' == type:
        v, dL, nu = args
        return v / Munit * dL / nu

    elif 'kappa to Msun/ly^2' == type:
        v, dL, nu = args
        return v * Munit / dL * nu

    elif 'Msun/kpc^2 to kappa' == type:
        return convert('Msun/ly^2 to kappa', *args) / convert('kpc to ly', 1)**2 

    elif 'kappa to Msun/kpc^2' == type:
        return convert('kappa to Msun/ly^2', *args) / convert('ly to kpc', 1)**2

    elif 'kappa to Msun/arcsec^2' == type:
        v, dL, nu = args
        return convert('kappa to Msun/ly^2', *args) / convert('ly to arcsec', 1, dL, nu)**2

    elif 'den_stel to den_lum' == type:
        v, dL, nu = args
        return v * ((nu / N**2) / dL) ** 2

    else:
        assert 0, 'Unknown conversion "%s"' % type
        
    
#-------------------------------------------------------------------------------
# Density conversions
#-------------------------------------------------------------------------------

def density_to_internal(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return v / obj.scales['critden'] / H0inv

def density_to_physical(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return v * obj.scales['critden'] * H0inv

def MsunKpc2_to_MsunArcsec2(*args):
    pass

def MsunKpc2_to_KappaKpc2(*args):
    pass

def get_args(args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return obj,v,H0inv

def MsunKpc2_to_Kappa(*args):
    obj,v,H0inv = get_args(args)
    return v * (obj.scales['angdist'] * H0inv)**2 \
             / (obj.scales['critden'] * H0inv)

def Kappa_to_MsunKpc2(*args):
    obj,v,H0inv = get_args(args)
    return v *  obj.scales['critden'] \
             / (obj.scales['angdist']**2 * H0inv)

#   return v / (obj.scales['angdist'] * H0inv)**2 \
#            * (obj.scales['critden'] * H0inv)

def Kappa_to_MsunArcsec2(*args):
    obj,v,H0inv = get_args(args)
    return v * (obj.scales['critden'] * H0inv)

#-------------------------------------------------------------------------------
# Distance conversions
#-------------------------------------------------------------------------------

def distance_to_internal(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return v / obj.scales['angdist'] / H0inv

def distance_to_physical(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return v * obj.scales['angdist'] * H0inv

def Kpc_to_Arcsec(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return v / obj.scales['angdist'] / H0inv

def Arcsec_to_Kpc(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    assert obj.scales, "Scaling requires that zlens() be set first."
    return v * obj.scales['angdist'] * H0inv

#-------------------------------------------------------------------------------
# Time conversions
#-------------------------------------------------------------------------------

def time_to_physical(obj, v):
    assert obj.scales, "Scaling requires that zlens() be set first."
    return v / obj.scales['time']
    #return 1 / v / obj.scales['time']

def time_to_internal(obj, v):
    assert obj.scales, "Scaling requires that zlens() be set first."
    return 1 / v / obj.scales['time']

def Days_to_Expansion(obj, v):
    assert obj.scales, "Scaling requires that zlens() be set first."
    return 1 / v / obj.scales['time']

def Expansion_to_Days(obj, v):
    assert obj.scales, "Scaling requires that zlens() be set first."
    return 1 / v / obj.scales['time']

