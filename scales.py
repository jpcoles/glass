from __future__ import division
from cosmo import scales

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
    return v / (obj.scales['angdist'] * H0inv)**2 \
             * (obj.scales['critden'] * H0inv)

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
    return v * obj.scales['time']
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

