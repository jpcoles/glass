from __future__ import division
from cosmo import scales

#-------------------------------------------------------------------------------
# Density conversions
#-------------------------------------------------------------------------------

def density_to_internal(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    return v / obj.scales['critden'] / H0inv

def density_to_physical(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    return v * obj.scales['critden'] * H0inv

#-------------------------------------------------------------------------------
# Distance conversions
#-------------------------------------------------------------------------------

def distance_to_internal(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    return v / obj.scales['angdist'] / H0inv

def distance_to_physical(*args):
    if len(args) == 2: [obj, data], v = args; H0inv = data['1/H0']
    else:              obj,v,H0inv    = args
    return v * obj.scales['angdist'] * H0inv

#-------------------------------------------------------------------------------
# Time conversions
#-------------------------------------------------------------------------------

def time_to_physical(obj, v):
    return 1 / v / obj.scales['time']

def time_to_internal(obj, v):
    return 1 / v / obj.scales['time']

