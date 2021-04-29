from glass.environment import env, Environment
from glass.command import command, Commands
from glass.exmass import * #PointMass
from glass.exceptions import GLInputError
import glass.plots 
import glass.glcmds
import glass.scales
import builtins

__all__ = ['cosmo', 'environment', 'filters', 'glcmds', 'plots',\
           'potential', 'scales', 'shear', 'spherical_deproject', \
           'sigp', 'exmass']

@command('Load a glass basis set')
def glass_basis(env, name, **kwargs):
    env.basis_options = kwargs
    f = __import__(name, globals(), locals())
    for name,[f,g,help_text] in Commands.glass_command_list.items():
        #if name in __builtins__.__dict__:
        if name in builtins.__dict__:
            print('WARNING: Glass command %s (%s) overrides previous function %s' % (name, f, builtins.__dict__[name]))
        builtins.__dict__[name] = g


