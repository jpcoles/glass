from __future__ import division
from itertools import izip
from numpy import mean, zeros
from priors import include_prior, exclude_prior, \
                   def_priors, all_priors, inc_priors, exc_priors, acc_objpriors, acc_enspriors
from log import log as Log
from environment import env, Object, command

import glcmds
import funcs
import priors
from funcs import default_post_process

opts = env().basis_options
if opts.has_key('solver') and opts['solver'] is None:
    pass
elif not opts.has_key('solver') or opts['solver'] == 'rwalk':
    from solvers.rwalk.samplex import Samplex
    import solvers.rwalk.glcmds 
elif not opts.has_key('solver') and opts['solver'] == 'samplex':
    from solvers.samplex.samplex import Samplex
    import solvers.samplex.glcmds 
elif opts.has_key('solver') and opts['solver'] == 'lpsolve':
    from solvers.lpsolve.samplex import Samplex
    import solvers.lpsolve.glcmds 
elif opts.has_key('solver') and opts['solver'] == 'samplexsimple':
    from solvers.samplexsimple.samplex import Samplex
    import solvers.samplexsimple.glcmds 
elif opts.has_key('solver') and opts['solver'] == 'samplexsimple2':
    from solvers.samplexsimple.samplex2 import Samplex
    import solvers.samplexsimple.glcmds 
else:
    assert 0, 'Unknown solver %s' % opts['solver']


if env().withgfx:
    import plots

def _expand_array(nvars, offs, f):
    """Returns a function that will properly prepare a constraint equation
       for the solver. The solver expects all equations to be of the same
       length and span the entire range of variables over all objects. We
       stack the input equations such that the first column of the input
       always goes in the first column for the solver but the rest is
       shifted to the right by offs, which places the input into a region
       of the solver matrix that is just for the same object."""
    def work(eq):
        new_eq = zeros(nvars+1, order='Fortran')
        new_eq[0] = eq[0]
        new_eq[offs+1:offs+len(eq)] = eq[1:]
        f(new_eq)

    return work

def init_model_generator(nmodels, regenerate=False):
    """Construct the linear constraint equations by applying all the
       enabled priors."""

    objs = env().objects

    # ------------- 

    nvars = reduce(lambda s,o: s+o.basis.nvar, objs, 0)
    Log( "Number of variables (nvars) = %i" % nvars )


    offs = 0
    for o in objs:
        o.basis.array_offset = 1+offs
        offs += o.basis.nvar 

    # ------------- 

    Log( '=' * 80 )
    Log( 'PIXEL BASIS MODEL GENERATOR' )
    Log( '=' * 80 )
#   if nmodels == 0:
#       Log( "No models requested." )
#       return

    #---------------------------------------------------------------------------
    # Decide which priors to use. The initial list is the list of default
    # priors. The user can then modify this list be selecting which priors
    # should be included from the entire list, or which ones should be excluded.
    #
    #---------------------------------------------------------------------------
    priors = def_priors

    if exc_priors:
        priors = filter(lambda x: x not in exc_priors, priors)

    if inc_priors:
        priors += filter(lambda x: x not in priors, inc_priors)


    Log( 'Priors:' )
    for p in all_priors:
        Log( '%10s %s' % ('[EXCLUDED]' if p not in priors else '', p.f.__name__) )

    lp = filter(lambda x: x.where == 'object_prior',   priors)
    gp = filter(lambda x: x.where == 'ensemble_prior', priors)

    #---------------------------------------------------------------------------
    # Initialize our model generator, the simplex.
    #---------------------------------------------------------------------------
    opts = env().model_gen_options
    opts['ncols'] = nvars
    if not opts.has_key('nthreads'): opts['nthreads'] = env().ncpus

    #mg = env().model_gen = env().model_gen_factory(env().model_gen_options)
    mg = env().model_gen = Samplex(**env().model_gen_options)

    #---------------------------------------------------------------------------
    # Apply the object priors
    #---------------------------------------------------------------------------
    Log( 'Applying Priors:' )
    for o in objs:
        offs = o.basis.array_offset - 1
        Log( 'array offset %i' % offs )
        for p in lp:
            leq = _expand_array(nvars, offs, mg.leq)
            eq  = _expand_array(nvars, offs, mg.eq)
            geq = _expand_array(nvars, offs, mg.geq)
            p.f(o, leq, eq, geq)

    #---------------------------------------------------------------------------
    # Apply the ensemble priors
    #---------------------------------------------------------------------------
    for p in gp:
        p.f(objs, nvars, mg.leq, mg.eq, mg.geq)


    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    global acc_objpriors
    global acc_enspriors
    del acc_objpriors[:]
    del acc_enspriors[:]

    acc_objpriors += lp
    acc_enspriors += gp

def solution_to_dict(obj, sol):
    return obj.basis.solution_to_dict(sol)

def _model_dict(objs, sol):
    if isinstance(objs, Object):
        objs = [objs]

    print objs
    print sol
    return {'sol':      sol,
            'obj,data': zip(objs, map(lambda x: solution_to_dict(x, sol), objs)),
            'tagged':   False}

@command
def package_solution(sol, objs, fn_package_sol = None):
    if fn_package_sol is None:
        fn_package_sol = lambda x: solution_to_dict(x, sol)
    
    return {'sol':  sol,
            'obj,data': zip(objs, map(fn_package_sol, objs)),
            'tagged':  False}

def check_model(objs, sol):
    for o in objs:
        for p in acc_objpriors:
            if p.check: p.check(o, sol)
    for p in acc_enspriors:
        if p.check: p.check(objs, sol)

@command
def generate_models(objs, n, *args, **kwargs):

    #if n <= 0: return

    mode = kwargs.get('mode', 'default')

    if mode == 'particles':
        assert n==1, 'Can only generate a single model in particles mode.'
        assert len(objs) == 1, 'Can only model a single object from particles.'
        data = kwargs.get('data', None)
        assert data is not None, 'data keyword must be given with model parameters.'
        objs[0].basis.array_offset = 1
        ps = _particle_model(objs[0], *data)

        if opts.get('solver', None):
            init_model_generator(n)
            check_model(objs, ps['sol'])

        yield ps

    elif mode == 'grid':
        assert n==1, 'Can only generate a single model in grid mode.'
        assert len(objs) == 1, 'Can only model a single object from a grid.'
        data = kwargs.get('data', None)
        assert data is not None, 'data keyword must be given with model parameters.'
        objs[0].basis.array_offset = 1
        ps = _grid_model(objs[0], *data)

        if opts.get('solver', None):
            init_model_generator(n)
            check_model(objs, ps['sol'])

        yield ps

    elif mode == 'isothermal':
        assert n==1, 'Can only generate a single model in isothermal mode.'
        assert len(objs) == 1, 'Can only model a single object from isothermal.'
        data = kwargs.get('data', None)
        assert data is not None, 'data keyword must be given with model parameters.'
        objs[0].basis.array_offset = 1

        ps = objs[0].basis.solution_isothermal(*data)
        yield package_solution(ps, [objs[0]])

    elif mode != 'default':
        assert False, 'Unsupported model mode "%s"' % mode
    else:

        if opts.get('solver', None):
            init_model_generator(n)
            mg = env().model_gen
            mg.start()
            for sol in mg.next(n):
                check_model(objs, sol)
                yield package_solution(sol, objs)

    for o in objs:
        o.post_process_funcs.append([default_post_process, [], {}])

def regenerate_models(objs):

    assert env().solutions is not None

    init_model_generator(len(env().solutions))

    for sol in env().solutions:
        yield package_solution(sol, objs)
#       yield {'sol':  sol,
#              'obj,data': zip(objs, map(lambda x: solution_to_dict(x, sol), objs)),
#              'tagged':  False}

@command
def make_ensemble_average():
#   Log( "s*********" )
#   for m in env().models:
#       Log( m['sol'] )
#   Log( "s*********" )


    sol = mean([m['sol'] for m in env().models], axis=0)
    objs = env().objects
    env().ensemble_average = package_solution(sol, objs)
    for od in env().ensemble_average['obj,data']:
        default_post_process(od)

def _grid_model(obj, grid, grid_size, src, H0inv):
    grid_mass = obj.basis.grid_to_grid(grid, grid_size, H0inv)
    ps = obj.basis.solution_from_grid(grid_mass, src=src, H0inv=H0inv)
    return package_solution(ps, [obj])

def _particle_model(obj, X,Y,M, src, H0inv):

    grid_mass = obj.basis.grid_mass(X,Y,M, H0inv)
    ps = obj.basis.solution_from_grid(grid_mass, src=src, H0inv=H0inv)
    return package_solution(ps, [obj])
    #return package_solution(ps, [obj], fn_package_sol = lambda x:ps)
#   return {'sol':      ps,
#           'obj,data': [[obj,ps]],
#           'tagged':   False}


@command 
def change_source(models, src, invalidate=True):
    for m in models:
        assert len(src) == len(m['obj,data'])
        for [obj,data],s in izip(m['obj,data'], src):
            #if not isinstance(s, complex):
                #raise ValueError('The new source position must be given as a complex number or tuple')
            data['src'] = s

    if invalidate:
        for m in models:
            for n in ['srcdiff', 'srcdiff_grid', 'arrival_grid', 'arrival_contour_levels', 'time_delays']:
                if m['obj,data'][0][1].has_key(n):
                    del m['obj,data'][0][1][n]


