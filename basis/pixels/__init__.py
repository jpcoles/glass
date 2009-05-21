from __future__ import division
from glcmds import *
from priors import *
from funcs import *

cons = 0
def _expand_array(nvars, offs, f):
    """Returns a function that will properly prepare a constraint equation
       for the solver. The solver expects all equations to be of the same
       length and span the entire range of variables over all objects. We
       stack the input equations such that the first column of the input
       always goes in the first column for the solver but the rest is
       shifted to the right by offs, which places the input into a region
       of the solver matrix that is just for the same object."""
    def work(eq):
        new_eq = zeros(nvars, order='Fortran')
        new_eq[0] = eq[0]
        new_eq[offs+1:offs+len(eq)] = eq[1:]
        global cons
        #print "EQ", cons, f
        cons += 1
        #if eq[0] == 12.86: print "HAHAHAH"
        f(new_eq)

    return work

def init_model_generator():
    """Construct the linear constraint equations by applying all the
       enabled priors."""

    objs = env.objects
    lp = filter(lambda x: x.where == 'object_prior',  all_priors)
    gp = filter(lambda x: x.where == 'ensemble_prior', all_priors)

    nvars = reduce(lambda s,o: s+o.basis.nvar, objs, 0) - len(objs) + 1

    mg = env.model_gen = env.model_gen_factory(nvars)

    print "nvars=",nvars

    #lp = [smooth,steepness ]

    offs = 0
    #obj_offs = [offs]
    for o in objs:
        o.basis.array_offset = offs
        for p in lp:
            leq = _expand_array(nvars, offs, mg.leq)
            eq  = _expand_array(nvars, offs, mg.eq)
            geq = _expand_array(nvars, offs, mg.geq)
            p.f(o, leq, eq, geq)
        offs += o.basis.nvar - 1
        #obj_offs.append(offs)

    for p in gp:
        p.f(objs, nvars, mg.leq, mg.eq, mg.geq)

def packaged_solution(obj, sol):
    return obj.basis.packaged_solution(sol)

def generate_models(objs, n):
    mg = env.model_gen
    
    mg.start()
    for sol in mg.next(n):
        yield [sol, zip(objs, map(lambda x: packaged_solution(x, sol), objs))]

