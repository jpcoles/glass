from __future__ import division
import sys
import numpy
import gc
from numpy import isfortran, asfortranarray, sign, logical_and, any
from numpy import set_printoptions
from numpy import insert, zeros, vstack, append, hstack, array, all, sum, ones, delete, log, empty, dot, sqrt, arange
from numpy import argwhere, argmin, inf, isinf
from numpy import histogram, logspace, flatnonzero, isinf
from numpy.random import random, normal, random_integers, seed as ran_set_seed
#from glrandom import random, ran_set_seed

from lpsolve55 import *

if 0:
    from pylab import figimage, show, imshow, hist, matshow, figure

from log import log as Log

from copy import deepcopy

set_printoptions(linewidth=10000000, precision=20, threshold=2000)

class SamplexUnboundedError:
    def __init__(self, msg=""):
        pass

class SamplexNoSolutionError:
    def __init__(self, msg=""):
        pass

class SamplexUnexpectedError:
    def __init__(self, msg=""):
        pass

class SamplexSolution:
    def __init__(self):
        self.lhv = None
        self.vertex = None

class Samplex:
    #INFEASIBLE, FEASIBLE, NOPIVOT, FOUND_PIVOT, UNBOUNDED = range(5)
    SML = 1e-6
    EPS = 1e-14

    def __init__(self, **kw):

        ncols    = kw.get('ncols', None)
        nthreads = kw.get('nthreads', 1)
        rngseed  = kw.get('rngseed',  0)
        self.objf_choice = kw.get('objf choice', 'facet')
        self.sol_type  = kw.get('solution type', 'interior')

        self.lp = lpsolve('make_lp', 0, ncols)
        lpsolve('set_verbose', self.lp, IMPORTANT)


        Log( "Samplex created" )
        Log( "    ncols = %i" % ncols )
        if ncols is not None:
            self.nVars = ncols
            self.nRight = self.nVars

        ran_set_seed(rngseed)
        self.random_seed = rngseed

        self.nthreads = nthreads

        self.data = None
        self.dcopy = []

        self.ineqs = []

        self.n_equations = 0
        self.lhv = []
        self.rhv = []
        self.nVars = None            # Number of variables + 1(constant column) [N]
        self.nLeft = 0               # Number of left hand variables            [L]
        self.nSlack = 0              # Number of slack variables                [S]
        self.nTemp = 0               # Number of temporary variables            [Z]
        self.nRight = 0              # Number of right hand variables           [R]
        self.eq_count = 0
        self.leq_count = 0
        self.geq_count = 0

        self.eq_list = []

        self.iteration = 0
        self.prev_sol = None
        self.sum_ln_k = 0
        self.curr_sol = None
        self.n_solutions = 0

        self.forbidden_variables = []

        print self.sol_type
        self.prepare_return_sol = {'vertex':   self.current_sol_copy,
                                   'interior': self.interior_point}[self.sol_type]

    def current_sol_copy(self):
        return self.curr_sol.sol.copy()

    def __del__(self):
        lpsolve('delete_lp', self.lp)


    def check_data_shape(self, len=None):

        if len is None: len = self.nLeft

        if self.data.shape[0] == len: 
            self.data = insert(self.data, self.data.shape[0], 0, axis=0)

    def start(self):
        #print "%6s %6s %6s\n%6i %6i %6i" \
        #    % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count)

        Log( '=' * 80 )
        Log( 'SAMPLEX' )
        Log( '=' * 80 )

        Log( "random seed = %s" % self.random_seed )
        Log( "threads = %s" % self.nthreads )

        Log( "%6s %6s %6s\n%6i %6i %6i" 
            % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count) )

    def status(self):
        if self.iteration & 15 == 0:
            Log( "model %i]  iter % 5i  obj-val %f" % (self.n_solutions+1, self.iteration, self.data[0,0]) )

    def next(self, nsolutions=None):

        Log( "Getting solutions" )

        self.start_new_objective('random')
        res = lpsolve('solve', self.lp)
        print 'solve result', res

        if res != OPTIMAL: return

        Log( "------------------------------------" )
        Log( "Found feasible" )
        Log( "------------------------------------" )

        self.curr_sol = self.package_solution()                
        self.prev_sol = self.curr_sol.vertex.copy()

        self.sum_ln_k = 0
        self.n_solutions = 0
        while self.n_solutions != nsolutions:
            self.iteration=0
            self.n_solutions += 1
            while True:
                self.next_solution()
                self.curr_sol = self.package_solution()                

                p = self.prepare_return_sol()

                if p is not None: 
                    break
                
                print 'SAME VERTEX!'

            yield p

    def next_solution(self):

        while True:

            r = self.start_new_objective(kind=self.objf_choice)

            result = lpsolve('solve', self.lp)
            if   result == OPTIMAL:   break
            elif result == INFEASIBLE: raise SamplexNoSolutionError()
            elif result == UNBOUNDED: raise SamplexUnboundedError()
            else:
                Log( result )
                raise SamplexUnexpectedError("unknown pivot result = %i" % result)

            objv = lpsolve('get_objective', self.lp)

            if self.objf_choice == 'facet' and abs(objv) > 1e-6:
                print 'BAD VARIABLE', objv
                #self.forbidden_variables.append(r)
                del self.ineqs[r]
            else:
                break


    def package_solution(self):
        objv  = array(lpsolve('get_objective', self.lp))
        vars  = array(lpsolve('get_variables', self.lp)[0])
        slack = array(lpsolve('get_constraints', self.lp)[0])

        slack -= array(lpsolve('get_rh', self.lp)[1:])
        slack[abs(slack) < 1e-10] = 0

        nvars = len(vars)
        nslack = len(slack)

        #print nvars, nslack
        #print 'vars', vars
        #print slack

        s = SamplexSolution()
        s.sol = empty(nvars + 1)
        s.sol[1:] = vars
        s.sol[0] = objv

        s.vertex = empty(nvars + nslack + 1)
        s.vertex[1:nvars+1] = vars
        s.vertex[1+nvars:1+nvars+nslack] = slack
        s.vertex[0] = objv

        return s


    def interior_point(self, r=None):
        if r is None: r = random()

        sol = self.curr_sol

        k = 0
        smallest_scale = inf

        iv    = sol.vertex[1:]
        dist  = iv - self.prev_sol[1:]
        a = dist > self.SML
        if not any(a): return None #self.prev_sol.copy(), 'g'

        #print '*', iv
        #print '*', dist[a]

        scale = iv[a] / dist[a]

        smallest_scale = min(scale)

        assert not isinf(smallest_scale)
        assert smallest_scale > 0.99, smallest_scale

        k = smallest_scale * (1.0-r)

        self.sum_ln_k += log(k)
        #assert self.sum_ln_k < 1

        #old_prev_sol = self.prev_sol.copy()

        self.prev_sol[1:] = sol.vertex[1:] + k * (self.prev_sol[1:]-sol.vertex[1:])
        assert all(self.prev_sol[1:] >= -self.SML), (self.prev_sol[self.prev_sol < 0], self.prev_sol)

        s = self.prev_sol.copy()[:sol.sol.size]
        return s

    #=========================================================================

    def start_new_objective(self, kind=2, last_r=-1):

        if kind=='random':

            lpsolve('set_obj_fn', self.lp, random(lpsolve('get_Ncolumns', self.lp)) - 0.5)

        elif kind=='facet':

            r = self.ineqs[random_integers(len(self.ineqs))-1]
            lpsolve('set_obj_fn', self.lp, r[0])
            return r

    #=========================================================================

    def eq(self, a):
        lpsolve('add_constraint', self.lp, a[1:], EQ, -a[0])
        self.eq_count += 1

    def geq(self, a):
        lpsolve('add_constraint', self.lp, a[1:], GE, -a[0])
        self.geq_count += 1
        self.ineqs.append([a[1:], GE, -a[0]])

    def leq(self, a):
        #
        # We convert <= constraints to >= so that in package_solution we can simply subtract the
        # current contraint value in the tableau from the original right hand side values given
        # here to derive the amount of slack on each constraint. This is important to have in 
        # interior_point()
        #
        lpsolve('add_constraint', self.lp, -a[1:], GE, a[0])
        self.leq_count += 1
        self.ineqs.append([-a[1:], GE, a[0]])
        #self.ineqs.append([a[1:], LE, -a[0]])


