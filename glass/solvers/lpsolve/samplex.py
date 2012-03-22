from __future__ import division
from numpy import isfortran, asfortranarray, sign, logical_and, any, amin
from numpy import set_printoptions
from numpy import zeros, array, all, log, empty
from numpy import inf, isinf
from numpy.random import random, random_integers, seed as ran_set_seed

import lpsolve55 as lp
from lpsolve55 import lpsolve, EQ,GE,LE
from lpsolve55 import NORMAL, DETAILED, FULL, IMPORTANT

from lpsolve55 import NOMEMORY, OPTIMAL, SUBOPTIMAL, INFEASIBLE
from lpsolve55 import UNBOUNDED, DEGENERATE, NUMFAILURE, USERABORT, TIMEOUT, PRESOLVED

from log import log as Log

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

    SML = 1e-6

    def __init__(self, **kw):

        self.ncols       = kw.get('ncols', None)
        self.nthreads    = kw.get('nthreads', 1)
        self.random_seed = kw.get('rngseed',  None)
        self.objf_choice = kw.get('objf choice', 'random')
        self.sol_type    = kw.get('solution type', 'interior')
        self.add_noise   = kw.get('add noise', False)

        ran_set_seed(self.random_seed)

        self.lp = lpsolve('make_lp', 0, self.ncols)
        #lpsolve('set_bounds_tighter', self.lp, True) # important so that we don't loosen tight bounds

        self.ineqs = []

        self.eq_count  = 0
        self.leq_count = 0
        self.geq_count = 0
        self.bnd_count  = 0

        self.iteration   = 0
        self.prev_sol    = None
        self.sum_ln_k    = 0
        self.curr_sol    = None
        self.n_solutions = 0

    def prepare_return_sol(self):
        if self.sol_type == 'vertex':
            return self.curr_sol.sol.copy()
        elif self.sol_type == 'interior':
            return self.interior_point()

#   def current_sol_copy(self):
#       return self.curr_sol.sol.copy()

    def __del__(self):
        if self.lp:
            lpsolve('delete_lp', self.lp)

    def start(self):

        Log( '=' * 80 )
        Log( 'SAMPLEX' )
        Log( '=' * 80 )

        Log( 'Using lpsolve %s' % lpsolve('lp_solve_version') )

        Log( "ncols         = %i" % self.ncols )
        Log( "random seed   = %s" % self.random_seed )
        Log( "threads       = %s" % self.nthreads )
        Log( "objf choice   = %s" % self.objf_choice )
        Log( "solution type = %s" % self.sol_type )
        Log( "add noise     = %s" % self.add_noise )

        Log( "%6s %6s %6s %6s\n%6i %6i %6i %6i" 
            % (">=", "<=", "=", "bnd", self.geq_count, self.leq_count, self.eq_count, self.bnd_count) )

        #lpsolve('set_verbose', self.lp, DETAILED)
        #lpsolve('set_verbose', self.lp, DETAILED)
        lpsolve('set_verbose', self.lp, IMPORTANT)

    def status(self):
        if self.iteration & 15 == 0:
            Log( "model %i]  iter % 5i  obj-val %f" % (self.n_solutions+1, self.iteration, self.data[0,0]) )

    def next(self, nsolutions=None):

        Log( "Getting solutions" )

        self.start_new_objective('random')
        #lpsolve('set_simplextype', self.lp, lp.SIMPLEX_DUAL_DUAL)
        #lpsolve('set_simplextype', self.lp, lp.SIMPLEX_PRIMAL_PRIMAL)
        #lpsolve('set_pivoting', self.lp, PRICER_DANTZIG)
        #lpsolve('set_pivoting', self.lp, lp.PRICER_DANTZIG | lp.PRICE_ADAPTIVE)
        #lpsolve('set_pivoting', self.lp, lp.PRICER_STEEPESTEDGE | lp.PRICE_ADAPTIVE)
        #lpsolve('set_presolve', self.lp, PRESOLVE_LINDEP)
        #lpsolve('set_timeout', self.lp, 0)
        res = lpsolve('solve', self.lp)
        #lpsolve('set_presolve', self.lp, PRESOLVE_NONE)
        restxt = {NOMEMORY:          'NOMEMORY',
                  OPTIMAL:           'OPTIMAL',
                  SUBOPTIMAL:        'SUBOPTIMAL',
                  INFEASIBLE:        'INFEASIBLE',
                  UNBOUNDED:         'UNBOUNDED',
                  DEGENERATE:        'DEGENERATE',
                  NUMFAILURE:        'NUMFAILURE',
                  USERABORT:         'USERABORT',
                  TIMEOUT:           'TIMEOUT',
                  PRESOLVED:         'PRESOLVED'}[res]

        print 'solve result %s (%i)' % (restxt, res)

        if res != OPTIMAL: return

        #lpsolve('set_timeout', self.lp, 1)

        Log( "------------------------------------" )
        Log( "Found feasible" )
        Log( "------------------------------------" )

        self.curr_sol = self.package_solution()                
        self.prev_sol = self.curr_sol.vertex.copy()

        if self.sol_type in ['vertex', 'interior']:
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
        else:
            self.sum_ln_k = 0
            self.n_solutions = 0
            all = []
            while self.n_solutions != nsolutions*100:
                self.iteration=0
                self.n_solutions += 1
                while True:
                    self.next_solution()
                    self.curr_sol = self.package_solution()                

                    all.append(self.curr_sol)
                    break

            while nsolutions > 0:
                yield self.CLT(all)
                nsolutions -= 1

    def CLT(self, all):
        N  = min(len(all), 2)
        rs = random_integers(len(all), size=N) - 1

        avg = all[rs[0]].sol.copy()
        for r in rs[1:]:
            avg += all[r].sol
        avg /= N

        return avg

    def next_solution(self):

        while True:

            r = self.start_new_objective(kind=self.objf_choice)

            result = lpsolve('solve', self.lp)
            if   result in [OPTIMAL, TIMEOUT]:   break
            elif result == SUBOPTIMAL: continue
            elif result == INFEASIBLE: raise SamplexNoSolutionError()
            elif result == UNBOUNDED: raise SamplexUnboundedError()
            else:
                Log( result )
                raise SamplexUnexpectedError("unknown pivot result = %i" % result)

            objv = lpsolve('get_objective', self.lp)

            if self.objf_choice == 'facet' and abs(objv) > 1e-6:
                print 'BAD VARIABLE', objv
                del self.ineqs[r]
            else:
                break

        print 'Solution after %i steps.' % lpsolve('get_total_iter', self.lp)


    def package_solution(self):
        objv  = array(lpsolve('get_objective', self.lp))
        vars  = array(lpsolve('get_variables', self.lp)[0])
        slack = array(lpsolve('get_constraints', self.lp)[0]) - array(lpsolve('get_rh', self.lp)[1:])

        slack[abs(slack) < 1e-5] = 0

        nvars = len(vars)
        nslack = len(slack)

        s = SamplexSolution()
        s.sol = empty(nvars + 1)
        s.sol[1:] = vars
        s.sol[0] = objv

        s.vertex = empty(nvars + nslack + 1)
        s.vertex[1:nvars+1] = vars
        s.vertex[1+nvars:1+nvars+nslack] = slack
        s.vertex[0] = objv

        assert all(s.vertex[1:] >= 0), s.vertex[s.vertex < 0]

        return s


    def interior_point(self, r=None):

        #-----------------------------------------------------------------------
        # Set smallest_scale to maximum allowed step. 
        #
        # The current point on the edge of the simplex is in vertex. prev_sol
        # is another point within the simplex.  We wish to start at vertex and
        # find how far we can go in the direction of prev_sol before we violate
        # one of the constraints.  Obviously, we can go at least as far as
        # prev_sol (a step size of 1).  As one moves along this direction the
        # values of the left-hand variables (LHVs) will change.  We are
        # interested in those LHVs that are decreasing but remain greater than
        # zero (as a value less than zero is invalid). The following loop finds
        # the the largest step allowed that doesn't violate one of the
        # constraints.
        #-----------------------------------------------------------------------

        if r is None: r = random()

        sol = self.curr_sol

        k = 0
        smallest_scale = inf

        iv    = sol.vertex[1:]
        dist  = iv - self.prev_sol[1:]
        a = dist > self.SML
        if not any(a): return None

        #print '*', iv
        #print '*', dist[a]

        scale = iv[a] / dist[a]

        smallest_scale = amin(scale)

        assert not isinf(smallest_scale)
        assert smallest_scale > 0.99, smallest_scale

        k = smallest_scale * (1.0-r)
        #k = smallest_scale * pow((1.0-r), 1./3)
        #k = smallest_scale * pow((1.0-r), 1./self.ncols)

        T = sol.vertex[1:] + k * (self.prev_sol[1:]-sol.vertex[1:])
        #t = (T - sol.vertex[1:]) - k * (self.prev_sol[1:]-sol.vertex[1:])

        #T += t
        #assert all(t == 0), t

        self.prev_sol[1:] = T

        #self.sum_ln_k += log(k)
        #assert self.sum_ln_k < 1, self.prev_sol

        assert all(self.prev_sol[1:] >= 0), (self.prev_sol[self.prev_sol < 0], self.prev_sol)
        #assert all(self.prev_sol[1:] >= -self.SML), (self.prev_sol[self.prev_sol < 0], self.prev_sol)

        s = self.prev_sol.copy()[:sol.sol.size]
        return s

    #===========================================================================

    def start_new_objective(self, kind=2, last_r=-1):

        if kind=='random':

            lpsolve('set_obj_fn', self.lp, random(lpsolve('get_Ncolumns', self.lp)) - 0.5)

        elif kind=='facet':

            r = self.ineqs[random_integers(len(self.ineqs))-1]
            lpsolve('set_obj_fn', self.lp, r[0])
            return r

    #===========================================================================

    def noise(self, a):
        if a[0] == 0:
            w = abs(a) > 1e-14
            w[0] = True
            b = a.copy()
            b[w] += 1e-6*(2*random(len(w.nonzero()))-1)
            return b

        return a

    def eq(self, a):
        #if self.add_noise: a = noise(a)
        lpsolve('add_constraint', self.lp, a[1:], EQ, -a[0])
        self.eq_count += 1

    def set_bound(self, a, type):
        return False

        func = {'low': 'set_lowbo',
                'up':  'set_upbo'}.get(type, None)

        if func is None:
            assert 0, 'Bad type to set_bound'

        w = a.nonzero()[0]
        if w.size == 2 and w[0] == 0:
            bnd = -a[0] / a[w[1]]
            lpsolve(func, self.lp, w[1], bnd)
            self.bnd_count += 1
            return True

        return False

    def geq(self, a):
        if not self.set_bound(a, 'low'):
            if self.add_noise: a = self.noise(a)
            lpsolve('add_constraint', self.lp, a[1:], GE, -a[0])
            self.geq_count += 1
            self.ineqs.append([a[1:], GE, -a[0]])

    def leq(self, a):
        #-----------------------------------------------------------------------
        # We convert <= constraints to >= so that in package_solution we can
        # simply subtract the current contraint value in the tableau from the
        # original right hand side values given here to derive the amount of
        # slack on each constraint. This is important to have in
        # interior_point().
        #-----------------------------------------------------------------------
        if not self.set_bound(a, 'up'):

            if self.add_noise: a = self.noise(a)
            lpsolve('add_constraint', self.lp, -a[1:], GE, a[0])
            self.leq_count += 1

            self.ineqs.append([-a[1:], GE, a[0]])
            #self.ineqs.append([a[1:], LE, -a[0]])


