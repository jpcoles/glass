from __future__ import division
import sys
import numpy
import gc
from numpy import isfortran, asfortranarray, sign, logical_and, any, ceil, amax, fmin, fmax
from numpy import set_printoptions
from numpy import insert, zeros, vstack, append, hstack, array, all, sum, prod, ones, delete, log, empty, sqrt, arange, cov, empty_like
from numpy import argwhere, argmin, inf, isinf, amin, abs, where, multiply, eye, mean
from numpy import histogram, logspace, flatnonzero, isinf
from numpy.random import random, normal, random_integers, seed as ran_set_seed
from numpy.linalg import eigh, pinv, eig, norm, inv, det
from numpy import dot
import scipy.linalg.fblas
from scipy.optimize import fmin_bfgs
from itertools import izip, count
import time

from multiprocessing import Process, Queue, Value, Lock
from Queue import Empty as QueueEmpty

#from glrandom import random, ran_set_seed

#dot = lambda a, b: scipy.linalg.fblas.ddot(a, b)
#dot = lambda a, b: scipy.linalg.fblas.dgemm(1., a, b, trans_b=True)

if 0:
    from pylab import figimage, show, imshow, hist, matshow, figure

try:
    from log import log as Log
except ImportError:
    def l(x):
        print x
    Log = l

import csamplex
if 0:
    import glpklpsolve as lpsolve55
    import lpsolve55 as lpsolve55
    from lpsolve55 import lpsolve, EQ,GE,LE
    from lpsolve55 import NORMAL, DETAILED, FULL, IMPORTANT

    from lpsolve55 import NOMEMORY, OPTIMAL, SUBOPTIMAL, INFEASIBLE
    from lpsolve55 import UNBOUNDED, DEGENERATE, NUMFAILURE, USERABORT, TIMEOUT, PRESOLVED
else:
    import glpklpsolve as lpsolve55
    from glpklpsolve import lpsolve, EQ,GE,LE
    from glpklpsolve import NORMAL, DETAILED, FULL, IMPORTANT

    from glpklpsolve import NOMEMORY, OPTIMAL, SUBOPTIMAL, INFEASIBLE
    from glpklpsolve import UNBOUNDED, DEGENERATE, NUMFAILURE, USERABORT, TIMEOUT, PRESOLVED

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

def should_stop(id,stopq):
    try:
        return stopq.get(block=False) == 'STOP'
    except QueueEmpty:
        return False

def rwalk_async(id, nmodels, samplex, store, n_stored, q,stopq, vec,twiddle, window_size, lock, eval,evec):

    vec = vec.copy('A')
    S   = zeros(samplex.eqs.shape[0])
    S0  = zeros(samplex.eqs.shape[0])
    eval = eval.copy('A')
    evec = evec.copy('A')

    accepted = 0
    rejected = 0
    time_begin = 0
    time_end = 0

    #print evec
    #print evec.shape
    #print evec[:,20].shape
    #print dot(evec.T, evec[:,20])
    #assert 0

    #nmodels = int(ceil(nmodels / max(1,samplex.nthreads-1)))

    print ' '*39, 'STARTING rwalk_async THREAD %i [this thread makes %i models]' % (id,nmodels)

    store = store.copy('A')
    eqs  = samplex.eqs.copy('A')

    eqs[:,1:] = dot(samplex.eqs[:,1:], evec)
    #vec[:] = dot(evec.T, vec)

#   ok,v = samplex.in_simplex_matrix(vec, eqs)
#   print ok,v
#   assert ok
#   assert 0



    #eqs[:,1:] = dot(samplex.eqs[:,1:], evec)
    #print eqs.flags
    #assert 0
    I = eye(evec.shape[0]).copy('F')

    csamplex.set_rwalk_seed(1 + id + samplex.random_seed)

    done = should_stop(id,stopq)
    time_begin = time.clock()
    models_since_last_eval = 0
    for i in xrange(window_size + nmodels):

        #if (models_since_last_eval / n_stored) > 0.25:
        if n_stored < window_size and (n_stored % 10) == 0:
        #if (n_stored % int(0.1*window_size+1)) == 0:
            print ' '*39, 'Computing eigenvalues...'
            samplex.compute_eval_evec(store, eval, evec, n_stored, window_size)
            eqs[:,1:] = dot(samplex.eqs[:,1:], evec)
            models_since_last_eval = 0

        models_since_last_eval += 1

        vec[:] = dot(evec.T, vec)
        #print vec

        #ok,v = samplex.in_simplex_matrix(vec, eqs)
        #print ok,v
        #assert ok

        while not done:
            accepted = 0
            rejected = 0
            #print 'Walking...'
            #print 'twiddle is', twiddle

            if should_stop(id,stopq):
                done = True
                break


            accepted,rejected = csamplex.rwalk(samplex, eqs, vec,eval,I,S,S0, twiddle, accepted,rejected)

            r = accepted / (accepted + rejected)
            #lock.acquire()
            print ' '*39, 'THREAD %i %.3f Acceptance rate  (%i Accepted  %i Rejected  %e twiddle)' % (id, r, accepted, rejected, twiddle)
            #lock.release()

            #-------------------------------------------------------------------
            # If the actual acceptance rate was OK then leave this loop,
            # otherwise change our step size twiddle factor to improve the rate.
            # Even if the accepance rate was OK, we adjust the twiddle but only
            # with a certain probability. This drives the acceptance rate to 
            # the specified one even if we are within the tolerance but doesn't
            # throw away the results if we are not so close. This allows for
            # a larger tolerance.
            #-------------------------------------------------------------------
            if abs(r - samplex.accept_rate) < samplex.accept_rate_tol:
                if random() < abs(r - samplex.accept_rate)/samplex.accept_rate_tol:
                    twiddle *= 1 + ((r-samplex.accept_rate) / samplex.accept_rate / 2)
                    twiddle = max(1e-14,twiddle)
                break

            twiddle *= 1 + ((r-samplex.accept_rate) / samplex.accept_rate / 2)
            twiddle = max(1e-14,twiddle)
            #lock.acquire()
            print ' '*39, 'RESTARTING r =',r, twiddle
            #lock.release()

        if done:
            break

        vec[:] = dot(evec, vec)
        assert numpy.all(vec >= 0), vec[vec < 0]
        if numpy.any(vec < 0): sys.exit(0)

        samplex.project(vec)

        store[:,n_stored] = vec
        n_stored += 1
        if n_stored >= window_size:
            q.put([id,vec.copy('A')])


    time_end = time.clock()
    q.put(['TIME', time_end-time_begin])
    print ' '*39, 'RWALK THREAD %i LEAVING  n_stored=%i  time=%.4fs' % (id,i,time_end-time_begin)

class Samplex:
    INFEASIBLE, FEASIBLE, NOPIVOT, FOUND_PIVOT, UNBOUNDED = range(5)
    SML = 1e-5
    EPS = 1e-14

    def __init__(self, **kw):

        ncols                = kw.get('ncols', None)
        nthreads             = kw.get('nthreads', 1)
        rngseed              = kw.get('rngseed',  0)
        self.with_noise      = kw.get('add noise', 1e-6)
        self.stride          = kw.get('stride', 1)
        self.accept_rate     = kw.get('acceptance rate', 0.25)
        self.accept_rate_tol = kw.get('acceptance tol', 0.05)
        self.redo_factor     = kw.get('redo factor', 1)
        self.twiddle         = kw.get('twiddle', 2.4)

        Log( "Samplex created" )
        Log( "    ncols = %s" % ncols )
        if ncols is not None:
            self.nVars = ncols

        self.random_seed = rngseed

        self.nthreads = nthreads

        self.n_equations = 0
        self.eq_count = 0
        self.leq_count = 0
        self.geq_count = 0

        self.eq_list_no_noise = []

        self.ineqs = []
        self.lp = lpsolve('make_lp', 0, ncols+1)
        lpsolve('set_epsb', self.lp, 1e-14)
        lpsolve('set_epsd', self.lp, 1e-14)
        lpsolve('set_epsint', self.lp, 1e-14)
        lpsolve('set_epsel', self.lp, 1e-8)
        lpsolve('set_verbose', self.lp, IMPORTANT)

        self.iteration = 0
        self.moca = None
        self.curr_sol = None

        self.forbidden_variables = []

    def start(self):
        #print "%6s %6s %6s\n%6i %6i %6i" \
        #    % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count)

        Log( '=' * 80 )
        Log( 'Simplex Random Walk' )
        Log( '=' * 80 )

        Log( 'Using lpsolve %s' % lpsolve('lp_solve_version') )
        Log( "random seed = %s" % self.random_seed )
        Log( "threads = %s" % self.nthreads )
        Log( "with noise = %s" % self.with_noise )
        Log( "stride = %s" % self.stride )
        Log( "acceptence rate = %s" % self.accept_rate )
        Log( "acceptence rate tolerance = %s" % self.accept_rate_tol )
        Log( "redo factor = %s" % self.redo_factor )
        Log( "starting twiddle = %s" % self.twiddle )

#       self.geq_count = 0
#       self.leq_count = 0
#       self.eq_count  = 0

#       def eq_key(x):
#           if x[0] == self._geq: return 2
#           if x[0] == self._leq: return 1
#           if x[0] == self._eq:  return 0
#           assert False, 'Bad function %s' % str(x[0])

        def eq_key(x):
            if x[0] == 'geq': return 2
            if x[0] == 'leq': return 1
            if x[0] == 'eq':  return 0
            assert False, 'Bad function %s' % str(x[0])

        self.eq_list_no_noise.sort(key=eq_key)
        #print "random seed =", ran_set_seed(0)

        if 0:
            def print_array(out, fs, arr):
                out.write("%s %i " % (fs, len(arr)));
                for i in arr: 
                    if i in [0,1,-1]:
                        out.write("%10i " % i)
                    else:
                        out.write("%.4e " % i)
                out.write("\n")
            Log( 'Writing out equations...' )
            out = open('eqs-new', 'w')
            for f,a in self.eq_list_no_noise:
                if f == self._eq:  fs = 'eq'
                if f == self._geq: fs = 'geq'
                if f == self._leq: fs = 'leq'
                print_array(out, fs, a)
            out.close()
            Log( 'done.' )


        Log( "    %i equations" % len(self.eq_list_no_noise) )

        Log( "%6s %6s %6s\n%6i %6i %6i" 
            % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count) )

        self.eqn_count = self.eq_count + self.geq_count + self.leq_count

        if 0:
            import numpy as np
            import pylab as pl
            m = np.empty((len(self.eq_list_no_noise), len(self.eq_list_no_noise[0][1])))
            print m.shape
            for i,e in enumerate(self.eq_list_no_noise):
                f,a = e
                m[i] = a
                if f == self._eq:  m[i][m[i] != 0] = 1
                if f == self._geq: m[i][m[i] != 0] = 2
                if f == self._leq: m[i][m[i] != 0] = 3
            #m[m != 0] = 1
            pl.matshow(m)
            pl.show()

    def in_simplex_matrix(self, np, eqs=None, tol=0, eq_tol=1e-8, verbose=0):
        if eqs is None:
            eqs = self.eqs

        ok = 0

        eq_offs = 0;
        leq_offs = eq_offs + self.eq_count;
        geq_offs = leq_offs + self.leq_count;

        a_min = inf
        for i,e in enumerate(eqs):
            a0 = dot(np, eqs[i,1:])
            a  = eqs[i,0] + a0

            #print np.flags, e[1:].flags
            #assert 0
            if i >= geq_offs:
                a_min = min(a_min, a)
                if a < -tol: 
                    if verbose: print 'F>', i,a
                    ok += 1
            elif i >= leq_offs:
                a_min = min(a_min, a)
                if a > tol: 
                    if verbose: print 'F<', i,a
                    ok += 1
            else:
                if abs(a) > eq_tol: 
                    if verbose: print 'F=', i,a, (1 - abs(e[0]/a0))
                    ok += 1

            #if verbose > 1: print "TT", c, a
              
        if verbose > 1:
            print 'Smallest a was %e' % (a_min,)

        #print 'T '
        return ok==0, ok

    def in_simplex(self, np, tol=0, eq_tol=1e-8, verbose=0):
        bad = []

        a_min = inf
        for i,[c,e] in enumerate(self.eq_list_no_noise):
            a0 = dot(np, e[1:])
            a  = e[0] + a0

            #print np.flags, e[1:].flags
            #assert 0
            if c == 'geq':
                a_min = min(a_min, a)
                if a < -tol: 
                    if verbose: print 'F>', i,a
                    bad.append([i,a])
            elif c == 'leq':
                a_min = min(a_min, a)
                if a > tol: 
                    if verbose: print 'F<', i,a
                    bad.append([i,a])
            elif c == 'eq':
                if abs(a) > eq_tol: 
                    if verbose: print 'F=', i,a, (1 - abs(e[0]/a0))
                    bad.append([i,a])

            #if verbose > 1: print "TT", c, a
              
        if verbose > 1:
            print 'Smallest a was %e' % (a_min,)

        #print 'T '
        return not bad, bad

    def distance_to_plane(self,pt,dir):
        ''' dir should be a normalized vector '''

        dist = inf

        p = self.dist_eqs
        a = dot(dir, p[:,1:].T)
        w = a > 0

        if w.any():
            dtmp = -(p[:,0] + dot(pt, p[:,1:].T)) / a
            dist = amin(dtmp[w])

        #print '!!!dist', dist

        #print pt

        # check implicit >= 0 contraints
        a = -dir 
        w = a > 0
        if w.any():
            dtmp = pt[w] / a[w]
            dist = amin([dist, amin(dtmp)])
            #print '???dist', dist

        assert dist != inf

        return dist

    def distance_to_planeX(self,pt,dir):
        ''' dir should be a normalized vector '''

        dist = inf
        for c,e in self.eq_list_no_noise:
            if c == 'eq':
                continue
            elif c == 'leq':
                p = e
            elif c == 'geq':
                p = -e

            a = dot(dir, p[1:])
            if a > 0:
                dtmp = -(p[0] + dot(pt, p[1:])) / a
                if dtmp > 0:
                    dist = min(dist, dtmp)
                elif dtmp > -1e-12:
                    pass
                else:
                    assert dtmp >= 0, dtmp

        # check implicit >= 0 contraints
        for i in xrange(pt.size):
            a = dir[i] * -1
            if a > 0:
                dtmp = -(0 + pt[i] * -1) / a
                if dtmp >= 0:
                    assert dtmp >= 0, dtmp
                    if dtmp == 0:
                        print 'ZERO dist was', dist
                    dist = min(dist, dtmp)


        assert dist != inf
        #if dist == inf:
        #    return 0

        return dist

    def compute_midpoint(self):
        s = min(max(0,n_stored - window_size), window_size)

        ev,evec = eigh(cov(store[:, s:n_stored]))
        avg = store[:, s:n_stored].mean(axis=1)
        nzero = 0
        for r in range(dim):
            if ev[r] < 1e-12:
                eval[r] = 0
                nzero += 1
            else:
                direction = evec[:,r]
                tmax1 = -distance_to_plane(avg, -direction)
                tmax2 = +distance_to_plane(avg, +direction)
                avg += direction * ((tmax2 + tmax1) / 2)
        
        if nzero != self.eq_count:
            print '-'*80
            Log( 'WARNING:', 'Expected number of zero length eigenvectors (%i) to equal number of equality constraints (%i)' % (nzero, self.eq_count) )
            print '-'*80

        return avg

    def compute_eval_evec(self, store, eval, evec, n_stored, window_size):
        s = min(max(0,n_stored - window_size), window_size)

        eval0,evec0 = eigh(cov(store[:,  :n_stored]))
        avg = store[:,  :n_stored].mean(axis=1)
        nzero = 0
        for r in range(eval.shape[0]):
            if eval0[r] < 1e-12:
                eval[r] = 0
                nzero += 1
            else:
                direction = evec0[:,r]
                tmax1 = -self.distance_to_plane(avg, -direction)
                tmax2 = +self.distance_to_plane(avg, +direction)
                #print 'tmax', tmax1, tmax2
                eval[r] = (tmax2 - tmax1) / sqrt(12)

        evec[:] = evec0
        print 'eval(inside)', eval
        if nzero != self.eq_count:
            print '-'*80
            Log( 'WARNING:', 'Expected number of zero length eigenvectors (%i) to equal number of equality constraints (%i)' % (nzero, self.eq_count) )
            print '-'*80

    def random_direction(self,np):
        return dot(normal(0, 2.4/dof, dim), np.T)


    def project(self,x):
        if self.Apinv is not None:
            q = dot(self.A, x)
            q += self.b
            q = dot(self.Apinv, q)
            x -= q

    def project_evec(self,ev):
        for e in ev:
            e -= dot(self.Apinv, dot(A, e))

    def inner_point0(self, np):

        #lpsolve('set_presolve', self.lp, 1 + 4)
        #lpsolve('set_pivoting', self.lp, 3+128+32+512)
        #print 'BEFORE', lpsolve('get_Nrows', self.lp)
        #o = 20*random(lpsolve('get_Ncolumns', self.lp)) - 10

        #lpsolve('set_obj_fn', self.lp, o.tolist())
        #print lpsolve('get_Ncolumns', self.lp)
        #assert 0
        #o = 2000*random(lpsolve('get_Ncolumns', self.lp)) - 1000
        #lpsolve('set_sense', self.lp, False)
        #lpsolve('set_obj_fn', self.lp, o.tolist())
        self.next_solution()
        v1 = self.package_solution()

        #ok,fail_count = self.in_simplex(v1.vertex[1:self.nVars+1], eq_tol=1e-12, tol=0, verbose=2)
        #ok,fail_count = self.in_simplex(v1.vertex[1:self.nVars+1], eq_tol=1e-12, tol=-1e-13, verbose=2)
        #assert ok, len(fail_count)
#       return v1.vertex[1:self.nVars+1]
        #assert 0

        np = zeros_like(np)
        np2 = None
        i=0
        while True:
            i += 1

            Log('Inner point step %i' % i)

            o = 2000*random(lpsolve('get_Ncolumns', self.lp)) - 1000
            lpsolve('set_sense', self.lp, False)

            lpsolve('set_obj_fn', self.lp, o.tolist())
            self.next_solution()

            v1 = self.package_solution()

            np += v1.vertex[1:self.nVars+1]
            np2 = np.copy()
            np2 /= i
            self.project(np2)
            ok,failed = self.in_simplex(np2, eq_tol=1e-12, tol=-1e-13, verbose=0)

            if ok: break

            Log('%i equations to satisfy' % len(failed))

        return np2

    def inner_point(self, np):
        ncols = lpsolve('get_Ncolumns', self.lp)
        o = zeros(ncols)
        o[-1] = 1

        assert ncols == self.nVars+1
        for i in range(self.nVars):
            q = zeros(ncols)
            q[i] = -1
            q[-1] = 1
            lpsolve('add_constraint', self.lp, q.tolist(), LE, 0)

        lpsolve('set_sense', self.lp, False)
        lpsolve('set_obj_fn', self.lp, o.tolist())
        lpsolve('set_verbose', self.lp, FULL)
        Log('Finding first inner point')
        self.next_solution()
        lpsolve('set_verbose', self.lp, IMPORTANT)

        v1 = self.package_solution()

        ok,fail_count = self.in_simplex(v1.vertex[1:self.nVars+1], eq_tol=1e-12, tol=0, verbose=1)
        ok,fail_count = self.in_simplex(v1.vertex[1:self.nVars+1], eq_tol=1e-12, tol=-1e-13, verbose=1)
        assert ok, len(fail_count)
        np = v1.vertex[1:self.nVars+1]
        self.project(np)
        ok,fail_count = self.in_simplex(np, eq_tol=1e-12, tol=0, verbose=1)
        ok,fail_count = self.in_simplex(np, eq_tol=1e-12, tol=-1e-5, verbose=1)

        #print np
        #print len(np)
        #assert 0
        #return np

        np = np.copy()
        np2 = None
        i=1
        while i < 20:
            i += 1

            Log('Refining inner point, step %i/20' % i)

            o = 2*random(lpsolve('get_Ncolumns', self.lp)) - 1
            lpsolve('set_sense', self.lp, False)

            lpsolve('set_obj_fn', self.lp, o.tolist())
            self.next_solution()

            v1 = self.package_solution()

            np += v1.vertex[1:self.nVars+1]
            np2 = np.copy()
            np2 /= i
            self.project(np2)
            ok,failed = self.in_simplex(np2, eq_tol=1e-12, tol=-1e-13, verbose=0)

        return np2

    def estimated_middle(self, np, eval, evec, iter=4, eval_tol=1e-12):
        np = np.copy()
        dirs = evec[:,eval >= eval_tol]

        for i in range(iter):
            for direction in dirs.T:
                tmax1 = -self.distance_to_plane(np, -direction)
                tmax2 = +self.distance_to_plane(np, +direction)
                assert tmax1 < tmax2, 'tmax %e %e' % (tmax1, tmax2)
                np += direction * ((tmax1+tmax2) / 2)

        return np

        for i in range(iter):
            for r in range(len(eval)):
                if eval[r] >= eval_tol:
                    direction = evec[:,r]
                    tmax1 = -self.distance_to_plane(np, -direction)
                    tmax2 = +self.distance_to_plane(np, +direction)
                    assert tmax1 < tmax2, 'tmax %e %e  ev[%i] %e' % (tmax1, tmax2, r, eval[r])
                    np += direction * ((tmax1+tmax2) / 2)
        return np

    def initial_store(self, np, ev, eval, evec, store, eval_tol=1e-12):
        nzero = 0
        n_stored = 0
        for r in range(eval.size):
            if ev[r] < eval_tol:
                eval[r] = 0
                nzero += 1
            else:
                direction = evec[:,r]
                tmax1 = -self.distance_to_plane(np, -direction)
                tmax2 = +self.distance_to_plane(np, +direction)
                eval[r] = (tmax2 - tmax1) / sqrt(12)
                #print 'tmax', tmax1, tmax2, eval[r]
                assert tmax1 < tmax2, 'tmax %i %i  ev[%i] %e' % (tmax1, tmax2, r, ev[r])
                store[:,n_stored+0] = np + direction * tmax1
                store[:,n_stored+1] = np + direction * tmax2
                n_stored += 2

        assert nzero == self.eq_count, "Number of zero length eigenvectors doesn't equal number of equalities. (%i != %i)" % (nzero, self.eq_count)

        return n_stored

    def next(self, nsolutions=None):


        time_begin_next = time.clock()

        if nsolutions == 0: return

        assert nsolutions is not None

        dim = self.nVars
        dof = dim - self.eq_count
        #window_size = max(10, int(0.1 * dim))

        window_size = 2*dim #max(10, int(1.5 * dim))

        redo = max(100, int((dim ** 2) * self.redo_factor))
        nmodels = nsolutions
        nthreads = self.nthreads

        self.stride = int(dim+1)

        n_stored = 0
        self.dim = dim
        self.dof = dof
        self.redo = redo

        store = zeros((dim, window_size+nmodels+1), order='Fortran', dtype=numpy.float64)
        eval  = zeros(dim, order='C', dtype=numpy.float64)
        evec  = zeros((dim,dim), order='F', dtype=numpy.float64)
        np    = zeros(dim, order='C', dtype=numpy.float64)

        self.eqs = zeros((self.eqn_count+dim,dim+1), order='C', dtype=numpy.float64)
        for i,[c,e] in enumerate(self.eq_list_no_noise):
            self.eqs[i,:] = e
        for i in xrange(dim):
            self.eqs[self.eqn_count+i,1+i] = 1

        self.dist_eqs = zeros((self.eqn_count-self.eq_count,dim+1), order='C', dtype=numpy.float64)
        i=0
        for c,e in self.eq_list_no_noise:
            if c == 'eq':
                continue
            elif c == 'leq':
                p = e
            elif c == 'geq':
                p = -e
            self.dist_eqs[i,:] = p
            i += 1


        #-----------------------------------------------------------------------
        # Create pseudo inverse matrix to reproject samples back into the
        # solution space.
        #-----------------------------------------------------------------------
        P = numpy.eye(dim) 
        if self.eq_count > 0:
            self.A = zeros((self.eq_count, dim), order='C', dtype=numpy.float64)
            self.b = zeros(self.eq_count, order='C', dtype=numpy.float64)
            for i,[c,e] in enumerate(self.eq_list_no_noise[:self.eq_count]):
                self.A[i] = e[1:]
                self.b[i] = e[0]
            self.Apinv = pinv(self.A)
            P -= dot(self.Apinv, self.A)
        else:
            self.A = None
            self.B = None
            self.Apinv = None

        ev, evec = eigh(P)
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Find a point that is completely inside the simplex
        #-----------------------------------------------------------------------
        time_begin_inner_point = time.clock()
        np = self.inner_point(np)
        time_end_inner_point = time.clock()
        ok,fail_count = self.in_simplex(np, eq_tol=1e-12, tol=0, verbose=1)
        assert ok

        #-----------------------------------------------------------------------
        # Estimate a point in the middle of the simplex
        #-----------------------------------------------------------------------
        print 'Estimating middle point'
        time_begin_middle_point = time.clock()
        np = self.estimated_middle(np, ev, evec)
        time_end_middle_point = time.clock()
#       ok,fail_count = self.in_simplex(np)
#       assert ok

        #-----------------------------------------------------------------------
        # Estimate the eigenvectors of the simplex
        #-----------------------------------------------------------------------
        print 'Estimating eigenvectors'
        time_begin_est_eigenvectors = time.clock()
        n_stored = self.initial_store(np, ev, eval, evec, store)
        store[:,n_stored] = np
        n_stored += 1
        self.compute_eval_evec(store, eval, evec, n_stored, window_size)
        time_end_est_eigenvectors = time.clock()

        #-----------------------------------------------------------------------
        # Now we can start the random walk
        #-----------------------------------------------------------------------

        #assert self.in_simplex(np, tol=1e-10, eq_tol=1e-12)

        Log( "Getting solutions" )
        #print 'window_size', window_size
        print 'redo', redo

        accept_rate     = self.accept_rate
        accept_rate_tol = self.accept_rate_tol

        q     = Queue()
        stopq = Queue()
        lock  = Lock()

        threads = []
        models_per_thread = ceil(nmodels / nthreads)
        models_over       = nthreads*models_per_thread - nmodels
        N = 0
        for i in range(nthreads):
            n = int(max(0, min(models_per_thread, nmodels - N)))
            if i < models_over:
                n -= 1
            assert n >= 0
            if n == 0:
                Log('Skipping thread %i. Not enough work.' % i)
                continue
            thr = Process(target=rwalk_async, args=(i, n, self, store,n_stored, q,stopq, np,self.twiddle, window_size, lock, eval.copy('A'), evec.copy('A')))
            threads.append(thr)

            N += n

        assert N == nmodels

        for thr in threads:
            thr.start()

        time_begin_get_models = time.clock()
        time_threads = []
        i=0
        while i < nmodels:
#           if q.qsize() + i >= nmodels:
#               for j,thr in enumerate(threads):
#                   stopq.put('STOP')

            k,vec = q.get()
            if k == 'TIME':
                time_threads.append(vec)
                continue

            assert numpy.all(vec >= 0), vec[vec < 0]
            print '%i models left to generate' % (nmodels-i-1)

            t = zeros(dim+1, order='Fortran', dtype=numpy.float64)
            i += 1
            t[1:] = vec
            yield t
        time_end_get_models = time.clock()

        for i,thr in enumerate(threads):
            stopq.put('STOP')

        for thr in threads:
            thr.terminate()

        # 1. Select initial set of random points
        # 2. Calculate the mean and eigenvectors of said points
        # 3. Find the distance of each eigenvector to the simplex edge
        # 4. Take random step
        # 5. Compute if in simplex
        # 6. Accept original or new point
        # 7. Recompute eigenvectors every so often

        del self.lp

        time_end_next = time.clock()
        max_time_threads = amax(time_threads) if time_threads else 0
        avg_time_threads = mean(time_threads) if time_threads else 0
        time_end_next += max_time_threads
        print '-'*80
        print 'SAMPLEX TIMINGS'
        print '-'*80
        print 'Initial inner point    %.2fs' % (time_end_inner_point - time_begin_inner_point)
        print 'Estimate middle point  %.2fs' % (time_end_middle_point - time_begin_middle_point)
        print 'Estimate eigenvectors  %.2fs' % (time_end_est_eigenvectors - time_begin_est_eigenvectors)
        print 'Modeling               %.2fs' % (time_end_get_models - time_begin_get_models)
        print 'Max/Avg thread time    %.2fs %.2fs' % (max_time_threads, avg_time_threads)
        print 'Total time             %.2fs' % (time_end_next - time_begin_next);

    def next_solution(self, lp=None):
        if lp is None: lp = self.lp

        while True:

            #r = self.start_new_objective()

            result = lpsolve('solve', lp)
            if   result in [OPTIMAL, TIMEOUT]:   break
            elif result == SUBOPTIMAL: continue
            elif result == INFEASIBLE: raise SamplexNoSolutionError()
            elif result == UNBOUNDED: raise SamplexUnboundedError()
            else:
                Log( result )
                raise SamplexUnexpectedError("unknown pivot result = %i" % result)

        #print 'Solution after %i steps.' % lpsolve('get_total_iter', lp)
        return False

    def package_solution(self, lp=None):
        if lp is None: lp = self.lp
        objv  = array(lpsolve('get_objective', lp))
        vars  = array(lpsolve('get_variables', lp)[0])
        slack = [] #array(lpsolve('get_constraints', lp)[0]) - array(lpsolve('get_rh', lp)[1:])

        assert len(vars) == lpsolve('get_Norig_columns', lp)
        vars = vars[:-1]

        #slack[abs(slack) < 1e-5] = 0

        nvars = len(vars)
        nslack = len(slack)

        s = SamplexSolution()
        s.sol = empty(nvars + 1)
        s.sol[1:] = vars
        s.sol[0] = objv

        s.vertex = empty(nvars + nslack + 1)
        s.vertex[1:nvars+1] = vars
        #s.vertex[1+nvars:1+nvars+nslack] = slack
        s.vertex[0] = objv

        s.vertex[abs(s.vertex) < 1e-14] = 0

        assert all(s.vertex[1:] >= 0), s.vertex[s.vertex < 0]

        return s

    def start_new_objective(self):
        lpsolve('set_obj_fn', self.lp, (random(lpsolve('get_Ncolumns', self.lp)) - 0.5).tolist())


    #=========================================================================

    TT = 1e-14
    def eq(self, a):
        #print 'eq range', amin(a), amax(a)
        a[abs(a)<self.TT] = 0
        #assert abs(a[0])
        l = (a[1:]).tolist()
        l.append(0)
        lpsolve('add_constraint', self.lp, l, EQ, -a[0])
        self.eq_count += 1
        self.eq_list_no_noise.append(['eq', a])

    def geq(self, a):
        return self.leq(-a)

        #print 'geq range', amin(a), amax(a)
        #a[abs(a)<self.TT] = 0
        #assert (a[a>0] > TT).all(), a[a>0]
        l = (a[1:]).tolist()
        l.append(1)
        lpsolve('add_constraint', self.lp, l, GE, -a[0])
        self.geq_count += 1
        self.ineqs.append([a[1:], GE, -a[0]])
        self.eq_list_no_noise.append(['geq', a])

        #if self.geq_count == 73:
            #assert 0

    def leq(self, a):
        #-----------------------------------------------------------------------
        # We convert <= constraints to >= so that in package_solution we can
        # simply subtract the current contraint value in the tableau from the
        # original right hand side values given here to derive the amount of
        # slack on each constraint. This is important to have in
        # package_soltion().
        #-----------------------------------------------------------------------
        #a[abs(a)<self.TT] = 0
        #print 'leq range', amin(a), amax(a)
        #assert (a[a>0] > TT).all()
        l = (a[1:]).tolist()
        l.append(1)
        lpsolve('add_constraint', self.lp, l, LE, -a[0])
        #lpsolve('add_constraint', self.lp, (-a[1:]).tolist(), GE, a[0])
        self.leq_count += 1

        self.ineqs.append([-a[1:], GE, a[0]])
        self.eq_list_no_noise.append(['leq', a])


        #if self.leq_count == 17:
            #assert 0
