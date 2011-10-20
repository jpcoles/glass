from __future__ import division
import sys
import numpy
import gc
from numpy import isfortran, asfortranarray, sign, logical_and, any
from numpy import set_printoptions
from numpy import insert, zeros, vstack, append, hstack, array, all, sum, ones, delete, log, empty, sqrt, arange, cov
from numpy import argwhere, argmin, inf, isinf, amin, abs, where, multiply
from numpy import histogram, logspace, flatnonzero, isinf
from numpy.random import random, normal, random_integers, seed as ran_set_seed
from numpy.linalg import eigh, pinv
from numpy import dot
import scipy.linalg.fblas
from itertools import izip
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
    INFEASIBLE, FEASIBLE, NOPIVOT, FOUND_PIVOT, UNBOUNDED = range(5)
    SML = 1e-5
    EPS = 1e-14

    def __init__(self, **kw):

        ncols    = kw.get('ncols', None)
        nthreads = kw.get('nthreads', 1)
        rngseed  = kw.get('rngseed',  0)
        self.sol_type  = kw.get('solution type', 'interior')
        self.with_noise   = kw.get('add noise', 1e-6)
        self.stride = kw.get('stride', 1)

        Log( "Samplex created" )
        Log( "    ncols = %s" % ncols )
        if ncols is not None:
            self.nVars = ncols
            self.nRight = self.nVars

        csamplex.set_rnd_cseed(rngseed)

        self.random_seed = rngseed

        self.nthreads = nthreads
        Samplex.pivot = lambda s: csamplex.pivot(s)
        Samplex.rwalk = lambda s: csamplex.rwalk(s)

        self.data = None
        self.dcopy = []

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
        self.eq_list_no_noise = []

        self.iteration = 0
        self.moca = None
        self.sum_ln_k = 0
        self.curr_sol = None
        self.n_solutions = 0

        self.forbidden_variables = []

    def check_data_shape(self, len=None):

        if len is None: len = self.nLeft

        if self.data.shape[0] == len: 
            self.data = insert(self.data, self.data.shape[0], 0, axis=0)

    def start(self):
        #print "%6s %6s %6s\n%6i %6i %6i" \
        #    % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count)

        Log( '=' * 80 )
        Log( 'Simplex Random Walk' )
        Log( '=' * 80 )

        Log( "random seed = %s" % self.random_seed )
        Log( "threads = %s" % self.nthreads )
        Log( "solution type = %s" % self.sol_type )
        Log( "with noise = %s" % self.with_noise )
        Log( "stride = %s" % self.stride )

        Log( "N = %i" % self.nVars )
        Log( "L = %i" % self.nLeft )
        Log( "R = %i" % self.nRight )
        Log( "S = %i" % self.nSlack )
        self.data = zeros((self.nLeft+1, self.nRight+1), order='Fortran', dtype=numpy.float64)

        self.nLeft = 0
        self.nSlack = 0
        self.nTemp = 0
        self.nRight = self.nVars

        # Tag the first element because we shouldn't be using it.
        #self.lhv = [numpy.nan]  # numpy on MacOSX doesn't like this
        self.lhv = [999999]
        self.rhv = range(self.nVars+1)

        self.geq_count = 0
        self.leq_count = 0
        self.eq_count  = 0

        def eq_key(x):
            if x[0] == self._geq: return 2
            if x[0] == self._leq: return 1
            if x[0] == self._eq:  return 0
            assert False, 'Bad function %s' % str(x[0])

        self.eq_list.sort(key=eq_key)
        self.eq_list_no_noise.sort(key=eq_key)
        #print "random seed =", ran_set_seed(0)
        #print self.data

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
            for f,a in self.eq_list:
                if f == self._eq:  fs = 'eq'
                if f == self._geq: fs = 'geq'
                if f == self._leq: fs = 'leq'
                print_array(out, fs, a)
            out.close()
            Log( 'done.' )


        Log( "Building matrix" )
        for i,[f,a] in enumerate(self.eq_list):
            f(a)
            if i%500 == 0:
                Log( "%i/%i" % (i,len(self.eq_list)) )

        #print self.data

        Log( "    %i equations" % len(self.eq_list) )
        Log( "    N = %i" % self.nVars )
        Log( "    L = %i" % self.nLeft )
        Log( "    R = %i" % self.nRight )
        Log( "    S = %i" % self.nSlack )

        Log( "%6s %6s %6s\n%6i %6i %6i" 
            % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count) )

        if 0:
            import numpy as np
            import pylab as pl
            m = np.empty((len(self.eq_list), len(self.eq_list[0][1])))
            print m.shape
            for i,e in enumerate(self.eq_list):
                f,a = e
                m[i] = a
                if f == self._eq:  m[i][m[i] != 0] = 1
                if f == self._geq: m[i][m[i] != 0] = 2
                if f == self._leq: m[i][m[i] != 0] = 3
            #m[m != 0] = 1
            pl.matshow(m)
            pl.show()

        # We delete this now so that pickeling doesn't fail later when the
        # state is saved.
        #del self.eq_list

        f2n = {self._eq: 'eq',
               self._leq: 'leq',
               self._geq: 'geq'}
        for i in range(len(self.eq_list)):
            self.eq_list[i][0] = f2n[self.eq_list[i][0]]
            self.eq_list_no_noise[i][0] = f2n[self.eq_list_no_noise[i][0]]


        #print self.lhv
        self.lhv = array(self.lhv, dtype=numpy.int32)
        self.rhv = array(self.rhv, dtype=numpy.int32)

        #print 'samplex sum', sum(self.data)

#       for x in self.lhv: print x
#       print "----------------------------------------------------"
#       for x in self.rhv: print x

#       for i in xrange(1,self.data.shape[0]):
#           for j in xrange(self.data.shape[1]):
#               print self.data[i,j]

#       for i in xrange(self.nLeft+1):
#           for j in xrange(self.nRight+1):
#               print "P %i %i %.8f" % (j,i, self.data[i][j])

#       d = self.data.copy('F')
#       d[d < 1e-6] = 0
#       d[d > 1e+1] = 0
#       #d[d != 0.] = 2
#       print "dddddddddd"
#       print d.shape, self.data.shape, self.nVars, self.nLeft
#       d[:,self.nVars] = 3
#       matshow(d.T)
#       figure()
#       d = log(d)
#       d[isinf(d)] = 0
#       hist(d[d!=0].flatten(), bins=100)
#       #imshow(d.T)
#       show()

    def status(self):
        if self.iteration & 15 == 0:
            Log( "model %i]  iter % 5i  obj-val %.8g" % (self.n_solutions, self.iteration, self.data[0,0]) )

    def next(self, nsolutions=None):


        def distance_to_plane(pt,dir):
            dist = 1e30
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
                    dist = min(dist, dtmp)

            return dist

        def compute_eval():
            ev,evec = eigh(cov(store[:, :n_stored]))
            #print evec
            #print '-' * 80
            #print ev.T
            #print '-' * 80
            #project_evec(evec)
            #print evec
            avg = store[:, :n_stored].mean(axis=1)
            for r in range(dim):
                if ev[r] < 1e-15:
                    eval[r] = 0
                else:
                    direction = evec[:,r]
                    tmax1 = -distance_to_plane(avg, -direction)
                    tmax2 = +distance_to_plane(avg, +direction)
                    eval[r] = (tmax2 - tmax1) / sqrt(12)

            #print eval
            multiply(eval, evec, est_evec)
            #print est_evec
            #assert 0

        def random_direction(np):
            return dot(normal(0, 2.4/dof, dim), np.T)

        def in_simplex(np, tol=0, eq_tol=1e-8, verbose=False):
            for i,[c,e] in enumerate(self.eq_list_no_noise):
                a = e[0] + dot(np, e[1:])

                #print np.flags, e[1:].flags
                #assert 0
                if c == 'geq':
                    if a < -tol: 
                        #print 'F>', i,a
                        return False
                elif c == 'leq':
                    if a > tol: 
                        #print 'F<', i,a
                        return False
                elif c == 'eq':
                    if abs(a) > eq_tol: 
                        #print 'F=', i,a
                        return False

                if verbose: print "TT", c, a
                  
            #print 'T '
            return True

        def project(x):
            q = dot(A, x)
            q += b
            q = dot(Apinv, q)
            x -= q

        def project_evec(ev):
            for e in ev:
                e -= dot(Apinv, dot(A, e))

        self.dcopy = [self.data.copy('F'),
                      self.lhv.copy(),
                      self.rhv.copy(),
                      self.nVars,
                      self.nLeft,
                      self.nSlack,
                      self.nTemp,
                      self.nRight]

        dim = self.nVars
        dof = dim - self.eq_count
        window_size = int(1.5 * dim)
        redo = dim ** 2
        nmodels = 10

        store = zeros((dim, window_size+nmodels), order='Fortran', dtype=numpy.float64)
        eval  = zeros(dim, order='C', dtype=numpy.float64)
        np    = zeros(dim, order='C', dtype=numpy.float64)
        est_evec = zeros((dim,dim), order='F', dtype=numpy.float64)

        self.eqs = zeros((self.eq_count + self.geq_count + self.leq_count,dim+1), order='C', dtype=numpy.float64)
        for i,[c,e] in enumerate(self.eq_list_no_noise):
            self.eqs[i,:] = e

        print self.eqs[0,:]
        print self.eqs[1,:]

        n_stored = 0

        self.store = store
        self.eval = eval
        self.np = np
        self.est_evec = est_evec
        self.dim = dim
        self.dof = dof
        self.redo = redo

        #-----------------------------------------------------------------------
        # Create pseudo inverse matrix to reproject samples back into the
        # solution space.
        #-----------------------------------------------------------------------
        A = zeros((self.eq_count, dim), order='C', dtype=numpy.float64)
        b = zeros(self.eq_count, order='C', dtype=numpy.float64)
        for i,[c,e] in enumerate(self.eq_list_no_noise[:self.eq_count]):
            A[i] = e[1:]
            b[i] = e[0]
        Apinv = pinv(A)
        #-----------------------------------------------------------------------


        Log( "Getting solutions" )
        if not self.find_feasible(): return

        Log( "------------------------------------" )
        Log( "Found feasible" )
        Log( "------------------------------------" )


        self.curr_sol = self.package_solution()                
        self.moca     = self.curr_sol.vertex.copy()

        #-----------------------------------------------------------------------
        # First we need to find a small sample within the solution space
        #-----------------------------------------------------------------------

        print "Finding %i initial solutions" % window_size

        self.sum_ln_k = 0
        self.n_solutions = 0
        while n_stored < window_size:
            self.iteration=0
            self.n_solutions += 1
            while True:
                for i in range(100):
                    self.next_solution()
                    self.curr_sol = self.package_solution()                
#                   p = self.interior_point(self.curr_sol)
                    p = self.curr_sol.vertex[:self.nVars+1]

                if p is not None: 
                    break
                
                print 'SAME VERTEX!'
                assert 0

            if in_simplex(p[1:], tol=1e30, eq_tol=1e-5):
                project(p[1:])
                store[:,n_stored] = p[1:]
                n_stored += 1
                print n_stored
#           else:
#               print 'FAILED'

        #-----------------------------------------------------------------------
        # Now we can start the random walk
        #-----------------------------------------------------------------------

        #est_evec = None
        vec = store[:,:n_stored].mean(axis=1)
        self.vec = vec
        assert in_simplex(vec, tol=0, eq_tol=1e-4)
        print '**********'
        print '**********'

        print 'window_size', window_size
        print 'redo', redo

        print 'vec', vec

        self.accepted = 0
        self.rejected = 0
        for i in range(nmodels):

            #if i==0 or random_integers(nmodels+window_size)/4 > i+window_size:
            if (i%20) == 0:
                print 'Computing eigenvalues...'
                compute_eval()
                #print '**', est_evec[0,:]
                #assert 0

            print 'Walking...'
            self.rwalk()

#           accepted_once = False
#           for j in range(redo):
#               self.np[:] = random_direction(self.est_evec) 
#               self.np += self.vec
#               if in_simplex(np):
#                   vec[:] = np
#                   self.accepted += 1
#                   accepted_once = True
#               else:
#                   self.rejected += 1
#           if accepted_once:
#               print "ACCEPTED ONCE"
            #print vec;

            project(self.vec)
            #assert in_simplex(vec)
            store[:,n_stored] = self.vec
            n_stored += 1
            #assert 0
            print 'Stored %i models' % n_stored

        # 1. Select initial set of random points
        # 2. Calculate the mean and eigenvectors of said points
        # 3. Find the distance of each eigenvector to the simplex edge
        # 4. Take random step
        # 5. Compute if in simplex
        # 6. Accept original or new point
        # 7. Recompute eigenvectors every so often

        #for s in store[:,window_size:].T:
        for s in store[:,window_size:n_stored].T:
        #for s in store[:,:n_stored].T:
            q = zeros(dim+1, order='Fortran', dtype=numpy.float64)
            q[1:] = s
            yield q

        print '%i Acceptance  %i Rejected' % (self.accepted, self.rejected)
        print '%.3f Acceptance rate' % (self.accepted / (self.accepted + self.rejected))

    def next_solution(self):

        step = 0
        while True:
            result = self.pivot()
            if   result == self.NOPIVOT:   
                step += 1
                if step >= self.stride:
                    break
            elif result == self.FEASIBLE:  pass
            elif result == self.UNBOUNDED: raise SamplexUnboundedError()
            else:
                Log( result )
                raise SamplexUnexpectedError("unknown pivot result = %i" % result)

            self.status()
            self.iteration += 1

    def package_solution(self):
        s = SamplexSolution()
        #print "***", self.nVars+self.nSlack+1
        s.vertex = zeros(self.nVars+self.nSlack+1)

        assert self.lhv.size == self.nLeft+1, '%i %i' % (self.lhv.size, self.nLeft+1)
        s.lhv = self.lhv.copy()
        s.vertex[self.lhv[1:]] = self.data[1:self.nLeft+1,0]
        s.vertex[0] = self.data[0,0]

        #print 'Testing solution is negative...'
        assert all(s.vertex[1:] >= 0), ("Negative vertex coordinate!", s.vertex[s.vertex < 0])
        #print 'Nope.'

        #assert all(s.vertex[1:] >= -self.SML), ("Negative vertex coordinate!", s.vertex[s.vertex < 0])
        #s.vertex[0] = self.data[0,0]

        return s

    def find_feasible(self):

        if self.nTemp == 0: return True

        Log( "find_feasible" )
        self.set_auxil_objective()

        #print self.data
        #print self.lhv
        #print self.rhv
        Log( "------------------------" )
        self.iteration = 1
        while True:
            result = self.pivot()
            #print self.data
            #print self.lhv
            #print self.rhv
            if   result == self.UNBOUNDED:   raise SamplexUnboundedError()
            elif result == self.NOPIVOT:     raise SamplexNoSolutionError()
            elif result == self.FEASIBLE:    break
            elif result == self.FOUND_PIVOT: pass

            self.status()
            self.iteration += 1

        return True

    def set_auxil_objective(self):
        # This is the same as below. Just wanted to check correctness

        # Sum the coefficients for each normal variable and store in the first row
        sum( self.data[self.lhv < 0, :self.nRight+1], axis=0, out=self.data[0,:self.nRight+1] )

        self.data[0,:self.nRight+1] *= -1
        print 'Auxiliary obj fn', self.data[0]
        return


        #print self.data
        #print self.lhv
        #print self.rhv
        for r in xrange(self.nRight+1):
            col = self.data[:,r]
            col[0] = 0
            for k in xrange(1,self.nLeft+1):
                if self.lhv[k] < 0:
                    col[0] -= col[k]

        #self.data[0,:] = [0,0,2,-4,0]
        Log( "Auxiliary objective function" )
        #print self.data[0,:]
        #print self.nLeft, self.nRight
        #print self.lhv[1:]
        #sys.exit(0)
        #print "-----------------------------------"

    #=========================================================================

    def interior_point_ORIG(self, r=None):
        if r is None: r = random()

        sol = self.curr_sol

        k = 0
        smallest_scale = inf
        best_iv=0
        best_dist = 0
        best_moca = 0
        best_dist_err = 0

#       for i in xrange(1, len(self.lhv)+1):
#           l2 = self.lhv[i]
#           iv = self.data[l2,0]
#           dist = iv - self.moca[l2]
#           #dist_err = dist + self.moca[l2] - iv

##           scale = iv / dist
##           scale[dist < self.SML] = inf
##           smallest_scale = min(smallest_scale, min(scale))


#           if dist > self.SML:
#               scale = iv / dist
#               if scale < smallest_scale:
#                   smallest_scale = scale
#                   k = smallest_scale * (1.0-r)

#               assert iv+self.SML < dist, '%f !< %f' % (iv+self.SML, dist)

        #for i in sol.lhv: print sol.vertex[i], self.moca[i]

        iv    = sol.vertex[sol.lhv[1:]]
        dist  = iv - self.moca[sol.lhv[1:]]
        a = dist > self.SML
        if not any(a):
            return None #self.moca.copy(), 'g'

        scale = iv[a] / dist[a]

        #print iv
        #print self.moca[sol.lhv]
        #print dist[a]
        #print scale
        #print scale[dist > self.SML]
        smallest_scale = amin(scale) 
        #print 'interior point: smallest scale is %.15e' % smallest_scale
        #print 'interior point: r is %.15e' % r
        #smallest_scale = min(smallest_scale, min(scale[dist > self.SML]))

        assert not isinf(smallest_scale)
        assert smallest_scale > 0.99, smallest_scale

        print dist

        k = smallest_scale * (1.0-r)

        self.sum_ln_k += log(k)
        #assert self.sum_ln_k < 1

        #old_moca = self.moca.copy()

        spanvars = slice(1,self.nVars+self.nSlack+1)
        self.moca[spanvars] = sol.vertex[spanvars] + k * (self.moca[spanvars]-sol.vertex[spanvars])
        #assert all(self.moca >= -self.SML), self.moca[self.moca < 0]
        assert all(self.moca >= 0), (self.moca[self.moca < 0], self.moca)
        #assert all(self.moca >= -self.SML), (self.moca[self.moca < 0], self.moca)

        s = self.moca.copy()[:self.nVars+1]
        #print s
        return s


    #=========================================================================

    def interior_point(self, sol, r=None):
        if r is None: r = random()

        #sol = self.curr_sol

        k = 0
        smallest_scale = inf

#       q = slice(1, self.nVars + self.nSlack +1)
#       #q = slice(1, self.nLeft+1)
#       iv    = sol.vertex[q]
#       assert all(iv >= 0)
#       dist  = iv - self.moca[q]

        iv    = sol.vertex[sol.lhv[1:]]
        assert all(iv >= 0), iv[iv < 0]
        dist  = iv - self.moca[sol.lhv[1:]]
        a = dist > 0 #self.EPS
        if not any(a):
            return None #self.moca.copy(), 'g'

        scale = iv[a] / dist[a]

        smallest_scale = amin(scale) 
        #print 'interior point: smallest scale is %.15e' % smallest_scale
        #print 'interior point: r is %.15e' % r
        #smallest_scale = min(smallest_scale, min(scale[dist > self.SML]))

        assert not isinf(smallest_scale)
        assert smallest_scale > 0.99, smallest_scale

        #print dist

        k = smallest_scale * (1.0-r)

        self.sum_ln_k += log(k)
        #assert self.sum_ln_k < 1

        spanvars = slice(1,self.nVars+self.nSlack+1)
        q = sol.vertex[spanvars] + k * (self.moca[spanvars]-sol.vertex[spanvars])
        if any(q < 0):
            print '!! k is ', k
            print '!!', self.moca[q < 0]
            print '!!', where(q < 0)
            print 
            print '!!', sol.vertex[q < 0]
            print '!!', self.moca[q < 0]
            #print '!!', self.moca
            assert 0

        self.moca[spanvars] = q
        s = self.moca.copy()[:self.nVars+1]
        return s


    #=========================================================================

    def add_noise(self, a):
        if a[0] == 0: 
            w = abs(a) > self.EPS
            w[0] = True
            b = a.copy()
            #b[w] += self.SML * (2*random(len(w.nonzero())) - 1 )
            b[w] += self.with_noise * (random(len(w.nonzero())))
            return b
        return a

    #=========================================================================

    def eq(self, a):
        #print a
        #print self.nVars
        assert len(a)
        if self.nVars is None: 
            self.nVars = len(a)-1
            self.nRight = self.nVars
        assert len(a) == self.nVars+1, '%i != %i' % (len(a), self.nVars+1)
        self.nLeft += 1
        self.nTemp += 1
        self.eq_count += 1

        self.eq_list_no_noise.append([self._eq, a.copy()])
        self.eq_list.append([self._eq, a])

    def geq(self, a):
        assert len(a)
        if self.nVars is None: 
            self.nVars = len(a)-1
            self.nRight = self.nVars
        assert len(a) == self.nVars+1

        self.eq_list_no_noise.append([self._geq, a.copy()])
        if self.with_noise:
            a = self.add_noise(a)

        if a[0] < 0: 
            self.nLeft  += 1
            self.nSlack += 1
            self.nTemp  += 1
            self.nRight += 1
        else:
            self.nLeft  += 1
            self.nSlack += 1
        self.geq_count  += 1
        self.eq_list.append([self._geq, a])

    def leq(self, a):
        assert len(a)
        if self.nVars is None: 
            self.nVars = len(a)-1
            self.nRight = self.nVars
        assert len(a) == self.nVars+1

        self.eq_list_no_noise.append([self._leq, a.copy()])
        if self.with_noise:
            a = self.add_noise(a)

        if a[0] <= 0: 
            self.nLeft  += 1
            self.nSlack += 1
            #print "called geq "
        else:
            self.nLeft  += 1
            self.nSlack += 1
            self.nTemp  += 1
            self.nRight += 1
            #print "set -Z"
        self.leq_count += 1
        self.eq_list.append([self._leq, a])

    def _eq(self, a): 
        assert len(a) == self.nVars+1

        if a[0] < 0: 
            a = a.copy() * -1

        self.nLeft += 1
        self.nTemp += 1

        self.eq_count += 1
        self.lhv.append(-self.nTemp)
        #a[abs(a) < self.EPS] = 0
        self.data[self.nLeft, 0:1+self.nVars] = a

    def _geq(self, a): 
        assert len(a) == self.nVars+1
        self.geq_count += 1

        if a[0] < 0: 
            #a *= -1
            a = a.copy() * -1
            self._leq(a)
            self.leq_count -= 1
        else:
            self.nLeft  += 1
            self.nSlack += 1
            self.lhv.append(self.nVars+self.nSlack)
            self.data[self.nLeft, 0:1+self.nVars] = a

    def _leq(self, a): 
        assert len(a) == self.nVars+1
        self.leq_count += 1

        if a[0] <= 0: 
            #a *= -1
            a = a.copy() * -1
            self._geq(a)
            self.geq_count -= 1
        else:
            self.nLeft  += 1
            self.nSlack += 1
            self.nTemp  += 1
            self.nRight += 1

            self.lhv.append(-self.nTemp)
            self.rhv.append(self.nVars+self.nSlack)

            self.data[self.nLeft, 0:1+self.nVars] = a
            self.data[self.nLeft, self.nRight] = 1.0

