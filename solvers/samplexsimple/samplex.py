from __future__ import division
import sys
import numpy
import gc
from numpy import isfortran, asfortranarray, sign, logical_and, any
from numpy import set_printoptions
from numpy import insert, zeros, vstack, append, hstack, array, all, sum, ones, delete, log, empty, dot, sqrt, arange
from numpy import argwhere, argmin, inf, isinf, amin, abs, where
from numpy import histogram, logspace, flatnonzero, isinf
from numpy.random import random, normal, random_integers, seed as ran_set_seed
#from glrandom import random, ran_set_seed

if 0:
    from pylab import figimage, show, imshow, hist, matshow, figure

from log import log as Log

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
        self.with_noise   = kw.get('add noise', False)

        Log( "Samplex created" )
        Log( "    ncols = %i" % ncols )
        if ncols is not None:
            self.nVars = ncols
            self.nRight = self.nVars

        csamplex.set_rnd_cseed(rngseed)

        self.random_seed = rngseed

        self.nthreads = nthreads
        Samplex.pivot = lambda s: csamplex.pivot(s)

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
        Log( 'SAMPLEX (Simple)' )
        Log( '=' * 80 )

        Log( "random seed = %s" % self.random_seed )
        Log( "threads = %s" % self.nthreads )
        Log( "solution type = %s" % self.sol_type )
        Log( "with noise = %s" % self.with_noise )

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
            if x[0] == self._geq: return 0
            if x[0] == self._leq: return 1
            if x[0] == self._eq:  return 2
            assert False, 'Bad function %s' % str(x[0])

        self.eq_list.sort(key=eq_key)
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
        del self.eq_list



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
            Log( "model %i]  iter % 5i  obj-val %f" % (self.n_solutions+1, self.iteration, self.data[0,0]) )

    def next(self, nsolutions=None):

        Log( "Getting solutions" )
        if not self.find_feasible(): return

        Log( "------------------------------------" )
        Log( "Found feasible" )
        Log( "------------------------------------" )


        self.curr_sol = self.package_solution()                
        self.moca     = self.curr_sol.vertex.copy()

        self.dcopy = [self.data.copy('F'),
                      self.lhv.copy(),
                      self.rhv.copy(),
                      self.nVars,
                      self.nLeft,
                      self.nSlack,
                      self.nTemp,
                      self.nRight]

        #yield self.curr_sol.vertex[0:self.nVars+1]

        #spanvars = slice(1,self.nVars+self.nSlack+1)
        #self.moca = self.data[spanvars, 0].copy()

#       print histogram(self.data[0:self.nLeft+1, 0:self.nRight+self.nSlack+1],
#                       bins = logspace(-14, 4),
#                       new = False)

##      x = self.data[0:self.nLeft+1, 0:self.nRight+self.nSlack+1].flatten()
##      hist(x, bins = 100, log=True)
##      show()
##      sys.exit(0)

        self.sum_ln_k = 0
        self.n_solutions = 0
        while self.n_solutions != nsolutions:
            self.iteration=0
            self.n_solutions += 1
            while True:
                self.next_solution()
                self.curr_sol = self.package_solution()                

                #print self.sol_type
                if self.sol_type == 'vertex':
                    p = self.curr_sol.vertex[:self.nVars+1].copy()
                elif self.sol_type == 'interior':
                    p = self.interior_point(self.curr_sol)

                if p is not None: 
                    break
                
                print 'SAME VERTEX!'

            #print 'sol', p
            yield p

    def next_solution(self):

        while True:
            result = self.pivot()
            if   result == self.NOPIVOT:   break
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

        print 'Testing solution is negative...'
        assert all(s.vertex[1:] >= 0), ("Negative vertex coordinate!", s.vertex[s.vertex < 0])
        print 'Nope.'

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
        print 'interior point: smallest scale is %.15e' % smallest_scale
        print 'interior point: r is %.15e' % r
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
        assert all(iv > 0)
        dist  = iv - self.moca[sol.lhv[1:]]
        a = dist > 0 #self.EPS
        if not any(a):
            return None #self.moca.copy(), 'g'

        scale = iv[a] / dist[a]

        smallest_scale = amin(scale) 
        print 'interior point: smallest scale is %.15e' % smallest_scale
        print 'interior point: r is %.15e' % r
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
            b[w] += self.SML * (2*random(len(w.nonzero())) - 1 )
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
        self.eq_list.append([self._eq, a])

    def geq(self, a):
        assert len(a)
        if self.nVars is None: 
            self.nVars = len(a)-1
            self.nRight = self.nVars
        assert len(a) == self.nVars+1

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

        if a[0] < 0: a *= -1

        self.nLeft += 1
        self.nTemp += 1

        self.eq_count += 1
        self.lhv.append(-self.nTemp)
        #print a
        #print self.nVars
        #print self.data.shape
        #a[abs(a) < self.EPS] = 0
        self.data[self.nLeft, 0:1+self.nVars] = a

    def _geq(self, a): 
        assert len(a) == self.nVars+1
        self.geq_count += 1

        if a[0] < 0: 
            a *= -1
            self._leq(a)
            self.leq_count -= 1
        else:
            self.nLeft  += 1
            self.nSlack += 1
            self.lhv.append(self.nVars+self.nSlack)

            #-------------------------------------------------------------------
            # First copy the array into data before adding noise so that the
            # values are correctly typecast.
            #-------------------------------------------------------------------

            self.data[self.nLeft, 0:1+self.nVars] = a

    def _leq(self, a): 
        assert len(a) == self.nVars+1
        self.leq_count += 1

        if a[0] <= 0: 
            a *= -1
            self._geq(a)
            self.geq_count -= 1
        else:
            self.nLeft += 1
            self.nSlack += 1
            self.nTemp += 1
            #print "nRight=", self.nRight
            self.nRight += 1

            self.lhv.append(-self.nTemp)
            self.rhv.append(self.nVars+self.nSlack)

            #print "adding ", a
            #print self.data[self.nLeft, 0:1+self.nVars]
            #print self.nLeft, self.nRight
            #print self.data.shape

            #-------------------------------------------------------------------
            # First copy the array into data before adding noise so that the
            # values are correctly type cast.
            #-------------------------------------------------------------------

            self.data[self.nLeft, 0:1+self.nVars] = a
            self.data[self.nLeft, self.nRight] = 1.0

