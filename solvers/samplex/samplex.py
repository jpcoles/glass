from __future__ import division
import sys
import numpy
from numpy import isfortran, asfortranarray, sign, logical_and
from numpy import set_printoptions
from numpy import insert, zeros, vstack, append, hstack, array, all, sum, ones, delete, log, empty
from numpy import argwhere, argmin, inf, isinf
from numpy import histogram, logspace, flatnonzero, isinf
from glrandom import random, ran_set_seed

from pylab import figimage, show, imshow, hist, matshow, figure

import csamplex

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
    data = None
    n_equations = 0
    lhv = []
    rhv = []
    SML = 1e-6
    EPS = 1e-14
    nVars = None            # Number of variables + 1(constant column) [N]
    nLeft = 0               # Number of left hand variables            [L]
    nSlack = 0               # Number of slack variables               [S]
    nTemp = 0               # Number of temporary variables            [Z]
    nRight = 0               # Number of right hand variables          [R]
    eq_count = 0
    leq_count = 0
    geq_count = 0


    eq_list = []

    INFEASIBLE, FEASIBLE, NOPIVOT, FOUND_PIVOT, UNBOUNDED \
        = range(5)

    iteration = 0

    moca = None

    sum_ln_k = 0

    curr_sol = None

    n_solutions = 0

    def __init__(self, ncols=None, nthreads=1):
        #print "Samplex created"
        #print "    ncols = %i" % ncols
        if ncols is not None:
            self.nVars = ncols-1
            self.nRight = self.nVars

        self.random_seed = ran_set_seed(0)

        self.nthreads = nthreads
        Samplex.pivot = lambda s: csamplex.pivot(s)

    def check_data_shape(self, len=None):

        if len is None: len = self.nLeft

        if self.data.shape[0] == len: 
            self.data = insert(self.data, self.data.shape[0], 0, axis=0)

    def start(self):
        #print "%6s %6s %6s\n%6i %6i %6i" \
        #    % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count)

        print '=' * 80
        print 'SAMPLEX'
        print '=' * 80

        print "random seed =", self.random_seed

        print "N = %i" % self.nVars
        print "L = %i" % self.nLeft
        print "R = %i" % self.nRight
        print "S = %i" % self.nSlack
        self.data = zeros((self.nLeft+1, self.nRight+1), order='Fortran', dtype=numpy.float64)

        self.nLeft = 0
        self.nSlack = 0
        self.nTemp = 0
        self.nRight = self.nVars

        self.lhv = [numpy.nan]
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
            print 'Writing out equations...'
            out = open('eqs-new', 'w')
            for f,a in self.eq_list:
                if f == self._eq:  fs = 'eq'
                if f == self._geq: fs = 'geq'
                if f == self._leq: fs = 'leq'
                print_array(out, fs, a)
            out.close()
            print 'done.'

        print "Building matrix"
        for f,a in self.eq_list:
            f(a)

        #print self.data

        print "    num eqs = %i" % len(self.eq_list)
        print "    N = %i" % self.nVars
        print "    L = %i" % self.nLeft
        print "    R = %i" % self.nRight
        print "    S = %i" % self.nSlack


        print "%6s %6s %6s\n%6i %6i %6i" \
            % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count)

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
            print >>sys.stdout, "model %i]  iter % 5i  obj-val %f" % (self.n_solutions+1, self.iteration, self.data[0,0])

    def next(self, nsolutions=None):
        print "Getting solutions"
        if not self.find_feasible(): return

        print "------------------------------------"
        print "Found feasible"
        print "------------------------------------"

        self.curr_sol = self.package_solution()                
        self.moca     = self.curr_sol.vertex.copy()

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
            self.next_solution()
            self.curr_sol = self.package_solution()                
            yield self.interior_point()

    def next_solution(self):
        self.start_new_objective()
        while True:
            result = self.pivot()
            if   result == self.NOPIVOT:   return
            elif result == self.FEASIBLE:  pass
            elif result == self.UNBOUNDED: raise SamplexUnboundedError()
            else:
                print result
                raise SamplexUnexpectedError("unknown pivot result = %i" % result)

            self.status()
            self.iteration += 1

    def package_solution(self):
        s = SamplexSolution()
        #print "***", self.nVars+self.nSlack+1
        s.vertex = zeros(self.nVars+self.nSlack+1)
        s.lhv    = zeros(self.nLeft+1, numpy.int32)

        s.lhv[0] = self.lhv[0]
        for l in range(1,self.nLeft+1):
            lq = self.lhv[l]
            s.lhv[l] = lq
            s.vertex[lq] = self.data[l,0]

        #s.lhv    = self.lhv[1:self.nLeft+1]

        #s.vertex[0]     = self.data[0,0]
        #s.vertex[s.lhv] = self.data[1:self.nLeft+1, 0]
        #s.vertex[abs(s.vertex) < self.SML] = 0
        #print s.vertex
        assert all(s.vertex >= -self.SML), ("Negative vertex coordinate!", s.vertex[s.vertex < 0])
        #assert all(s.vertex >= 0), ("Negative vertex coordinate!", s.vertex[s.vertex < 0])
        s.vertex[0] = self.data[0,0]
        return s

    def start_new_objective(self):
        if 1:
            self.obj = random(1+self.nVars+self.nSlack) - 0.5
            t = abs(self.obj) * (1-2*self.SML)
            t += self.SML
            self.obj = t * sign(self.obj)
        else:

            self.obj = random(1+self.nVars+self.nSlack) - 0.5
            # TODO:
            w = abs(self.obj) < self.SML
            while w.any():
                n = flatnonzero(w)
                self.obj[n] = random(len(n)) - 0.5
                w = abs(self.obj) < self.SML

        #self.obj[abs(self.obj) < self.EPS] = 0
        self.set_objective(self.obj)

    def set_objective(self, obj):
        if 1:
            lhv    = self.lhv[0:self.nLeft+1]
            ks     = logical_and(0 <= lhv, lhv <= self.nVars)
            obj_vs = obj[lhv[ks]]
            for r in xrange(self.nRight+1):
                col = self.data[:,r]
                n   = self.rhv[r]
                col[0] = sum(col[ks] * obj_vs)
                col[0] += obj[n] if 0 <= n <= self.nVars else 0
        else:
            #print "obj", obj
            for r in xrange(self.nRight+1):
                col = self.data[:,r]
                n   = self.rhv[r]
                col[0] = obj[n] if 0 <= n <= self.nVars else 0
                for k in xrange(1, self.nLeft+1):
                    n = self.lhv[k]
                    if 0 <= n <= self.nVars:
                        col[0] += col[k] * obj[n]

        self.data[0,0] = 0

        #print self.data[:,0]

    def find_feasible(self):

        if self.nTemp == 0: return True

        print "find_feasible"
        self.set_auxil_objective()

        #print self.data
        #print self.lhv
        #print self.rhv
        print "------------------------"
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
        sum(self.data[self.lhv < 0,:self.nRight+1], axis=0, out=self.data[0,:self.nRight+1])
        self.data[0,:self.nRight+1] *= -1
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
        print "Auxiliary objective function"
        #print self.data[0,:]
        #print self.nLeft, self.nRight
        #print self.lhv[1:]
        #sys.exit(0)
        #print "-----------------------------------"


    #=========================================================================

    def interior_point(self, r=None):
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

        iv    = sol.vertex[sol.lhv]
        dist  = iv - self.moca[sol.lhv]
        a = dist > self.SML
        scale = iv[a] / dist[a]
        if not len(dist[a]): return self.moca.copy(), 'g'

        #print iv
        #print self.moca[sol.lhv]
        #print dist[a]
        #print scale
        #print scale[dist > self.SML]
        smallest_scale = min(scale)
        #smallest_scale = min(smallest_scale, min(scale[dist > self.SML]))

        assert not isinf(smallest_scale)
        assert smallest_scale > 0.99, smallest_scale

        k = smallest_scale * (1.0-r)

        self.sum_ln_k += log(k)
        #assert self.sum_ln_k < 1

        #old_moca = self.moca.copy()

        spanvars = slice(1,self.nVars+self.nSlack+1)
        self.moca[spanvars] = sol.vertex[spanvars] + k * (self.moca[spanvars]-sol.vertex[spanvars])
        #assert all(self.moca >= -self.SML), self.moca[self.moca < 0]
        assert all(self.moca[1:] >= -self.SML), (self.moca[self.moca < 0], self.moca)

        return self.moca.copy()[0:self.nVars+1]


    #=========================================================================

    def eq(self, a):
        #print a
        #print self.nVars
        assert len(a)
        if self.nVars is None: 
            self.nVars = len(a)-1
            self.nRight = self.nVars
        assert len(a) == self.nVars+1
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
        if a[0] < 0: a *= -1

        self.nLeft += 1
        self.nTemp += 1

        self.eq_count += 1
        self.lhv.append(-self.nTemp)
        #print a
        #print self.nVars
        #print self.data.shape
        a[abs(a) < self.EPS] = 0
        self.data[self.nLeft, 0:1+self.nVars] = a

    def _geq(self, a): 
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
#           a[abs(a) < self.EPS] = 0
            self.data[self.nLeft, 0:1+self.nVars] = a
#           if self.data[self.nLeft,0] == 0: 
#               self.data[self.nLeft, 0:1+self.nVars] += self.SML * random(len(a))

            for n in xrange(self.nVars+1):
                if n==0 or abs(self.data[self.nLeft, n]) > self.EPS:
                    if self.data[self.nLeft,0] == 0: self.data[self.nLeft, n] += self.SML * random()


    def _leq(self, a): 
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
#           a[abs(a) < self.EPS] = 0
            self.data[self.nLeft, 0:1+self.nVars] = a
#           if self.data[self.nLeft,0] == 0: 
#               self.data[self.nLeft, 0:1+self.nVars] += self.SML * random(len(a))

            for n in xrange(self.nVars+1):
                if n==0 or abs(self.data[self.nLeft, n]) > self.EPS:
                    if self.data[self.nLeft,0] == 0: self.data[self.nLeft, n] += self.SML * random()

            self.data[self.nLeft, self.nRight] = 1.0

