from __future__ import division
import sys
import time
import numpy as np
from numpy.random import random, normal, random_integers, seed as ran_set_seed
from numpy.linalg import eigh, pinv, eig, norm, inv, det
import scipy.linalg.fblas

import multiprocessing.dummy as MP
from Queue import Empty as QueueEmpty

from glass.solvers.error import GlassSolverError

#from glrandom import random, ran_set_seed

#dot = lambda a, b: scipy.linalg.fblas.ddot(a, b)
#dot = lambda a, b: scipy.linalg.fblas.dgemm(1., a, b, trans_b=True)

if 0:
    from pylab import figimage, show, imshow, hist, matshow, figure

try:
    from glass.log import log as Log
except ImportError:
    def l(x):
        print x
    Log = l

try:
    from glass.log import report_status as report_status
except ImportError:
    def rs(*args, **kwargs):
        pass
    report_status = rs


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

np.set_printoptions(linewidth=10000000, precision=20, threshold=2000)

class SamplexUnboundedError(GlassSolverError):
    def __init__(self, *args, **kwargs):
        GlassSolverError.__init__(self, 'Constraints are not strong enough to form a closed solution volume.', *args, **kwargs)

class SamplexNoSolutionError(GlassSolverError):
    def __init__(self, *args, **kwargs):
        GlassSolverError.__init__(self, 'Constraints are too strong and no solution exists.', *args, **kwargs)

class SamplexUnexpectedError(GlassSolverError):
    def __init__(self, *args, **kwargs):
        GlassSolverError.__init__(self, 'An unexpected error happened.', *args, **kwargs)

class SamplexSolution:
    def __init__(self):
        self.lhv = None
        self.vertex = None

def rwalk_burnin(id, nmodels, burnin_len, samplex, q, cmdq, ackq, vec, twiddle, eval,evec):

    S   = np.zeros(samplex.eqs.shape[0])
    S0  = np.zeros(samplex.eqs.shape[0])

    vec  = vec.copy('A')
    eval = eval.copy('A')
    evec = evec.copy('A')
    eqs  = samplex.eqs.copy('A')

    accepted = 0
    rejected = 0

    offs = ' '*39
    Log( offs + 'STARTING rwalk_burnin THREAD %i' % id, overwritable=True)

    eqs[:,1:] = np.dot(samplex.eqs[:,1:], evec)
    #vec[:] = np.dot(evec.T, vec)
    I = np.eye(evec.shape[0]).copy('F')

    csamplex.set_rwalk_seed(1 + id + samplex.random_seed)

    offs = ' '*36
    i = 0
    j=0
    while True:
        j+= 1


        done = False
        try:
            while not done:
                cmd = cmdq.get()
                if cmd[0] == 'CONT':
                    break
                elif cmd[0] == 'NEW DATA':
                    eval[:],evec[:],twiddle = cmd[1]
                    eqs[:,1:] = np.dot(samplex.eqs[:,1:], evec)
                elif cmd[0] == 'REQ TWIDDLE':
                    ackq.put(twiddle)
                elif cmd[0] == 'WAIT':
                    ackq.put('OK')
                elif cmd[0] == 'STOP':
                    done = True
                elif cmd[0] == 'RWALK':
                    done = True
                else:
                    print 'Unknown cmd:', cmd
        except QueueEmpty:
            pass

        if done:
            break

        vec[:] = np.dot(evec.T, vec)

        accepted = False
        while not accepted:
            Naccepted = 0
            Nrejected = 0

            Naccepted,Nrejected,t = csamplex.rwalk(samplex, eqs, vec,eval,S,S0, twiddle, Naccepted,Nrejected)

            r = Naccepted / (Naccepted + Nrejected)
            msg = 'THREAD %3i]  %i/%i  %4.1f%% accepted  (%6i/%6i Acc/Rej)  twiddle %5.2f  time %5.3fs' % (id, i, burnin_len, 100*r, Naccepted, Nrejected, twiddle, t)

            #-------------------------------------------------------------------
            # If the actual acceptance rate was OK then leave this loop,
            # otherwise change our step size twiddle factor to improve the rate.
            # Even if the accepance rate was OK, we adjust the twiddle but only
            # with a certain probability. This drives the acceptance rate to 
            # the specified one even if we are within the tolerance but doesn't
            # throw away the results if we are not so close. This allows for
            # a larger tolerance.
            #-------------------------------------------------------------------
            accepted =  np.abs(r - samplex.accept_rate) < samplex.accept_rate_tol

            state = 'B'
            if not accepted:
                twiddle *= 1 + ((r-samplex.accept_rate) / samplex.accept_rate / 2)
                twiddle = max(1e-14,twiddle)
                state = 'R' + state

            Log( offs + '% 2s %s' % (state, msg), overwritable=True )
            #print ' '*36, '% 2s %s' % (state, msg)

        vec[:] = np.dot(evec, vec)

        if random() < np.abs(r - samplex.accept_rate)/samplex.accept_rate_tol:
            twiddle *= 1 + ((r-samplex.accept_rate) / samplex.accept_rate / 2)
            twiddle = max(1e-14,twiddle)

        assert np.all(vec >= 0), vec[vec < 0]
        #if np.any(vec < 0): sys.exit(0)

        samplex.project(vec)

        i += 1
        q.put([id,vec.copy('A')])

    time_begin = time.clock()
    if cmd[0] == 'RWALK':
        rwalk(id, nmodels, samplex, q, cmdq, vec, twiddle, eval, evec)
    time_end = time.clock()

    cmd = cmdq.get()
    assert cmd[0] == 'STOP', cmd[0]
    ackq.put(['TIME', time_end-time_begin])

    #print ' '*39, 'RWALK THREAD %i LEAVING  n_stored=%i  time=%.4fs' % (id,i,time_end-time_begin)

def rwalk(id, nmodels, samplex, q, cmdq, vec,twiddle, eval,evec):

    S   = np.zeros(samplex.eqs.shape[0])
    S0  = np.zeros(samplex.eqs.shape[0])

    vec  = vec.copy('A')
    eqs  = samplex.eqs.copy('A')

    accepted = 0
    rejected = 0

    offs = ' '*39
    Log( offs + 'STARTING rwalk THREAD %i [this thread makes %i models]' % (id,nmodels), overwritable=True)

    eqs[:,1:] = np.dot(samplex.eqs[:,1:], evec)

    csamplex.set_rwalk_seed(1 + id + samplex.random_seed)

    offs = ' '*36
    state = ''
    for i in xrange(nmodels):

        accepted = 0
        rejected = 0

        done = False

        vec[:] = np.dot(evec.T, vec)
        accepted,rejected,t = csamplex.rwalk(samplex, eqs, vec,eval,S,S0, twiddle, accepted,rejected)
        vec[:] = np.dot(evec, vec)

        r = accepted / (accepted + rejected)
        Log( offs + '% 2s THREAD %3i  %i  %4.1f%% accepted  (%6i/%6i Acc/Rej)  twiddle %5.2f  time %5.3fs  %i left.' % (state, id, i, 100*r, accepted, rejected, twiddle, t, nmodels-i), overwritable=True )
        #print ' '*36, '% 2s THREAD %3i  %i  %4.1f%% accepted  (%6i/%6i Acc/Rej)  twiddle %5.2f  time %5.3fs  %i left.' % (state, id, i, 100*r, accepted, rejected, twiddle, t, nmodels-i)
        assert np.all(vec >= 0), vec[vec < 0]
        #if numpy.any(vec < 0): sys.exit(0)

        samplex.project(vec)

        q.put([id,vec.copy('A')])

class Samplex:
    INFEASIBLE, FEASIBLE, NOPIVOT, FOUND_PIVOT, UNBOUNDED = range(5)
    SML = 1e-5
    EPS = 1e-14

    def __init__(self, **kw):

        ncols                   = kw.get('ncols', None)
        nthreads                = kw.get('nthreads', 1)
        rngseed                 = kw.get('rngseed',  None)
        self.stride             = kw.get('stride', 1)
        self.accept_rate        = kw.get('acceptance rate', 0.25)
        self.accept_rate_tol    = kw.get('acceptance tol', 0.05)
        self.redo_factor        = kw.get('redo factor', 1)
        self.redo_exp           = kw.get('redo exp', 2)
        self.twiddle            = kw.get('twiddle', 2.4)
        self.burnin_factor      = kw.get('burnin factor', 10)
        
        #self.report             = kw.get('reporter', lambda _: None)

        assert ncols is not None
        self.nVars = ncols

        if rngseed is None:
            self.random_seed = int(time.time())
        else:
            self.random_seed = rngseed

        self.nthreads = nthreads

        self.eq_count = 0
        self.leq_count = 0
        self.geq_count = 0

        self.eq_list = []
        self.ineqs = []

        self.avg0 = None


    def start(self):

        def eq_key(x):
            if x[0] == 'geq': return 2
            if x[0] == 'leq': return 1
            if x[0] == 'eq':  return 0
            assert False, 'Bad function %s' % str(x[0])

        self.eq_list.sort(key=eq_key)

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


        self.eqn_count = self.eq_count + self.geq_count + self.leq_count

        if 0:
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


    def next(self, nsolutions=None):
    # this does the first part of the cpu intensive tasks

        Log( '=' * 80 )
        Log( 'Simplex Random Walk' )
        Log( '=' * 80 )

        Log( "    %i equations" % len(self.eq_list) )

        Log( "%6s %6s %6s\n%6i %6i %6i" 
            % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count) )


        if nsolutions == 0: return

        assert nsolutions is not None

        dim = self.nVars
        dof = dim - self.eq_count

        burnin_len  = max(10, int(self.burnin_factor * dof))
        redo        = max(100,  int((dof ** self.redo_exp) * self.redo_factor))

        nmodels = nsolutions
        nthreads = self.nthreads

        self.stride = int(dim+1)

        n_stored = 0
        self.dim = dim
        self.dof = dof
        self.redo = redo

        self.burnin_len = burnin_len

        accept_rate     = self.accept_rate
        accept_rate_tol = self.accept_rate_tol

        store = np.zeros((dim, 1+burnin_len), order='Fortran', dtype=np.float64)
        newp = np.zeros(dim, order='C', dtype=np.float64)
        eval  = np.zeros(dim, order='C', dtype=np.float64)
        evec  = np.zeros((dim,dim), order='F', dtype=np.float64)

        self.eqs = np.zeros((self.eqn_count+dim,dim+1), order='C', dtype=np.float64)
        for i,[c,e] in enumerate(self.eq_list):
            self.eqs[i,:] = e
        for i in xrange(dim):
            self.eqs[self.eqn_count+i,1+i] = 1

        self.dist_eqs = np.zeros((self.eqn_count-self.eq_count,dim+1), order='C', dtype=np.float64)
        i=0
        for c,e in self.eq_list:
            if c == 'eq':
                continue
            elif c == 'leq':
                p = e
            elif c == 'geq':
                p = -e
            self.dist_eqs[i,:] = p
            i += 1

        Log( 'Using lpsolve %s' % lpsolve('lp_solve_version') )
        Log( "random seed = %s" % self.random_seed )
        Log( "threads = %s" % self.nthreads )
        Log( "acceptence rate = %s" % self.accept_rate )
        Log( "acceptence rate tolerance = %s" % self.accept_rate_tol )
        Log( "dof = %s" % self.dof)
        Log( "sample distance = max(100,%s * %s^%s) = %s" % (self.redo_factor, self.dof, self.redo_exp, redo) )
        Log( "starting twiddle = %s" % self.twiddle )
        Log( "burn-in length = %s" % burnin_len )

        time_begin_next = time.clock()

        #-----------------------------------------------------------------------
        # Create pseudo inverse matrix to reproject samples back into the
        # solution space.
        #-----------------------------------------------------------------------
        P = np.eye(dim) 
        if self.eq_count > 0:
            self.A = np.zeros((self.eq_count, dim), order='C', dtype=np.float64)
            self.b = np.zeros(self.eq_count, order='C', dtype=np.float64)
            for i,[c,e] in enumerate(self.eq_list[:self.eq_count]):
                self.A[i] = e[1:]
                self.b[i] = e[0]
            self.Apinv = pinv(self.A)
            P -= np.dot(self.Apinv, self.A)
        else:
            self.A = None
            self.B = None
            self.Apinv = None

        ev, evec = eigh(P)
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Find a point that is completely inside the simplex
        #-----------------------------------------------------------------------
        Log('Finding first inner point')
        time_begin_inner_point = time.clock()
        self.inner_point(newp)
        time_end_inner_point = time.clock()
        ok,fail_count = self.in_simplex(newp, eq_tol=1e-12, tol=0, verbose=1)
        assert ok

        self.avg0 = newp

#       eqs  = self.eqs.copy('A')
#       eqs[:,1:] = np.dot(self.eqs[:,1:], evec)

#       print newp

#       S = zeros(self.eqs.shape[0])
#       newp[:] = np.dot(evec.T, newp)
#       newp0 = newp.copy()
#       steps = newp.copy()
#       for q in range(100):
#           csamplex.refine_center(self, eqs, newp, ev, S, steps)
#           d = newp - newp0
#           #print d
#           print norm(d)
#           #print
#           newp0 = newp.copy()

#       #assert 0
#       newp[:] = np.dot(evec, newp)


        store[:,0] = newp
        n_stored = 1


        #-----------------------------------------------------------------------
        # Estimate the eigenvectors of the simplex
        #-----------------------------------------------------------------------
        Log('Estimating eigenvectors')
        time_begin_est_eigenvectors = time.clock()
        self.measured_ev(newp, ev, eval, evec)
        time_end_est_eigenvectors = time.clock()

        #-----------------------------------------------------------------------
        # Now we can start the random walk
        #-----------------------------------------------------------------------

        Log( "Getting solutions" )

        q = MP.Queue()

        #-----------------------------------------------------------------------
        # Launch the threads
        #-----------------------------------------------------------------------
        threads = []
        models_per_thread = nmodels // nthreads
        models_under      = nmodels - nthreads*models_per_thread
        id,N = 0,0
        while id < nthreads and N < nmodels:
            n = models_per_thread
            if id < models_under:
                n += 1
            assert n > 0
            Log( 'Thread %i gets %i' % (id,n) )
            cmdq = MP.Queue()
            ackq = MP.Queue()
            thr = MP.Process(target=rwalk_burnin, args=(id, n, int(np.ceil(burnin_len/nthreads)), self, q, cmdq, ackq, newp, self.twiddle, eval.copy('A'), evec.copy('A')))
            threads.append([thr,cmdq,ackq])
            N += n
            id += 1

        assert N == nmodels

        for thr,cmdq,_ in threads:
            thr.start()
            cmdq.put(['CONT'])

        def drainq(q):
            try:
                while True:
                    q.get(block=False)
            except QueueEmpty:
                pass

        def pause_threads(threads):
            for _,cmdq,ackq in threads:
                cmdq.put(['WAIT'])
                assert ackq.get() == 'OK'

        def adjust_threads(i, cont_cmd):
            pause_threads(threads)
            drainq(q)
            Log( 'Computing eigenvalues... [%i/%i]' % (i, burnin_len) )
            self.compute_eval_evec(store, eval, evec, n_stored)

            # new twiddle <-- average twiddle
            t = 0
            for _,cmdq,ackq in threads:
                cmdq.put(['REQ TWIDDLE'])
                t += ackq.get()
            t /= len(threads)

            Log( 'New twiddle %f' % t )
            for _,cmdq,_ in threads:
                cmdq.put(['NEW DATA', [eval.copy('A'), evec.copy('A'), t]])
                cmdq.put([cont_cmd])

        #-----------------------------------------------------------------------
        # Burn-in
        #-----------------------------------------------------------------------
        time_begin_burnin = time.clock()
        compute_eval_window = 2 * self.dof
        j = 0
        for i in xrange(burnin_len):
            j += 1
            k,vec = q.get()

            store[:, n_stored] = vec
            n_stored += 1

            if j == compute_eval_window:
                j = 0
                adjust_threads(i+1,'CONT')
                compute_eval_window = int(0.1*burnin_len + 1)
            elif len(threads) < compute_eval_window:
                threads[k][1].put(['CONT'])
        time_end_burnin = time.clock()

        #-----------------------------------------------------------------------
        # Actual random walk
        #-----------------------------------------------------------------------
        time_begin_get_models = time.clock()
        adjust_threads(burnin_len, 'RWALK')
        i=0
        while i < nmodels:
            k,vec = q.get()
            t = np.zeros(dim+1, order='Fortran', dtype=np.float64)
            t[1:] = vec
            i += 1
            Log( '%i models left to generate' % (nmodels-i), overwritable=True)
            yield t

        time_end_get_models = time.clock()

        #-----------------------------------------------------------------------
        # Stop the threads and get their running times.
        #-----------------------------------------------------------------------
        time_threads = []
        for thr,cmdq,ackq in threads:
            cmdq.put(['STOP'])
            m,t = ackq.get()
            assert m == 'TIME'
            time_threads.append(t)
            #thr.terminate()

        time_end_next = time.clock()

        max_time_threads = np.amax(time_threads) if time_threads else 0
        avg_time_threads = np.mean(time_threads) if time_threads else 0

        Log( '-'*80 )
        Log( 'SAMPLEX TIMINGS' )
        Log( '-'*80 )
        Log( 'Initial inner point    %.2fs' % (time_end_inner_point - time_begin_inner_point) )
        Log( 'Estimate eigenvectors  %.2fs' % (time_end_est_eigenvectors - time_begin_est_eigenvectors) )
        Log( 'Burn-in                %.2fs' % (time_end_burnin - time_begin_burnin) )
        Log( 'Modeling               %.2fs' % (time_end_get_models - time_begin_get_models) )
        Log( 'Max/Avg thread time    %.2fs %.2fs' % (max_time_threads, avg_time_threads) )
        Log( 'Total wall-clock time  %.2fs' % (time_end_next - time_begin_next) )
        Log( '-'*80 )

    def in_simplex(self, newp, tol=0, eq_tol=1e-8, verbose=0):
        bad = []

        a_min = np.inf
        for i,[c,e] in enumerate(self.eq_list):
            a0 = np.dot(newp, e[1:])
            a  = e[0] + a0

            #print newp.flags, e[1:].flags
            #assert 0
            if c == 'geq':
                a_min = min(a_min, a)
                if a < -tol: 
                    if verbose: Log( 'F> %i %e' % (i,a) )
                    bad.append([i,a])
            elif c == 'leq':
                a_min = min(a_min, a)
                if a > tol: 
                    if verbose: Log( 'F< %i %e' % (i,a) )
                    bad.append([i,a])
            elif c == 'eq':
                if np.abs(a) > eq_tol: 
                    if verbose: Log( 'F= %i %e %e' %(i,a, (1 - np.abs(e[0]/a0))) )
                    bad.append([i,a])

            #if verbose > 1: print "TT", c, a
              
        if verbose > 1:
            Log( 'Smallest a was %e' % (a_min,) )

        #print 'T '
        return not bad, bad

    def distance_to_plane(self,pt,dir):
        ''' dir should be a normalized vector '''

        dist = np.inf

        p = self.dist_eqs
        a = np.dot(dir, p[:,1:].T)
        w = a > 0

        if w.any():
            dtmp = -(p[:,0] + np.dot(pt, p[:,1:].T)) / a
            dist = np.amin(dtmp[w])

        # check implicit >= 0 contraints
        a = -dir 
        w = a > 0
        if w.any():
            dtmp = pt[w] / a[w]
            dist = np.amin([dist, np.amin(dtmp)])

        assert dist != np.inf

        return dist

    def compute_eval_evec(self, store, eval, evec, n_stored):

        #s = max(0, n_stored - ceil(0.5*self.burnin_len))
        s = 0

        eval0,evec0 = eigh(np.cov(store[:,  s:n_stored]))
        avg = store[:,  s:n_stored].mean(axis=1)

        if self.avg0 is not None:
            Log( 'average store delta %s' % str(norm(avg-self.avg0)) )
        self.avg0 = avg.copy()

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
                eval[r] = (tmax2 - tmax1) / np.sqrt(12)

        evec[:] = evec0
        #print 'eval(inside)', eval
        if nzero != self.eq_count:
            Log( '!'*80 )
            Log( 'ERROR:', 'Expected number of zero length eigenvectors (%i) to equal number of equality constraints (%i)' % (nzero, self.eq_count) )
            Log( '!'*80 )
            sys.exit(0)

    def project(self,x):
        if self.Apinv is not None:
            q = np.dot(self.A, x)
            q += self.b
            q = np.dot(self.Apinv, q)
            x -= q

    def inner_point(self, newp):

        lp = lpsolve('make_lp', 0, self.nVars+1) # +1 for variable used to find the first inner point
        lpsolve('set_epsb', lp, 1e-14)
        lpsolve('set_epsd', lp, 1e-14)
        lpsolve('set_epsint', lp, 1e-14)
        lpsolve('set_epsel', lp, 1e-8)
        lpsolve('set_verbose', lp, FULL)
        lpsolve('set_sense', lp, False)

        for eq,a in self.eq_list:
            l = (a[1:]).tolist()
            if eq ==  'eq': l.append(0); lpsolve('add_constraint', lp, l, EQ, -a[0])
            if eq == 'leq': l.append(1); lpsolve('add_constraint', lp, l, LE, -a[0])
            if eq == 'geq': l.append(1); lpsolve('add_constraint', lp, l, GE, -a[0])

        for i in range(self.nVars):
            q = np.zeros(self.nVars+1)
            q[[i,-1]] = -1, 1
            lpsolve('add_constraint', lp, q.tolist(), LE, 0)

        o = np.zeros(self.nVars+1)
        o[-1] = 1
        lpsolve('set_obj_fn', lp, o.tolist())
        while True:
            result = lpsolve('solve', lp)
            if   result in [OPTIMAL, TIMEOUT]:   break
            elif result == SUBOPTIMAL: continue
            elif result == INFEASIBLE: raise SamplexNoSolutionError()
            elif result == UNBOUNDED: raise SamplexUnboundedError()
            else:
                Log( result )
                raise SamplexUnexpectedError("unknown pivot result %i from linear solver." % result)

        objv  = np.array(lpsolve('get_objective', lp))
        v1    = np.array(lpsolve('get_variables', lp)[0])
        assert len(v1) == lpsolve('get_Norig_columns', lp)
        assert len(v1) == self.nVars+1
        del lp

        v1 = v1[:-1] # Remove the temporary variable that tracks the distance from the simplex boundary
        v1[np.abs(v1) < 1e-14] = 0
        assert np.all(v1 >= 0), v1[v1 < 0]

        ok,fail_count = self.in_simplex(v1, eq_tol=1e-12, tol=0, verbose=1)
        ok,fail_count = self.in_simplex(v1, eq_tol=1e-12, tol=-1e-13, verbose=1)
        assert ok, len(fail_count)
        newp[:] = v1
        self.project(newp)
        ok,fail_count = self.in_simplex(newp, eq_tol=1e-12, tol=0, verbose=1)
        ok,fail_count = self.in_simplex(newp, eq_tol=1e-12, tol=-1e-5, verbose=1)

    def measured_ev(self, newp, ev, eval, evec, eval_tol=1e-12):
        nzero = 0
        for r in range(eval.size):
            if ev[r] < eval_tol:
                eval[r] = 0
                nzero += 1
            else:
                direction = evec[:,r]
                tmax1 = -self.distance_to_plane(newp, -direction)
                tmax2 = +self.distance_to_plane(newp, +direction)
                eval[r] = (tmax2 - tmax1) / np.sqrt(12)
                assert tmax1 < tmax2, 'tmax %i %i  ev[%i] %e' % (tmax1, tmax2, r, ev[r])

        assert nzero == self.eq_count, "Number of zero length eigenvectors doesn't equal number of equalities. (%i != %i)" % (nzero, self.eq_count)

    #=========================================================================

    TT = 1e-14
    def eq(self, a):
        self.eq_count += 1
        self.eq_list.append(['eq', a])

    def geq(self, a):
        return self.leq(-a)

    def leq(self, a):
        self.leq_count += 1

        self.ineqs.append([-a[1:], GE, a[0]])
        self.eq_list.append(['leq', a])

