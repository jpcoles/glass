
import sys
import time, datetime
import numpy as np
from numpy.random import SeedSequence, default_rng
try:
    from numpy.dual import pinv, norm, inv, det
except:
    from numpy.linalg import pinv, norm, inv, det

from scipy.optimize import linprog

import multiprocessing as MP
from queue import Empty as QueueEmpty

from glass.solvers.error import GlassSolverError
from glass.report import section

if 0:
    from pylab import figimage, show, imshow, hist, matshow, figure

try:
    from glass.log import log as Log
except ImportError:
    def l(x):
        print(x)
    Log = l

np.set_printoptions(linewidth=10000000, precision=20, threshold=2000)

class SamplexUnboundedError(GlassSolverError):
    def __init__(self, *args, **kwargs):
        GlassSolverError.__init__(self, 'Constraints are not strong enough to form a closed solution volume.', *args, **kwargs)

class SamplexNoSolutionError(GlassSolverError):
    def __init__(self, *args, **kwargs):
        GlassSolverError.__init__(self, 'Constraints are too strong and no solution exists.', *args, **kwargs)

class SamplexNumericalError(GlassSolverError):
    def __init__(self, *args, **kwargs):
        GlassSolverError.__init__(self, 'A numerical error occured and a starting solution could not be found.', *args, **kwargs)

class SamplexUnexpectedError(GlassSolverError):
    def __init__(self, *args, **kwargs):
        GlassSolverError.__init__(self, 'An unexpected error happened.', *args, **kwargs)

class SamplexSolution:
    def __init__(self):
        self.lhv = None
        self.vertex = None

def timestr(dt):
    dt = datetime.timedelta(seconds=dt)
    daystr = '%i-' % dt.days if dt.days else ''
    return daystr + time.strftime('%H:%M:%S', time.gmtime(dt.seconds + (dt.microseconds>0)))

def project(x, A,b, Apinv):
    if Apinv is not None:
        q = np.dot(A, x)
        q += b
        x -= np.dot(Apinv, q)

def new_dir(ip, projfn, rngfn):
    """Returns a new, normalized, direction vector that lies on the solution plane.

    The vector is not a point within the solution space, however.
    """
    while True:
        # Pick a random point on an n-sphere.
        u = rngfn()
        d = norm(u)
        p = u/d
        # Project that vector onto the solution plane
        p += ip
        projfn(p)
        p -= ip
        # We'll need to renormalize this projected vector.
        # But first reject very small vectors that could cause numerical problems.
        n = norm(p)
        if n > 1e-10: 
            break
        print('Rejecting new direction with length ', n)

    p /= n
    return p

def distance_to_plane_faster(pt,dir,p0,p):
    ''' dir should be a normalized vector '''

    dist = np.inf

    #p = self.dist_eqs
    #p0,p = self.dist_eqs
    #a = np.dot(dir, p)
    a = dir @ p
    w = np.nonzero(a > 0)[0]

    if len(w):
        # dtmp = -(p[:,0] + np.dot(pt, p[:,1:].T)) / a
        # ==>
        dtmp = pt @ p
        dtmp += p0
        dtmp *= -1
        dtmp /= a
        dist = np.amin(dtmp[w])


    # check implicit >= 0 contraints
    #a = -dir 
    #w = a > 0
    w = np.nonzero(dir < 0)[0]
    if len(w):
        dtmp = pt[w] / dir[w]
        dtmp *= -1
        dist = min(dist, np.amin(dtmp))

    assert dist != np.inf

    return dist

def rwalk(id, nmodels, burn, thin, S, q, rng, selection=['uniform']):

    #-----------------------------------------------------------------------
    # Find a point that is completely inside the simplex
    #-----------------------------------------------------------------------

    if id == 0: Log('Finding first inner point')
    ip = S.inner_point(S.nsteps_inner,id)

    ok,fail_count = S.in_simplex(ip, eq_tol=1e-12, tol=0, verbose=0)
    assert ok

    from copy import deepcopy
    p0,p,A,b,Apinv = map(deepcopy, S.dist_eqs + [S.A, S.b, S.Apinv])
    nVars = S.nVars

    #ip = np.ones(nVars, dtype=np.float64)

    i,j=0,0
    while i < nmodels:

        if j < burn:
            j += 1
        else:
            i += 1

        #t0 = time.perf_counter()
        if selection == 'uniform':
            rs = rng.random(size=thin)
        elif selection == 'beta':
            rs = rng.beta(2,2,size=thin)

        for k in range(thin):
            d0 = new_dir(
                    ip, 
                    lambda p: project(p, A,b,Apinv),
                    lambda  : rng.normal(0,1,nVars))

            # distances from the inner point ip to the edge of the polytope
            tmax1 = distance_to_plane_faster(ip, -d0, p0,p)
            tmax2 = distance_to_plane_faster(ip, +d0, p0,p)

            # We derive the active code below from the following:
            #
            #   r0 = ip - tmax1*d0      # the points at the edge
            #   r1 = ip + tmax2*d0      #
            #   ip = r0 + rs[k] * (r1-r0)
            #
            ip -= d0*(tmax1 - (rs[k] * (tmax2 + tmax1)))


        #if id != 0: return
        #t1 = time.perf_counter()
        mod = np.empty(nVars+1, order='C', dtype=np.float64)
        mod[0]  = 0
        mod[1:] = ip

        if i > 0:
            q.put([id,mod,'walk'])
        else:
            if id == 0: 
                #        print(id, t1-t0)
                q.put([id,mod,'burn'])


class Samplex:

    name = 'Hit-and-Run Random Walk'
    ver  = '1.0'

    def __init__(self, **kw):

        ncols              = kw.get('ncols', None)
        nthreads           = kw.get('nthreads', 1)
        rngseed            = kw.get('rngseed',  None)
        self.burnin_factor = kw.get('burnin factor', 1)
        self.thin_len      = kw.get('thin', 1000)
        self.nsteps_inner  = kw.get('nsteps inner pt', 10)
        self.selection     = kw.get('selection', 'uniform')

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
        #self.ineqs = []

        self.avg0 = None


    def start(self):

        assert self.nVars is not None

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
            print(m.shape)
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

        section( '%s  v%s' % (self.name, self.ver) )

        Log( "%8i equations" % len(self.eq_list) )
        Log( "%8i variables" % (self.nVars) )

        Log( "%6s %6s %6s\n%6i %6i %6i" 
            % (">=", "<=", "=", self.geq_count, self.leq_count, self.eq_count) )


        if nsolutions == 0: return

        assert nsolutions is not None

        dim = self.nVars
        dof = dim - self.eq_count

        burnin_len  = max(10, int(self.burnin_factor * dof))

        nmodels = nsolutions
        nthreads = self.nthreads

        self.dim        = dim
        self.dof        = dof
        self.burnin_len = burnin_len

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

        self.dist_eqs = [self.dist_eqs[:,0], self.dist_eqs[:,1:].T]

        #Log( 'Using lpsolve %s' % lpsolve('lp_solve_version') )
        Log( "random seed      = %i"        % self.random_seed          )
        Log( "n threads        = %i"        % self.nthreads             )
        Log( "model sampling   = %s distribution" % self.selection      )
        Log( "dof              = %i"        % self.dof                  )
        Log( "inner pt steps   = %i"        % self.nsteps_inner         )
        Log( "thinning         = %i models" % self.thin_len             )
        Log( "burn-in length   = %i models" % self.burnin_len           )
        Log( "requested models = %i" % nmodels                   )
        Log( "total models     = %i" % (nmodels+self.burnin_len)   )
        Log( "total MC steps   = %i" % ((nmodels+self.burnin_len)*self.thin_len) )

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
            self.b = None
            self.Apinv = None

        #-----------------------------------------------------------------------
        # Now we can start the random walk
        #-----------------------------------------------------------------------

        burn, thin = self.burnin_len, self.thin_len
        selection = self.selection

        q = MP.Manager().Queue()

        #-----------------------------------------------------------------------
        # Launch the threads
        #-----------------------------------------------------------------------
        threads = []
        models_per_thread = nmodels // nthreads
        models_under      = nmodels - nthreads*models_per_thread
        id,N = 0,0

        ss = SeedSequence(self.random_seed)
        child_seeds = ss.spawn(nthreads)

        while id < nthreads and N < nmodels:
            n = models_per_thread
            if id < models_under:
                n += 1
            assert n > 0


            thr = MP.Process(target=rwalk, 
                             args=(id, n, burn, thin, self, q, default_rng(child_seeds[id]), selection))
            threads.append(thr)
            N += n
            id += 1

        assert N == nmodels

        for thr in threads:
            thr.daemon=True
            thr.start()

        #Log('Burning %i models in each thread. Thinning by %i models.' % (burn, thin))

        def status_walk(j,nmodels,t):
            if mt > 0:
                Log( '%3i%%] %6i/%i models remaining.  %.3f s/model.  ETA %s.' % ((j*100)//nmodels, nmodels-j, nmodels, mt, timestr(mt*(nmodels-j))), overwritable=False)
            else:
                Log( '%3i%%] %6i/%i models remaining.' % ((j*100)//nmodels, nmodels-j, nmodels), overwritable=False)

        def status_burn(j,nburn,nmodels,t):
            if mt > 0:
                Log( '%3i%% - (%3i%% burn)] %6i/%i models remaining.  %.3f s/model.  ETA %s.' % 
                        ((j*100)//nmodels, (j*100)//nburn, nmodels-j, nmodels, mt, timestr(mt*(nmodels-j))), overwritable=False)
            else:
                Log( '%3i%% - (%3i%% burn)] %6i/%i models remaining.' % 
                        ((j*100)//nmodels, (j*100)//nburn, nmodels-j, nmodels), overwritable=False)


        #-----------------------------------------------------------------------
        # Actual random walk
        #-----------------------------------------------------------------------
 
        time_get_models = dict(start=time.perf_counter(), end=None)
        t0,t1 = 0,0
        for i in range(nmodels+burn):
            k,vec,phase = q.get()

            if i in [0,burn]:
                nt,mt,dt,t0 = 0,0,0,time.perf_counter()

            t1 = time.perf_counter()
            dt = t1-t0
            nt += 1

            if dt >= 5:
                mt = dt/nt
                nt = 0
                t0 = time.perf_counter()
                if i >  burn: status_walk(i,nmodels+burn,mt)
                if i <= burn: status_burn(i,burn,nmodels+burn,mt)

            if phase == 'burn': continue

            yield vec

        time_get_models['end'] = time.perf_counter()

        if nt:
            mt = dt/nt
            nt = 0
            t0 = time.perf_counter()
            status_walk(i,nmodels+burn,mt)

        for thr in threads:
            thr.terminate()

#       #-----------------------------------------------------------------------
#       # Stop the threads and get their running times.
#       #-----------------------------------------------------------------------
#       time_threads = []
#       for thr,cmdq,ackq in threads:
#           cmdq.put(['STOP'])
#           m,t = ackq.get()
#           assert m == 'TIME'
#           time_threads.append(t)
#           #thr.terminate()

        #max_time_threads = np.amax(time_threads) if time_threads else 0
        #avg_time_threads = np.mean(time_threads) if time_threads else 0

        dt = time_get_models['end'] - time_get_models['start']
        Log( 'Modeling complete.' )
        Log( 'Time for Hit and Run sampling : %s' % timestr(dt) )
        Log( 'Average time per model        : %s' % timestr(dt/(nmodels+burn)) )

        #Log( '-'*80 )
        #Log( 'SAMPLEX TIMINGS' )
        #Log( '-'*80 )
        #Log( 'Initial inner point    %.2fs' % (time_end_inner_point - time_begin_inner_point) )
        #Log( 'Estimate eigenvectors  %.2fs' % (time_end_est_eigenvectors - time_begin_est_eigenvectors) )
        #Log( 'Burn-in                %.2fs' % (time_end_burnin - time_begin_burnin) )
        #Log( 'Modeling               %.2fs' % (time_end_get_models - time_begin_get_models) )
        #Log( 'Max/Avg thread time    %.2fs %.2fs' % (max_time_threads, avg_time_threads) )
        #Log( 'Total wall-clock time  %.2fs' % (time_end_next - time_begin_next) )
        #Log( '-'*80 )

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
                    if verbose: Log( 'F> %i %e %e' % (i,a, -tol) )
                    bad.append([i,a])
            elif c == 'leq':
                a_min = min(a_min, a)
                if a > tol: 
                    if verbose: Log( 'F< %i %e %e' % (i,a, tol) )
                    bad.append([i,a])
            elif c == 'eq':
                if np.abs(a) > eq_tol: 
                    if verbose: Log( 'F= %i %e %e' %(i,a, (1 - np.abs(e[0]/a0))) )
                    bad.append([i,a])

            if verbose >= 2: print("TT", c, a)
              
        if verbose >= 1:
            Log( 'Smallest a was %e' % (a_min,) )

        #print 'T '
        return not bad, bad

    def distance_to_plane_faster(self,pt,dir):
        ''' dir should be a normalized vector '''

        dist = np.inf

        #p = self.dist_eqs
        p0,p = self.dist_eqs
        a = np.dot(dir, p)
        w = np.nonzero(a > 0)[0]

        if len(w):
            # dtmp = -(p[:,0] + np.dot(pt, p[:,1:].T)) / a
            # ==>
            dtmp = np.dot(pt, p)
            dtmp += p0
            dtmp *= -1
            dtmp /= a
            dist = np.amin(dtmp[w])

        # check implicit >= 0 contraints
        #a = -dir 
        #w = a > 0
        w = np.nonzero(dir < 0)[0]
        if len(w):
            dtmp = pt[w] / dir[w]
            dtmp *= -1
            dist = min(dist, np.amin(dtmp))

        assert dist != np.inf

        return dist

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

    def project(self,x):
        if self.Apinv is not None:
            q = np.dot(self.A, x)
            q += self.b
            x -= np.dot(self.Apinv, q)

    def new_dir(self, ip, u):
        d = norm(u) #np.sqrt(np.dot(u,u))
        #d = np.sqrt(np.dot(u,u))
        p = u/d
        p += ip
        self.project(p)
        p -= ip
        p /= norm(p)
        return p


    def new_dir_linprog(self):

        pts = []

        for i in range(2):
            if i % 2 == 0:
                c = np.random.random(self.nVars) 
            else:
                c *= -1

            res = linprog(c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, method='interior-point', options=dict(disp=False,tol=1e-2))

            if not res.success:
                if   res.status == 1: # Iteration limit reached
                    pass
                elif res.status == 2: # Problem appears to be infeasible.
                    raise SamplexNoSolutionError()
                elif res.status == 3: # Problem appears to be unbounded.
                    raise SamplexUnboundedError()
                elif res.status == 4: # Numerical difficulties encountered.
                    print(res)
                    raise SamplexNoSolutionError()
                else:
                    Log( result )
                    raise SamplexUnexpectedError("unknown pivot result %i from linear solver." % result)
            
            newp = res.x.copy()
            self.project(newp)
            pts.append(newp)
    
        # get a unit vector point along the span direction
        sd = pts[0] - pts[1]
        #print (sd)
        d0 = sd / norm(sd) #np.sqrt(np.dot(sd,sd))

        return d0

    def inner_point(self, nsamples, id=None):

        #lp = lpsolve('make_lp', 0, self.nVars+1) # +1 for variable used to find the first inner point
        #lpsolve('set_epsb', lp, 1e-14)
        #lpsolve('set_epsd', lp, 1e-14)
        #lpsolve('set_epsint', lp, 1e-14)
        #lpsolve('set_epsel', lp, 1e-8)
        #lpsolve('set_verbose', lp, FULL)
        #lpsolve('set_sense', lp, False)

        # Objective function coefficients
        # +1 for variable used to find the first inner point
        linprog_opts = dict(disp=False,tol=1e-6, sparse=True)

        if not linprog_opts['sparse']:
            A_ub, A_eq = [],[]
            b_ub, b_eq = [],[]
            for eq,a in self.eq_list:
                ls = np.zeros(self.nVars+1)
                # const + (...) <= 0  leq
                # const + (...) >= 0  geq
                # const + (...) == 0   eq
                l0, ls[:-1] = -a[0], a[1:]

                if eq == 'eq': 
                    ls[-1] = 0
                    A_eq.append(ls)
                    b_eq.append(l0)
                if eq == 'leq': 
                    ls[-1] = 1
                    A_ub.append(ls)
                    b_ub.append(l0)
                if eq == 'geq':
                    ls[-1] = 1
                    ls *= -1
                    l0 *= -1
                    A_ub.append(ls)
                    b_ub.append(l0)

            if not A_ub: A_ub = None
            if not b_ub: b_ub = None
            if not A_eq: A_eq = None
            if not b_eq: b_eq = None

        else:

            from scipy.sparse import csr_matrix
            from scipy.sparse import lil_matrix as sparse_matrix

            last = self.nVars
            A_eq, b_eq = None,None
            A_ub, b_ub = None,None
            for i,[eq,a] in enumerate(self.eq_list):
                #ls = np.zeros(self.nVars+1)
                # const + (...) <= 0  leq
                # const + (...) >= 0  geq
                # const + (...) == 0   eq

                if A_eq is None and eq == 'eq':
                    A_eq, b_eq = sparse_matrix((len(self.eq_list), self.nVars+1)), np.zeros(len(self.eq_list), dtype=np.float64)
                if A_ub is None and eq in ['leq', 'geq']:
                    A_ub, b_ub = sparse_matrix((len(self.eq_list), self.nVars+1)), np.zeros(len(self.eq_list), dtype=np.float64)


                if eq == 'eq':
                    for j in range(self.nVars): 
                        if a[1+j] != 0: A_eq[i,j] = a[1+j]
                    #A_eq[i,self.nVars] = 0 # Commented because the matrix is sparse and shouldn't explicitly store zeros
                    b_eq[i] = -a[0]

                if eq == 'leq': 
                    for j in range(self.nVars): 
                        if a[1+j] != 0: A_ub[i,j] = a[1+j]
                    A_ub[i,self.nVars] = 1
                    b_ub[i] = -a[0]

                if eq == 'geq': 
                    for j in range(self.nVars): 
                        if a[1+j] != 0: A_ub[i,j] = -a[1+j]
                    A_ub[i,self.nVars] = -1
                    b_ub[i] = a[0]


        #c = np.zeros(self.nVars+1) 
        #c[-1] = objval

        newp = np.zeros(self.nVars, dtype=np.float64)
        rs = np.random.random(nsamples)

        for i in range(nsamples):
            if id == 0: 
                if i == 0: Log('...step %3i/%i' % (i+1,nsamples))
                if i >  0: Log('...step %3i/%i.  ETA %s' % (i+1,nsamples, timestr(eta)))
            #c = np.random.rand(self.nVars+1) - 0.5
            #if i % 2 == 0:
            c = np.zeros(self.nVars+1, dtype=np.float64)
            c[-1] = rs[i]
                #c[-1] = 200
            #else:
                #c *= -1

            #res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options=dict(disp=True,autoscale=True,tol=1e-12))
            t0 = time.perf_counter()
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options=linprog_opts)
            t1 = time.perf_counter()

            if not res.success:
                if   res.status == 1: # Iteration limit reached
                    pass
                elif res.status == 2: # Problem appears to be infeasible.
                    raise SamplexNoSolutionError()
                elif res.status == 3: # Problem appears to be unbounded.
                    raise SamplexUnboundedError()
                elif res.status == 4: # Numerical difficulties encountered.
                    raise SamplexNumericalError()
                else:
                    Log( res )
                    raise SamplexUnexpectedError()

            if id == 0: 
                eta = (t1-t0) * (nsamples-i-1)
            newp += res.x[:-1]
    
        newp /= nsamples
        #assert len(newp) == len(res.x)
        #newp[:] = res.x[:-1]
        self.project(newp)
        return newp

    #   objv  = np.array(lpsolve('get_objective', lp))
    #   v1    = np.array(lpsolve('get_variables', lp)[0])
    #   assert len(v1) == lpsolve('get_Norig_columns', lp)
    #   assert len(v1) == self.nVars+1
    #   del lp

    #   v1 = v1[:-1] # Remove the temporary variable that tracks the distance from the simplex boundary
    #   v1[np.abs(v1) < 1e-14] = 0
    #   assert np.all(v1 >= 0), v1[v1 < 0]

    #   ok,fail_count = self.in_simplex(v1, eq_tol=1e-12, tol=0, verbose=1)
    #   ok,fail_count = self.in_simplex(v1, eq_tol=1e-12, tol=-1e-13, verbose=1)
    #   assert ok, len(fail_count)
    #   newp[:] = v1
    #   self.project(newp)
    #   ok,fail_count = self.in_simplex(newp, eq_tol=1e-12, tol=0, verbose=1)
    #   ok,fail_count = self.in_simplex(newp, eq_tol=1e-12, tol=-1e-5, verbose=1)

    #=========================================================================

    TT = 1e-14
    def eq(self, a):
        if self.nVars is None: self.nVars = len(a)-1
        self.eq_count += 1
        self.eq_list.append(['eq', a])

    def geq(self, a):
        if self.nVars is None: self.nVars = len(a)-1
        return self.leq(-a)

    def leq(self, a):
        if self.nVars is None: self.nVars = len(a)-1
        self.leq_count += 1

        #self.ineqs.append([-a[1:], GE, a[0]])
        self.eq_list.append(['leq', a])

