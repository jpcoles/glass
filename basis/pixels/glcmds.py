from __future__ import division
from environment import env
from solvers.samplex.samplex import Samplex
from basis import PixelBasis as basis_class
from numpy import load, mean, pi, radians
from scales import convert

def minsteep(a):                assert False, "minsteep not supported. Use steepness()."
def maxsteep(a):                assert False, "maxsteep not supported. Use steepness()."

def _foo(options):
   return Samplex(**options)
   #return Samplex(nvars, nthreads=env().ncpus) 

def globject(name):
    co = env().new_object(name)
    co.basis = basis_class()
    #env().model_gen_factory = _foo
    return co

def pixrad(r):
    env().current_object().basis.pixrad = r

def priors(*ps):
    env().current_object().basis.prior_list = ps

def subdiv(n):
    n = int(n)
    assert (n%2==1), "subdiv: n must be odd"
    env().current_object().basis.subdivision = n

def hires(r, refine=1):
    assert r > 0 and refine>=3 and refine%2==1, 'hires: Minimum refinement value is 3. Must be odd too.'
    env().current_object().basis.hiresR       = r
    env().current_object().basis.hires_levels = refine
    
def smooth(factor=2, include_central_pixel=None):
    o = env().current_object()
    #o.prior_options['smoothness'] = {}
    o.prior_options['smoothness']['factor'] = factor
    if include_central_pixel is not None:
        o.prior_options['smoothness']['include_central_pixel'] = include_central_pixel

def steepness(lb, ub):
    o = env().current_object()
    o.prior_options['steepness'] = [lb, ub]

def kann(theta):
    o = env().current_object()
    o.prior_options['annular_density'] = theta

def dgcone(theta):
    assert (0 < theta <= 90), "dgcone: need 0 < theta <= 90"
    o = env().current_object()
    o.prior_options['gradient'] = (90-theta) * pi/180

def local_gradient(theta=None, L=None):
    o = env().current_object()

    if theta is not None: 
        assert (0 < theta <= 90), "local_gradient: need 0 < theta <= 90"
        o.prior_options['J2Gradient']['theta'] = theta

    if L is not None: o.prior_options['J2Gradient']['size']  = L

def minkappa(X,Y,M,H0inv):

    o = env().current_object()
    o.prior_options['minkappa']['grid'] = [X,Y,M]
    o.prior_options['minkappa']['H0inv'] = H0inv
    o.prior_options['minkappa']['nu'] = convert('H0^-1 in Gyr to nu', H0inv)

def minkappa_from_model(model, obj_index):
    #assert len(env().g) == 1

    env().current_object().basis.min_kappa_model = model['obj,data'][obj_index][1]['kappa']

def central_pixel_maximum(M,H0inv):

    o = env().current_object()
    o.prior_options['central_pixel_maximum']['M'] = M
    o.prior_options['central_pixel_maximum']['H0inv'] = H0inv
    o.prior_options['central_pixel_maximum']['nu'] = convert('H0^-1 in Gyr to nu', H0inv)

def savestate_PixeLens(fname):
    obj0 = env().objects[0]
    pr = obj0.basis.pixrad
    w = pr*2 + 1
    pmap = obj0.basis._to_grid(range(1,len(obj0.basis.pmap)+1))

    with open(fname, 'w') as f:
        print >>f, '#BEGIN INPUT'
        print >>f, '''\
object sim
pixrad 10
maprad 50
zlens 0.3
models 600
g 14
cosm %.2f %.2f
minsteep 0
multi 4 1.0
-3.013622e+01  1.360461e+00 1
 2.287190e+01  4.055235e-01 1
 2.153107e+01  8.125506e+00 2
 1.989196e+01 -9.990781e+00 2
multi 4 1.5
-3.209855e+01 -8.905267e+00 1
 2.917315e+01 -1.335862e+01 1
 4.959569e+00 -2.999205e+01 2
 3.193337e+00  2.776801e+01 2
multi 4 1.8
 3.546026e+01 -6.865813e-01 1
-3.386628e+01  1.663847e-01 1
-4.480872e+00 -3.021939e+01 2
-3.476083e+00  3.138001e+01 2
multi 4 2.0
 3.056711e+01 -1.958938e+01 1
-2.916247e+01 -2.045603e+01 1
-4.859099e+00 -3.425608e+01 2
-5.889194e-01  2.922841e+01 2
multi 4 2.5
 3.280539e+01 -1.885161e+01 1
-3.257931e+01 -1.841556e+01 1
-2.726673e+00 -3.591265e+01 2
-1.633044e-01  3.124377e+01 2''' % (env().omega_matter, env().omega_lambda)

        #print >>f, env().input_file
        print >>f, '#END INPUT'

        print >>f, '#BEGIN PMAP'
        print >>f, '%i %i' % (pr,pr)
        for i in range(w):
            for j in range(w):
                print >>f, '%3i' % pmap[i,j], 
            print >>f
        print >>f, '#END PMAP'

        print >>f, '#BEGIN ENSEM'
        for m in env().models:
            print >>f, '#BEGIN MODEL'
            for d in m['sol'][1:]:
                print >>f, '%.15g' % d
            print >>f, '#END MODEL'

        print >>f, '#END ENSEM'

