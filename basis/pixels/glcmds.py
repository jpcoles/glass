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

        for obj in env().objects:
            print >>f, '''\
object %(objname)s
pixrad %(pixrad)i
maprad %(maprad)f
zlens %(zlens).2f
models %(models)i
g %(g).2f
cosm %(om).2f %(ol).2f''' % { \
            'objname': obj0.name,
            'pixrad':obj0.basis.pixrad, 
            'maprad':obj0.basis.maprad,
            'zlens': obj0.z,
            'models': len(env().models),
            'g': convert('nu to H0^-1 in Gyr', env().nu),
            'om':env().omega_matter, 
            'ol':env().omega_lambda,
             }

            for src in obj0.sources:
                print >>f, 'multi %i %.2f' % (len(src.images), src.z)
                for img in src.images:
                    print >>f, '% 12.12e % 12.12e %i' % (img.pos.real, img.pos.imag, img.parity+1)

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

