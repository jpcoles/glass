
import sys
import numpy as np

import pylab as pl
import matplotlib.cm as cm  

from numpy import amin, amax, diff, argsort, abs, array, sum, \
                  mat, eye, asarray, matrix, empty_like, zeros, \
                  sort, any, sqrt, dot, ceil, arctan2, pi, mean, identity, average
from random import random
from glass.basis.pixels.potential import poten, poten_dx, poten_dy, poten_dxdx, poten_dydy, poten_dxdy, poten_dydx
from scipy.ndimage.filters import correlate1d
from scipy.misc import central_diff_weights
from scipy.optimize import fsolve, fmin

from glass.scales import convert

from glass.log import log as Log

fig = None

def raytrace(model, nimgs=None, ipeps=None, speps=None, initial_guess=None, verbose=False, viz=False):
    """Find the positions of images by raytracing back to the source.

        ipeps - Radius on the image plane to consider two image as one.
        speps - Radius on the source plane to determine if a pixel maps near to the source
    """
                
    global fig

    obj,ps,src_index = model
#   if len(model) == 2:
#       [obj,ps], src_index = model
#   else:
#       obj_index, src_index = model[1:]
#       obj,ps = model[0]['obj,data'][obj_index]

    ploc    = obj.basis.ploc
    src     = ps['src'][src_index]
    zcap    = obj.sources[src_index].zcap
    srcdiff = None
    #zcap=1

    if ipeps is None:
        #ipeps = 2 * obj.basis.top_level_cell_size
        ipeps = 0.01 * obj.basis.mapextent

    if speps is None:
        speps = obj.basis.top_level_cell_size #/ sqrt(2)
        #speps = 0.01 * obj.basis.mapextent

    #---------------------------------------------------------------------------
    # (1) Make an initial guess where the images are. 
    #
    # srcdiff is a matrix giving the distance between the src and the point on
    # the source plane where each pixel maps back to. Taking the n pixels with
    # the smallest of those differences gives a good estimate for the image
    # positions.
    #---------------------------------------------------------------------------

    if not initial_guess:
        srcdiff = obj.basis.srcdiff(ps, src_index)
        initial_guess = []
        asort = argsort(srcdiff)

        for j in asort:
            if srcdiff[j] > speps: break
            initial_guess.append(ploc[j])

#   if not initial_guess:
#       initial_guess = []

#       for i in range(20):
#           r = obj.basis.mapextent * random()
#           t  = 2 * np.pi * random()
#           sx = r * np.cos(t)
#           sy = r * np.sin(t)
#           initial_guess.append(complex(sx,sy))
    



#-------------------------------------------------------------------------------

#           for ii in initial_guess:
#               if abs(ploc[j] - ii) <= eps: break
#           else:

#        if len(initial_guess) >= nimgs: break

#       raw_input()

#   #figure(f.number)
#   figure()

#       if not any(abs(ploc[j] - ploc[asort[:i]]) <= eps):
#           srcdiff[j] = 10
#           initial_guess.append(ploc[j])


#-------------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # (2) Minimize the lens equation beginning from each of the initial guesses.
    # Only those images that truly satisfy the equation are accepted.
    #---------------------------------------------------------------------------

    initial_guess.append(src)
    if verbose: print('Initial guesses', initial_guess)

    def lenseq(theta0):
        theta = complex(*theta0)
        r = src - theta + obj.basis.deflect(theta, ps) / zcap
        return r.real, r.imag

    def lenseq_prime(theta0):
        theta = complex(*theta0)
        dist  = theta - obj.basis.ploc
        dxdx = -1 + dot(ps['kappa'], (poten_dxdx(dist,obj.basis.cell_size))) / zcap
        dydy = -1 + dot(ps['kappa'], (poten_dydy(dist,obj.basis.cell_size))) / zcap
        dxdy = -1 + dot(ps['kappa'], (poten_dxdy(dist,obj.basis.cell_size))) / zcap
        dydx = -1 + dot(ps['kappa'], (poten_dydx(dist,obj.basis.cell_size))) / zcap
        return [ [dxdx, dydx], [dxdy, dydy] ]

    xs = []
    #if obj.shear: s1,s2 = ps['shear']
    for img in initial_guess:
        #x,_,ier,mesg = fsolve(lenseq, [img.real,img.imag], fprime=lenseq_prime, full_output=True) #, xtol=1e-12)
        x,_,ier,mesg = fsolve(lenseq, [img.real,img.imag], full_output=True, xtol=1e-12)
        #x = fmin(lenseq, [img.real,img.imag], full_output=False, disp=False, xtol=1e-10, ftol=1e-10)

        if not ier: 
            print(mesg)
            continue

        r = complex(*x)

        # if an initial guess was poor then the minimum will not be near zero.
        # Only accept solutions that are very close to zero.
        leq = abs(complex(*lenseq(x)))
        if leq < 2e-10:
            #print leq
            xs.append([img, r, leq])
        else:
            #print 'Image at %s rejected. %e' % (r, leq)
            pass


    #---------------------------------------------------------------------------
    # (3) Sort by how well we were able to minimize each function.
    #---------------------------------------------------------------------------

    xs.sort(lambda x,y: -1 if x[2] < y[2] else 1 if x[2] > y[2] else 0)

    #---------------------------------------------------------------------------
    # (4) Only accept a solution if it is distinct from previous solutions.
    #---------------------------------------------------------------------------

    imgs0 = []
    for img,r0,t0 in xs:

        for r,t in imgs0:
            if abs(r0-r) < ipeps: break
        else:
            tau  = abs(r0-src)**2 / 2
            tau *= zcap
            tau -= dot(ps['kappa'], poten(r0 - obj.basis.ploc, obj.basis.cell_size, obj.basis.maprad))
            tau -= np.sum( [ ps[e.name] * e.poten(r0).T for e in obj.extra_potentials ] )
            #print tau
            #print '!!', poten(i - obj.basis.ploc, obj.basis.cell_size)[0]
            #print '!!', dot(ps['kappa'], poten(complex(1,0) - obj.basis.ploc, obj.basis.cell_size))
            imgs0.append([r0,tau])

    if viz:
        if fig == None:
            fig = pl.figure()

        if srcdiff is None:
            srcdiff = obj.basis.srcdiff(ps, src_index)

        #print obj.sources
        reorder = empty_like(srcdiff)
        reorder.put(obj.basis.pmap, srcdiff)
        sd = zeros((2*obj.basis.pixrad+1)**2)
        sd[obj.basis.insideL] = reorder
        #sd[:len(srcdiff)] = srcdiff #reorder
        sd = sd.reshape((2*obj.basis.pixrad+1,2*obj.basis.pixrad+1))
        R = obj.basis.mapextent
        kw = {'extent': [-R,R,-R,R],
              'interpolation': 'nearest',
              'aspect': 'equal',
              'origin': 'upper',
              #'cmap': cm.terrain,
              'fignum': False,
              #'vmin': -1,
              #'vmax':  1
              }
        pl.cla()
        pl.matshow(sd, **kw)
        #pl.colorbar()
#       print '??'
#       for i in initial_guess: print i
#       print '??'
        #arrival_plot({'obj,data':[[obj,ps]]}, src_index=src_index, clevels=150)
        xs = [r.real for r in initial_guess]
        ys = [r.imag for r in initial_guess]
        pl.scatter(xs,ys)

#       print '**'
#       for i in obj.sources[src_index].images: print i
#       print '**'

#       xs = [r.pos.real for r in obj.sources[src_index].images]
#       ys = [r.pos.imag for r in obj.sources[src_index].images]
#       pl.scatter(xs,ys, color='r', s=40)

        xs = [r.real for r,tau in imgs0]
        ys = [r.imag for r,tau in imgs0]
        pl.scatter(xs,ys, color='g')
        pl.draw()
        print('*'*80)
        print('PRESS ANY KEY TO CONTINUE')
        print('*'*80)
        input()


    #---------------------------------------------------------------------------
    # (5) Determine magnification information
    #---------------------------------------------------------------------------
    from scipy.linalg import det

    imgs = []
    for img in imgs0:
        #print 'tau', tau
        r, tau = img
        theta = arctan2(r.imag, r.real) * 180/pi
        A = zcap*identity(2) - obj.basis.magnification(r, theta, ps)

        detA = det(A)
        trA  = A.trace()
        mu   = 1. / detA # magnification factor

        parity = ['sad', 'sad', 'max', 'min'][(detA > 0)*2 + (trA > 0)]

        print(mu)
        imgs.append(img + [ [mu, A], parity ])

    #Mavg = average(map(lambda x: (x[3] != 'max') * x[2][3], imgs))

    #imgs = filter(lambda x: abs(x[2][3]) > Mavg*0.8, imgs)

    #print imgs

    #---------------------------------------------------------------------------
    # (6) Sort by arrival time
    #---------------------------------------------------------------------------
    imgs.sort(lambda x,y: -1 if x[1] < y[1] else 1 if x[1] > y[1] else 0)

#   if fig == None:
#       fig = figure()

#   f = pl.gcf()
##  figure() #fig.number)
##  reorder = empty_like(srcdiff)
##  reorder.put(obj.basis.pmap, srcdiff)
##  sd = zeros((2*obj.basis.pixrad+1)**2)
##  sd[obj.basis.insideL] = reorder
##  #sd[:len(srcdiff)] = srcdiff #reorder
##  sd = sd.reshape((2*obj.basis.pixrad+1,2*obj.basis.pixrad+1))
##  R = obj.basis.mapextent
##  kw = {'extent': [-R,R,-R,R],
##        'interpolation': 'nearest',
##        'aspect': 'equal',
##        'origin': 'upper',
##        'cmap': cm.terrain,
##        'fignum': False,
##        'vmin': 0,
##        'vmax': R}
##  matshow(sd, **kw) #, fignum=fig.number)
##  kw.pop('cmap')
##  over(contour, sd, 100, extend='both', colors='w', alpha=1.0, **kw)

#   #figure(f.number)
#   figure()

    #print imgs

    return imgs

def check_model_magnifications(model, **kw_orig):

#   assert len(model) == 2
#   if type(model[1]) == type(0):
#       assert 0, 'Deprecated feature. If absolutely necessary, remove this assert.'
#       pass
#       obj,ps = model[0]['obj,data'][model[1]]
#   else:
#       obj,ps = model

    #obj,ps = model['obj,data'][obj_index]

    #for obj,ps in model['obj,data']:
    obj,ps = model
    for src_index,_ in enumerate(obj.sources):
        kw = kw_orig.copy()
        #kw['viz'] = True

        rt_imgs  = raytrace([obj,ps,src_index],**kw)
        Mrt = np.sum( [ abs(mu) for _,_,[mu,_],p in rt_imgs if p != 'max' ])

        kw['initial_guess'] = [x.pos for x in obj.sources[src_index].images]
        kw['viz'] = False

        obj_imgs = raytrace([obj,ps,src_index],**kw)
        Mobj = np.sum( [ abs(mu) for _,_,[mu,_],p in obj_imgs if p != 'max' ])

        print(abs(Mrt-Mobj) / Mobj)

        bad = abs(Mrt-Mobj) / Mobj > 0.05

        if bad:
            print(src_index)
            print('Rejected:', abs(Mrt-Mobj), Mobj)
            for r,t,[mu,_],p in rt_imgs: print(r,t,mu,p)
            print('-----')
            for r,t,[mu,_],p in obj_imgs: print(r,t,mu,p)
            del kw['initial_guess']
            #kw['viz'] = True
            #obj_imgs = raytrace([obj,ps,src_index],**kw)
            return False

    return True
    #return abs(Mrt-Mobj) / Mobj < 0.05, Mrt, Mobj

def magnification_filter(model):
    return check_model_magnifications(model)

def raytraceX(obj, ps, sys_index, nimgs=None, eps=None):

    #---------------------------------------------------------------------------
    # Find the derivative of the arrival time surface.
    #---------------------------------------------------------------------------
    arrival = obj.basis.arrival_grid(ps)[sys_index]

    w = central_diff_weights(3)
    d = abs(correlate1d(arrival, w, axis=0, mode='constant')) \
      + abs(correlate1d(arrival, w, axis=1, mode='constant'))
    d = d[1:-1,1:-1]

    pl.matshow(d)

    xy = obj.basis.refined_xy_grid(ps)[1:-1,1:-1]

    # Create flattened *views*
    xy     = xy.ravel()
    dravel = d.ravel()

    imgs = []
    offs = []
    print('searching...')
    for i in argsort(dravel):

        if nimgs == len(imgs): break

        new_image = True
        for img in imgs:
            if abs(img-xy[i]) <= eps: 
                new_image = False

        if new_image:
            imgs.append(xy[i])
            offs.append(i)

    #---------------------------------------------------------------------------
    # Print the output
    #---------------------------------------------------------------------------

    if imgs:
        #print imgs
        #if len(imgs) % 2 == 1: imgs = imgs[:-1]  # XXX: Correct?
        imgs = array(imgs)

        g0 = array(arrival[1:-1,1:-1], copy=True)
        g0ravel = g0.ravel()
        times = g0ravel[offs]
        order = argsort(times)
    else:
        order = []

    return [(times[i], imgs[i]) for i in order]

def observables(model, obj_index, src_index, seq):
    obj,ps = model['obj,data'][obj_index]

    if not seq: return

    imglist = [[seq[0][0], seq[0][3], None]]

    _,prev,_,_ = seq[0]
    for img,t,_,parity in seq[1:]:
        #print parity, t-prev, obj.z, ps['nu']
        t0 = convert('arcsec^2 to days', t-prev, obj.dL, obj.z, ps['nu'])
        imglist.append([img, parity,t0])
        prev = t

    return imglist

def write_glass_code(model, obj_index, src_index, seq, simple=False):

    obj,ps = model['obj,data'][obj_index]

    if not seq: return

    letters = [ x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    
    while len(seq) > len(letters):
        letters += [ x+x[0] for x in letters[-26:] ]

    obs = observables(model, obj_index, src_index, seq)

    #def img2str(img, time_delay, l, parity):
        #return "['%s', (% 9.5f,% 9.5f), '%s', %.4f]" % (l, img.real, img.imag, parity, time_delay)

    def img2str(a):
        return "%s = (% 9.15f,% 9.15f)" % (a[0],a[1][0].real,a[1][0].imag)

    def img2str2(a):
        if a[1][2] is None:
            return "%s, '%s'" % (a[0], a[1][1])
        else:
            return "%s, '%s', %.15f" % (a[0], a[1][1], a[1][2])
        
    print("\n".join(map(img2str, list(zip(letters, obs)))))
    print(("source(%0.2f, " % obj.sources[src_index].z) + ', '.join(map(img2str2, list(zip(letters, obs)))) + ")")

    return

def write_code(model, obj_index, src_index, seq, simple=False):
    code = gen_code(model, obj_index, src_index, seq, simple=False)
    print("[" + ",\n ".join(code) + "]")

def gen_code(model, obj_index, src_index, seq, simple=False):

    obj,ps = model['obj,data'][obj_index]

    if not seq: return

    letters = [ x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    
    while len(seq) > len(letters):
        letters += [ x+x[0] for x in letters[-26:] ]

    obs = observables(model, obj_index, src_index, seq)

    #def img2str(img, time_delay, l, parity):
        #return "['%s', (% 9.5f,% 9.5f), '%s', %.4f]" % (l, img.real, img.imag, parity, time_delay)

    def img2str(a):
        if a[1][2] is None:
            return "['%s', (% 9.15f,% 9.15f), '%s']" % (a[0],a[1][0].real,a[1][0].imag, a[1][1])
        else:
            return "['%s', (% 9.15f,% 9.15f), '%s', %.4f]" % (a[0],a[1][0].real,a[1][0].imag, a[1][1], a[1][2])
        
    return list(map(img2str, list(zip(letters, obs))))
        
    imglist = ["['%s', (% 9.5f,% 9.5f), '%s']" % (letters[0], seq[0][0].real, seq[0][0].imag,seq[0][3])]
    prev = seq[0][1]
    for [img,t,m,parity],l in zip(seq[1:], letters[1:]):
        t0 = convert('arcsec^2 to days', t-prev, obj.z, ps['nu'])
        imglist.append(img2str(img,t0,l,parity))
        prev = t

    print("%.2f, [%.4f, %.4f]," % (obj.sources[src_index].z, ps['src'][src_index].real, ps['src'][src_index].imag))
    print("[" + ",\n ".join(imglist) + "]")

    #---------------------------------------------------------------------------
    # Old stuff.
    #---------------------------------------------------------------------------
    if 0:
        if not simple:
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for [img,t,_,_], l in zip(seq, letters):
                sys.stdout.write('%s = %-.4f, %-.4f\n' % (l, img.real, img.imag))

            sys.stdout.write("source :: [%.2f, [%.4f, %.4f]]\n" % (obj.sources[src_index].z, ps['src'][src_index].real, ps['src'][src_index].imag))

            sys.stdout.write("source(%.2f,A,'min'" % obj.sources[src_index].z)
            prev = seq[0][1]
            for [img,t],l in zip(seq[1:], letters[1:]):
                print('@', t-prev, ps['1/H0'], obj.basis.top_level_cell_size**2)
                t0 = (t-prev) * ps['1/H0'] * obj.scales['time'] # / obj.basis.top_level_cell_size**2
                #t0 = time_to_physical(obj, t-prev) * ps['1/H0'] / obj.basis.top_level_cell_size**2
                if -1e-4 < t0 < 1e-4:
                    sys.stdout.write(", %s,'',%.4e" % (l, t0))
                else:
                    sys.stdout.write(", %s,'',%.4f" % (l, t0))
                prev = t
            sys.stdout.write(')\n')

        else:

            sys.stdout.write("source(%.2f, (%-.4g,%-.4g),'min'" % (
                obj.sources[src_index].z, seq[0][0].real, seq[0][0].imag))

            p = ''
            prev = seq[0][1]
            for i,[img,t] in enumerate(seq[1:]):

                if len(seq) in [4,5]:
                    if i <  1: p = 'min'
                    if i >= 2: p = 'sad'
                    if i >= 3: p = 'max'

                t0 = time_to_physical(obj, t-prev) * ps['1/H0']
                sys.stdout.write(", (%-.4g,%-.4g),'%s',%.4g" % (img.real, img.imag, p,t0))
                prev = t
            sys.stdout.write(')\n')

