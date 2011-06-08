from __future__ import division
import sys
from numpy import amin, amax, diff, argsort, abs, array, sum, \
                  mat, eye, asarray, matrix, empty_like, zeros, \
                  sort, any, sqrt, dot, ceil, arctan2, pi, mean, identity
from scales import time_to_physical
from potential import poten
from scipy.linalg import det
from scipy.ndimage.filters import correlate1d
from scipy.misc import central_diff_weights
from scipy.optimize import fsolve, fmin

from pylab import contour, matshow, show, over, figure, gcf
import matplotlib.cm as cm  
from scales import convert
from itertools import izip

fig = None

def raytrace(model, nimgs=None, eps=None, eps2=None, initial_guess=None, verbose=False):
    global fig

    if len(model) == 2:
        [obj,ps], src_index = model
    else:
        obj_index, src_index = model[1:]
        obj,ps = model[0]['obj,data'][obj_index]

    srcdiff = obj.basis.srcdiff(ps, src_index).copy()
    ploc    = obj.basis.ploc
    src     = ps['src'][src_index]
    zcap    = obj.sources[src_index].zcap

    if eps is None:
        eps = 2 * obj.basis.top_level_cell_size
        #eps = sqrt(2.1) * obj.basis.top_level_cell_size

#-------------------------------------------------------------------------------

    #if eps2 is None:
        #eps2 = sqrt(2.1) * obj.basis.top_level_cell_size

    #---------------------------------------------------------------------------
    # (1) Make an initial guess where the images. 
    #
    # srcdiff is a matrix giving the distance between the src and the point on
    # the source plane where each pixel maps back to. Taking the n pixels with
    # the smallest of those differences gives a good estimate for the image
    # positions.
    #---------------------------------------------------------------------------

    if not initial_guess:
        initial_guess = []
        asort = argsort(srcdiff)
        #m = srcdiff[asort[0]] * 10
        m = obj.basis.top_level_cell_size / sqrt(2)
        e = ceil(obj.basis.top_level_cell_size / eps) * obj.basis.top_level_cell_size
        for i,j in enumerate(asort):

            if srcdiff[j] > m: break

            #n = abs(ploc[j] - ploc) <= e
            #n = abs(ploc[j] - ploc) < sqrt(2.1)*obj.basis.top_level_cell_size

            #has_bad_neighbors = any(srcdiff[n] == -1)

            #srcdiff[n] = -2

            #if has_bad_neighbors: continue

            initial_guess.append(ploc[j])

#-------------------------------------------------------------------------------

#           for ii in initial_guess:
#               if abs(ploc[j] - ii) <= eps: break
#           else:

#        if len(initial_guess) >= nimgs: break

#       if fig == None:
#           fig = figure()

#       print obj.sources
#       reorder = empty_like(srcdiff)
#       reorder.put(obj.basis.pmap, srcdiff)
#       sd = zeros((2*obj.basis.pixrad+1)**2)
#       sd[obj.basis.insideL] = reorder
#       #sd[:len(srcdiff)] = srcdiff #reorder
#       sd = sd.reshape((2*obj.basis.pixrad+1,2*obj.basis.pixrad+1))
#       R = obj.basis.mapextent
#       kw = {'extent': [-R,R,-R,R],
#             'interpolation': 'nearest',
#             'aspect': 'equal',
#             'origin': 'upper',
#             #'cmap': cm.terrain,
#             'fignum': False,
#             #'vmin': -1,
#             #'vmax':  1
#             }
#       matshow(sd, **kw)
#       show()

#       raw_input()

#   #figure(f.number)
#   figure()

#       if not any(abs(ploc[j] - ploc[asort[:i]]) <= eps):
#           srcdiff[j] = 10
#           initial_guess.append(ploc[j])


#-------------------------------------------------------------------------------

    def lenseq(theta0):
        theta = complex(*theta0)
        r = src - theta + obj.basis.deflect(theta, ps) / zcap
        #print src, theta, obj.basis.deflect(theta, ps) / zcap, abs(r)
        return abs(r) #[r.real, r.imag]

    initial_guess.append(0j)
    if verbose: print 'Initial guesses', initial_guess


    xs = []
    if obj.shear: s1,s2 = ps['shear']
    for img in initial_guess:
        x = fmin(lenseq, [img.real,img.imag], full_output=False, disp=False) #, warning=True) #, xtol=1e-10)
        #x, infodict, ier, mesg = fsolve(lenseq, [img.real,img.imag], full_output=False, warning=True) #, xtol=1e-10)
        #q = fsolve(lenseq, [img.real,img.imag], full_output=False, warning=True) #, xtol=1e-10)
        #print q
        #x = q[0]
        #print '#', infodict['nfev']
#       if not ier:
#           print 'Near image found at', x, 'but not saving it.'
#           print 'From fsolve:', mesg
#           continue

#        print x
        i = complex(*x)
        #if eps2 and abs(img-i) > eps2: continue
        xs.append([img, i, lenseq(x)])

    #xs.sort(lambda x,y: -1 if abs(x[1]) < abs(y[1]) else 1 if abs(x[1]) > abs(y[1]) else 0)
    xs.sort(lambda x,y: -1 if x[2] < y[2] else 1 if x[2] > y[2] else 0)

#-------------------------------------------------------------------------------

    #print '*'*10, len(xs)
    imgs = []
    for img,i,r in xs:

        #-----------------------------------------------------------------------
        # Only accept if the solution is distinct from previous solutions.
        #-----------------------------------------------------------------------
        for j,t in imgs:
            if abs(i-j) < eps: break
        else:
            tau  = abs(i-src)**2 / 2
            tau *= zcap
            xxx = dot(ps['kappa'], poten(i - obj.basis.ploc, obj.basis.cell_size))
            tau -= xxx
            if obj.shear:
                tau -= s1*obj.shear.poten(1,i) + s2*obj.shear.poten(2,i)
            imgs.append([i,tau])

    M = []
    for img in imgs:
        i, tau = img
        theta = arctan2(i.imag, i.real) * 180/pi
        K = zcap*identity(2) - obj.basis.magnification(i, theta, ps)

        detK = det(K)
        trK  = K.trace()

        parity = ['sad', 'sad', 'max', 'min'][(detK > 0)*2 + (trK > 0)]

        Kinv = K.I
        img.extend([ [1./det(Kinv), Kinv, K, detK, trK], parity])

    imgs.sort(lambda x,y: -1 if x[1] < y[1] else 1 if x[1] > y[1] else 0)

#   if fig == None:
#       fig = figure()

#   f = gcf()
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

def check_model_magnifications(model, **kw):
    kw = kw.copy()

    assert len(model) == 2
    if type(model[1]) == type(0):
        obj,ps = model[0]['obj,data'][model[1]]
    else:
        obj,ps = model

    #obj,ps = model['obj,data'][obj_index]

    for src_index,_ in enumerate(obj.sources):
        rt_imgs  = raytrace([[obj,ps],src_index],**kw)

        Mrt = 0
        for _,_,M,_ in rt_imgs:
            Mrt += abs(M[0])

        kw['initial_guess'] = [x.pos for x in obj.sources[0].images]
        obj_imgs = raytrace([[obj,ps],src_index],**kw)

        Mobj = 0
        for _,_,M,parity in obj_imgs:
            Mobj += abs(M[0])

        if abs(Mrt-Mobj) / Mobj > 0.05: 
            print 'Rejected:', abs(Mrt-Mobj), Mobj
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

    matshow(d)

    xy = obj.basis.refined_xy_grid(ps)[1:-1,1:-1]

    # Create flattened *views*
    xy     = xy.ravel()
    dravel = d.ravel()

    imgs = []
    offs = []
    print 'searching...'
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

    imglist = [[seq[0][0], seq[0][3]]]

    prev = seq[0][1]
    for img,t,_,parity in seq[1:]:
        t0 = convert('arcsec^2 to days', t-prev, obj.z, ps['nu'])
        imglist.append([img, parity,t0])
        prev = t

    return imglist

def write_code(model, obj_index, src_index, seq, simple=False):

    obj,ps = model['obj,data'][obj_index]

    if not seq: return

    letters = [ x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    
    while len(seq) > len(letters):
        letters += [ x+x[0] for x in letters[-26:] ]

    obs = observables(model, obj_index, src_index, seq)

    #def img2str(img, time_delay, l, parity):
        #return "['%s', (% 9.5f,% 9.5f), '%s', %.4f]" % (l, img.real, img.imag, parity, time_delay)

    def img2str(a):
        if len(a[1]) == 2:
            return "['%s', (% 9.5f,% 9.5f), '%s']" % (a[0],a[1][0].real,a[1][0].imag, a[1][1])
        else:
            return "['%s', (% 9.5f,% 9.5f), '%s', %.4f]" % (a[0],a[1][0].real,a[1][0].imag, a[1][1], a[1][2])
        
    print "[" + ",\n ".join(map(img2str, zip(letters, obs))) + "]"
    return
        
    imglist = ["['%s', (% 9.5f,% 9.5f), '%s']" % (letters[0], seq[0][0].real, seq[0][0].imag,seq[0][3])]
    prev = seq[0][1]
    for [img,t,_,parity],l in zip(seq[1:], letters[1:]):
        t0 = convert('arcsec^2 to days', t-prev, obj.z, ps['nu'])
        imglist.append(img2str(img,t0,l,parity))
        prev = t

    print "%.2f, [%.4f, %.4f]," % (obj.sources[src_index].z, ps['src'][src_index].real, ps['src'][src_index].imag)
    print "[" + ",\n ".join(imglist) + "]"

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
                print '@', t-prev, ps['1/H0'], obj.basis.top_level_cell_size**2
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

