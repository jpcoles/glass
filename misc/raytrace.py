from __future__ import division
import sys
from numpy import amin, amax, diff, argsort, abs, array, sum, \
                  mat, eye, asarray, matrix, empty_like, zeros, \
                  sort, any, sqrt, dot, ceil
from scales import time_to_physical
from potential import poten
from scipy.ndimage.filters import correlate1d
from scipy.misc import central_diff_weights
from scipy.optimize import fsolve, fmin

from pylab import contour, matshow, show, over, figure, gcf

fig = None

def raytrace(obj, ps, src_index, nimgs=None, eps=None, eps2=None, initial_guess=None):
    global fig

    srcdiff = obj.basis.srcdiff(ps, src_index).copy()
    ploc    = obj.basis.ploc
    src     = ps['src'][src_index]
    zcap    = obj.sources[src_index].zcap

    if eps is None:
        eps = 2 * obj.basis.top_level_cell_size
        #eps = sqrt(2.1) * obj.basis.top_level_cell_size

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
        m = obj.basis.top_level_cell_size
        e = ceil(obj.basis.top_level_cell_size / eps) * obj.basis.top_level_cell_size
        for i,j in enumerate(asort):

            if srcdiff[j] > m: break

            n = abs(ploc[j] - ploc) <= e
            #n = abs(ploc[j] - ploc) < sqrt(2.1)*obj.basis.top_level_cell_size

            has_bad_neighbors = any(srcdiff[n] == -1)

            srcdiff[n] = -1

            if has_bad_neighbors: continue

            initial_guess.append(ploc[j])

#           for ii in initial_guess:
#               if abs(ploc[j] - ii) <= eps: break
#           else:

#        if len(initial_guess) >= nimgs: break

#       if fig == None:
#           fig = figure()

#       reorder = empty_like(srcdiff)
#       reorder.put(obj.basis.pmap, srcdiff)
#       sd = zeros((2*obj.basis.pixrad+1)**2)
#       sd[obj.basis.insideL] = reorder
#       #sd[:len(srcdiff)] = srcdiff #reorder
#       sd = sd.reshape((2*obj.basis.pixrad+1,2*obj.basis.pixrad+1))
#       matshow(sd, fignum=fig.number)

#       raw_input()

#   #figure(f.number)
#   figure()

#       if not any(abs(ploc[j] - ploc[asort[:i]]) <= eps):
#           srcdiff[j] = 10
#           initial_guess.append(ploc[j])

    def lenseq(theta0):
        theta = complex(*theta0)
        r = src - theta + obj.basis.deflect(theta, ps) / zcap
        #print src, theta, obj.basis.deflect(theta, ps) / zcap, abs(r)
        return [r.real, r.imag]

    initial_guess.append(0j)
    print 'Initial guesses', initial_guess

    imgs = []
    if obj.shear: s1,s2 = ps['shear']
    for img in initial_guess:
        x, infodict, ier, mesg = fsolve(lenseq, [img.real,img.imag], full_output=True, warning=True)
        if not ier:
            print 'Near image found at', x, 'but not saving it.'
            print 'From fsolve:', mesg
            continue

        #-----------------------------------------------------------------------
        # Only accept if the solution is distinct from previous solutions.
        #-----------------------------------------------------------------------
        i = complex(*x)
        if eps2 and abs(img-i) > eps2: break
        for j,t in imgs:
            if abs(i-j) < eps: break
        else:
            #tau  = abs(i)**2 / 2 - (i.real*src.real + i.imag*src.imag)
            tau  = abs(i-src)**2 / 2
            tau *= zcap
            tau -= dot(ps['kappa'], poten(i - obj.basis.ploc, obj.basis.cell_size))
            if obj.shear:
                tau -= s1*obj.shear.poten(1,i) + s2*obj.shear.poten(2,i)
            imgs.append([i,tau])

    imgs.sort(lambda x,y: -1 if x[1] < y[1] else 1 if x[1] > y[1] else 0)

#   if fig == None:
#       fig = figure()

#   f = gcf()
#   figure(fig.number)
#   reorder = empty_like(srcdiff)
#   reorder.put(obj.basis.pmap, srcdiff)
#   sd = zeros((2*obj.basis.pixrad+1)**2)
#   sd[obj.basis.insideL] = reorder
#   #sd[:len(srcdiff)] = srcdiff #reorder
#   sd = sd.reshape((2*obj.basis.pixrad+1,2*obj.basis.pixrad+1))
#   matshow(sd, fignum=fig.number)

#   #figure(f.number)
#   figure()

    print imgs

    return imgs

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

        new_image = true
        for img in imgs:
            if abs(img-xy[i]) <= eps: 
                new_image = false

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

def write_code(obj, ps, sys_index, seq, simple=False):

    if not seq: return

    if not simple:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for [img,t], l in zip(seq, letters):
            sys.stdout.write('%s = %-.4f, %-.4f\n' % (l, img.real, img.imag))

        sys.stdout.write("source(%.2f,A,'min'" % obj.sources[sys_index].z)
        prev = seq[0][1]
        for [img,t],l in zip(seq[1:], letters[1:]):
            t0 = time_to_physical(obj, t-prev) * ps['1/H0'] * obj.basis.top_level_cell_size**2
            if t0 < 1e-4:
                sys.stdout.write(", %s,'',%.4e" % (l, t0))
            else:
                sys.stdout.write(", %s,'',%.4f" % (l, t0))
            prev = t
        sys.stdout.write(')\n')

    else:

        sys.stdout.write("source(%.2f, (%-.4g,%-.4g),'min'" % (
            obj.sources[sys_index].z, seq[0][0].real, seq[0][0].imag))

        prev = seq[0][1]
        for img,t in seq[1:]:
            t0 = time_to_physical(obj, t-prev) * ps['1/H0']
            sys.stdout.write(", (%-.4g,%-.4g),'',%.4g" % (img.real, img.imag, t0))
            prev = t
        sys.stdout.write(')\n')

