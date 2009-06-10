from __future__ import division
import sys
from math import sqrt
from numpy import amin, amax, ndenumerate, diff, logical_and, argwhere, zeros_like, argsort, sort, abs
from basis.pixels import *
from scales import density_to_internal, distance_to_internal, time_to_internal, time_to_physical
from pylab import scatter, contour, over

#-------------------------------------------------------------------------------
# Setup an object
#-------------------------------------------------------------------------------
Rpix = 12       # [pixels]
Rmap = 0.5      # [arcsec]

srcx,srcy = 0.01,0.01

obj = globject('raytrace')
pixrad(Rpix)
maprad(Rmap)
zlens(0.5)
lens(1.5)

model(0)

#-------------------------------------------------------------------------------
# Prepare a solution and load it in.
#-------------------------------------------------------------------------------

sol = zeros(obj.basis.nvar+1)
ps = obj.basis.packaged_solution(sol)
ps['src'][0:2] = srcx, srcy
ps['1/H0'] = 14
model = [obj,ps]

print distance_to_physical(model, 1)

phys_cell_size = distance_to_physical(model, obj.basis.cell_size)
grid = zeros((2*Rpix+1, 2*Rpix+1))

if 1:
    fname = '../../tests/ddehnensphere_1.txt'
    fname = '../../tests/triaxial.txt'
    fname = '../../tests/triaxgal.txt'
    f = loadtxt(fname, usecols=(0,1,3))

    gxmin = distance_to_physical(model, -(Rmap+obj.basis.cell_size/2))
    gxmax = distance_to_physical(model,  (Rmap+obj.basis.cell_size/2))
    gymin = distance_to_physical(model, -(Rmap+obj.basis.cell_size/2))
    gymax = distance_to_physical(model,  (Rmap+obj.basis.cell_size/2))

    # FIXME: Should invert the y axis
    xs = []
    ys = []

    for x,y,m in f:

        #if m > 1e9: continue
        if not gxmin <= x < gxmax: continue
        if not gymin <= y < gymax: continue

        xs.append(distance_to_internal(model, x))
        ys.append(distance_to_internal(model, y))
        
        i = int((y - gymin) / phys_cell_size)
        j = int((x - gxmin) / phys_cell_size)
        grid[i,j] += m * 10

grid /= phys_cell_size**2
grid *= density_to_internal(model, 1) 

#-------------------------------------------------------------------------------
# Find the image positions
#-------------------------------------------------------------------------------

o = obj.basis.array_offset
mass = grid.ravel()[obj.basis.insideL].take(obj.basis.pmap)
sol[o+obj.basis.pix_start : o+obj.basis.pix_end] = mass
sol[o+obj.basis.H0] = time_to_internal(model, 14)
ps = obj.basis.packaged_solution(sol)
ps['src'][0:2] = srcx, srcy
model = [obj,ps]

arrival = array(obj.basis.arrival_grid(ps)[0], copy=True)

abs_tx = abs(diff(arrival, axis=1)[:-1,:])
abs_ty = abs(diff(arrival, axis=0)[:,:-1])
eps    = amax(abs_tx) - amin(abs_tx) + amax(abs_ty) - amin(abs_ty)

#-
# Tolerances on image finding. ztol specifies an upper-bound on "zero". Pixels
# with a value below this will be considered as potential image locations.
# rtol is a tolerance on radius. Pixels that satisfy ztol are considered new
# images if they are not within rtol distance from another image.
#-
#ztol = .02
rtol = obj.basis.cell_size * 2

d  = ((abs_tx+abs_ty) / eps)
xy = obj.basis.refined_xy_grid(ps)[:-1,:-1]

# Create flattened *views*
xy     = xy.ravel()
dravel =  d.ravel()

if 1:
    imgs = []
    offs = []
    print 'searching...'
    for i in argsort(dravel):
        if i > 0 and dravel[i] > 2*dravel[i-1]: break
        #if dravel[i] > ztol: break
        for img in imgs:
            if abs(img-xy[i]) <= rtol: break
        else:
            imgs.append(xy[i])
            offs.append(i)

    if imgs:
        print imgs
        imgs = array(imgs)

        g0 = array(arrival[:-1,:-1], copy=True)
        g0ravel = g0.ravel()
        times = g0ravel[offs]
        order = argsort(times)

        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i,o in enumerate(order):
            print '%s = %-.4f, %-.4f' % (letters[i], imgs[o].real, imgs[o].imag)

        print
        sys.stdout.write("lens(%.2f,A,'min'" % obj.zlens)
        td = times[order[0]]
        for i,o in enumerate(order[1:]):
            t = 1/(times[o]-td) / obj.scales['time'] * ps['1/H0'] * obj.basis.cell_size**2
            sys.stdout.write(",%s,'',%.4f" % (letters[i+1], t)) #time_to_physical(model, order[i]-td)))
            td = times[o]
            g0ravel[offs[o]] = i+1
        sys.stdout.write(')\n')

        matshow(g0, extent=[-Rmap,Rmap,-Rmap,Rmap])
        figure()
        contour(g0, 50,  colors='k',               linewidths=1, 
             extent=[-Rmap,Rmap,-Rmap,Rmap], origin='upper', extend='both')
        over(scatter,imgs.real, imgs.imag)

#-------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------

matshow(abs_tx, extent=[-Rmap,Rmap,-Rmap,Rmap])
matshow(abs_ty, extent=[-Rmap,Rmap,-Rmap,Rmap])
matshow(d, extent=[-Rmap,Rmap,-Rmap,Rmap])
over(contour, arrival, 50,  colors='k',               linewidths=1, 
     extent=[-Rmap,Rmap,-Rmap,Rmap], origin='upper', extend='both')
matshow(arrival, extent=[-Rmap,Rmap,-Rmap,Rmap])

if 0:
    
    matshow(grid, extent=[-Rmap,Rmap,-Rmap,Rmap])

    #obj.basis.sub_division = 1


    print ps['R_phys']
    print ps['encmass_phys']

    figure()
    encmass_plot([{'objs': [model]}])
    figure()
    arrival_plot([obj,ps], 0)
    src_plot(model)
    figure()
    mass_plot([obj, ps])

show()

