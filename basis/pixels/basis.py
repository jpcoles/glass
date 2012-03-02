#
# Coordinates should be located at the centers of pixels
#
from __future__ import division
import sys
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jonathan/GLASS/glass/')

import numpy as np
from numpy import zeros, amin, amax, min, max, argmax, argmin, abs, vectorize, negative, array, take,   \
                  ndindex, empty, arange, empty_like, ogrid, round, where,  \
                  unique, round, argwhere, asarray, lexsort, angle, floor,  \
                  conj, arctan2, atleast_2d, linspace, cumsum, sum, repeat, \
                  zeros_like, ndenumerate, s_, isinf, where, dot, array, \
                  add, subtract, multiply, append, ceil, ones, sort, sign, diff, \
                  trunc, argmin, logical_and, logical_not, nan_to_num, histogram2d, \
                  sin, cos, pi, matrix, diag, average, log, sqrt, mean, hypot

if 1:
    import pylab as pl
    from pylab import plot, show, matshow, figure, contour, over, scatter, subplot, draw, ion, ioff
    from matplotlib.patches import Circle, Rectangle

from scipy.ndimage.filters import correlate
from scipy.misc import central_diff_weights
from scipy.linalg import eig, eigh, norm
from scipy.signal import convolve2d

import scipy.ndimage._ni_support as _ni_support
import scipy.ndimage._nd_image as _nd_image

from math import atan2, pi
from itertools import izip

from environment import env
import potential
from potential import poten_indef, poten2d_indef, poten, poten_dx, poten_dy
from scales import convert
from handythread import parallel_map2, parallel_map

from potential import poten_dxdx, poten_dydy, maginv, poten_dxdy

from log import log as Log

def neighbors(r,s, Rs):
    rs = abs(Rs-r)
    return argwhere(logical_and(0 < rs, rs <= s)).ravel()

def all_neighbors(Rs, L):
    return [ [i, r, neighbors(r,s,Rs)] for i,[r,s] in enumerate(izip(Rs, L)) ]

def xy_grid(L, S=1, scale=1):
    """Return a grid with radius L (i.e. diameter=2*L+1) where each
       of the cells are subdivided into S parts. The result is scaled
       by 'scale'."""
    gx = linspace(-L,L, (2*L+1) * S) * scale
    gy = atleast_2d(-gx).T
    return vectorize(complex)(gx, gy)

def xy_list(L, R=0, refine=1):
    """Return a grid with radius L (i.e. diameter=2*L+1) where each
       of the cells are subdivided into S parts. The result is scaled
       by 'scale'."""
    gx = linspace(-L,L, (2*L+1))
    gy = atleast_2d(-gx).T
    xy = vectorize(complex)(gx, gy).ravel()

    size = ones(len(xy))
    rs = []
    rcs = []        # radial cell size

    if R:
        assert refine%2==1
        f = 1/refine

        #print "!@#!", R, refine, (R-1)*refine+(refine-1)
        rs  = [ r*1/refine for r in xrange((R-1)*refine+(refine-1)) ]
        rcs = [   1/refine for r in xrange((R-1)*refine+(refine-1)) ]

        if R == 1:
            rs = rs[:-1]
            rcs = rcs[:-1]

        t = []
        v = []
        s = []
        for i in argwhere(abs(xy) < R):
            t += [xy[i] + f * complex(x,y) 
                  for y in linspace(-(refine//2), refine//2, refine) # 'extra' parens are important 
                  for x in linspace(-(refine//2), refine//2, refine) # because of integer rounding
                  if not (x == 0 and y == 0) ]

            s += [f] * (refine**2-1)
            size[i] = f

        xy  = append(xy, t)
        size = append(size, s)

    rs  += [ r for r in xrange(R, L+1) ]
    rcs += [ 1 for r in xrange(R, L+1) ]
    
    return xy.ravel(), size.ravel(), array(rs, 'float'), array(rcs, 'float')

def fast_correlate(input, weights, output = None, mode = 'reflect', cval = 0.0, origin = 0):
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    output, return_value = _ni_support._get_output(output, input)
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate(input, weights, output, mode, cval, origins)
    return return_value

def memoize(func):
    def f(self, data, *args, **kwargs):
        #print func.__name__, data.has_key(func.__name__), data.keys()
        if not data.has_key(func.__name__): 
            data[func.__name__] = func(self, data, *args, **kwargs)
        return data[func.__name__]

    return f

def visual_neighbor_verification(self, nbrs):
    ion()

    f=figure(figsize=(8,8))
    sp=f.add_subplot(111)
    for N in nbrs:
        sp.clear()
        #plot(self.ploc.real, self.ploc.imag)
        #scatter(self.ploc.real, self.ploc.imag, s=(5)**2, marker='s')

        i=N[0]
        sp.plot(self.ploc[:i+1].real, self.ploc[:i+1].imag, 'k-', marker='s')
        sp.scatter(self.ploc.real, self.ploc.imag, s=(5*self.cell_size)**2, marker='s')
        for r,s in izip(self.ploc, self.cell_size):
            sp.add_artist(Rectangle([r.real-s/2, r.imag-s/2], s,s, fill=False))
            #sp.add_artist(Circle([r.real, r.imag], radius=1.5*s, fill=False))
        #scatter(self.ploc.real, self.ploc.imag, s=(5*self.cell_size)**2, marker='s')

        sp.set_aspect('equal')

        A = []
        for n in N[2]:
            r = self.ploc[n]
            s = self.cell_size[n]
            a = sp.add_artist(Rectangle([r.real-s/2, r.imag-s/2], s,s, fill=True))
            A.append(a)

        draw()
        raw_input()

def Xestimated_Re(obj, ps, src_index):

    #---------------------------------------------------------------------
    # Estimate an Einstein radius. 
    # Take the inertia tensor of the pixels above kappa_crit and use the
    # eigenvalues to scale the most distance pixel position to the major
    # and minor axes. Re is then defined here as the mean of the two.
    #
    # TODO: On convergence. Since the centers of each pixel are used, as
    # the resolution increases, Re will tend to move outward. A better
    # solution would be to use the maximum extent of each pixel.
    #---------------------------------------------------------------------

    kappa = ps['kappa'] / obj.sources[src_index].zcap

    w = kappa >= 1.0

    if not w.any(): return None

    r = obj.basis.ploc[w]
    I = matrix(zeros((3,3)))
    m = kappa[w]
    I[0,0] =  sum(m*(r.real**2 + r.imag**2))
    I[1,1] =  sum(m*(r.imag**2))
    I[2,2] =  sum(m*(r.real**2))
    I[1,2] = -sum(m*(r.real * r.imag))
    I[2,1] = I[1,2]

    V,D = eig(I)

    D1 = asarray(D[1:,0]).T # (throw away the axis perpendicular to the image plane)
    D2 = asarray(D[1:,1]).T

    r1 = r[argmax([abs(dot(abs([x.real,x.imag]),D1)) for x in r])]
    r2 = r[argmax([abs(dot(abs([x.real,x.imag]),D2)) for x in r])]
    Vl = abs(r1)
    Vs = abs(r2)
    if Vl < Vs: 
        Vl,Vs = Vs,Vl
        D1,D2 = D1,D2

    return mean([Vl,Vs]), Vl, Vs, arctan2(D1[1], D1[0]) * 180/pi

def gauss_kernel(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def intersect(A,B):
    if B[0] < A[1] or B[1] > A[0] or B[2] > A[3] or B[3] < A[2]: return [0,0,0,0]
    v = [A[0],A[1], B[0],B[1]]
    h = [A[2],A[3], B[2],B[3]]
    v.sort()
    h.sort()
    #print v,h
    b,t = max(v[0],v[1]), min(v[2],v[3])
    l,r = max(h[0],h[1]), min(h[2],h[3])
    #print t,b,l,r
    #print r-l, t-b
    return [t,b,l,r]

def intersect_frac(A,B):
    t,b,l,r = intersect(A,B) 
    areaB = (B[0]-B[1]) * (B[3]-B[2])
    return (max(r-l,0) * max(t-b,0)) / areaB

    areaA = (A[0]-A[1]) * (A[3]-A[2])
    areaB = (B[0]-B[1]) * (B[3]-B[2])
    v = [A[0],A[1], B[0],B[1]]
    h = [A[2],A[3], B[2],B[3]]
    v.sort()
    h.sort()
    #print v,h
    b,t = max(v[0],v[1]), min(v[2],v[3])
    l,r = max(h[0],h[1]), min(h[2],h[3])
    #print t,b,l,r
    #print r-l, t-b
    return (max(r-l,0) * max(t-b,0)) / areaB


class PixelBasis(object): 

    def __init__(self):

        self.myobject = None

        self.pixrad    = None       # [pixels]
        self.maprad    = None       # [arcsec]
        self.cell_size = None       # [arcsec]
        
        self.mapextent = None       # [arcsec]

        self.samplex_row_offset = None

        self.nvar = 0
        self.pix_start,    self.pix_end    = None,None
        self.shear_start,  self.shear_end  = None,None
        self.ptmass_start, self.ptmass_end = None,None
        self.srcpos_start, self.srcpos_end = None,None
        self.H0                            = None

        self.inner_image_ring, self.outer_image_ring = None,None

        #self.map_shift = 10        # [arcsec]

        self.lnr = None
        self.subdivision = 5

        self.hiresR = 0
        self.hires_levels = None

        self.grad_rmax = 1
        #self.grad_rmax = sqrt(10.5)

    def __getattr__(self, name):
        if name == 'nbrs':
            Log( 'Finding neighbors...' )
            super(PixelBasis, self).__setattr__('nbrs',  all_neighbors(self.int_ploc,            1.5 * self.int_cell_size))
            return self.nbrs
        elif name == 'nbrs2':
            Log( 'Finding neighbors 2...' )
            super(PixelBasis, self).__setattr__('nbrs2', all_neighbors(self.int_ploc, self.grad_rmax * self.int_cell_size))
            return self.nbrs2
        else:
            raise AttributeError('Attribute %s not found in PixelBasis' % name)

    def init(self, obj):
        self.myobject = obj

        L = obj.basis.pixrad

        #---------------------------------------------------------------------
        # Get all image positions (except maximums near the center)
        #---------------------------------------------------------------------
        rs = [ abs(img.pos) for src in obj.sources for img in src.images if img.parity_name != 'max']

        Log( '=' * 80 )
        Log( 'PIXEL BASIS for %s' % obj.name )
        Log( '=' * 80 )

        if rs:
            rmin, rmax = min(rs), max(rs)
            Log( "rmin=%s rmax=%s L=%s" % (rmin, rmax, L) )
        else:
            assert obj.maprad is not None, 'If no images are given, then maprad must be specified.'
            rmin, rmax = 0, obj.maprad

        del rs

        #---------------------------------------------------------------------
        # Select a maprad if not specified. Remember, coordinates are of
        # pixels centers.
        #---------------------------------------------------------------------
        self.maprad = obj.maprad
        if self.maprad is None:
            self.maprad = rmax * 1.1 
            self.maprad = rmax * 1.5
            self.maprad = rmax / (L-1) * L
            self.maprad = rmax * amax([1.2, L/(L-2)])
            #print self.maprad, rmax, amax([1.2, L/(L-2)])
            #assert 0
            #Log( 'Adjusting maprad to allow one ring outside images.' )
            #self.maprad = rmax+rmin
            self.maprad = rmax + amin([rmin, rmax-rmin])

        self.map_shift = self.maprad        # [arcsec]

        self.top_level_cell_size = self.maprad / L

        self.mapextent = self.top_level_cell_size * (2*L + 1)/2


        #---------------------------------------------------------------------
        # Create pixel map -- symmetry not supported
        #---------------------------------------------------------------------
        self.xy,        \
        self.int_cell_size, \
        self.rs,        \
        self.int_radial_cell_size = xy_list(L, self.hiresR, self.hires_levels)

        #print  self.rs

        mags = abs(self.xy)
        insideL  = self.insideL  = argwhere(mags <  (2*L+1)/2).T[0]
        outsideL = self.outsideL = argwhere(mags >= (2*L+1)/2).T[0]
        self.int_ploc      = self.xy[insideL]
        self.int_cell_size = self.int_cell_size[insideL]

        rkeys = array([self.rs[argmin(abs(abs(p)-self.rs))] for p in self.int_ploc])

        #---------------------------------------------------------------------
        # By sorting by the arctan2() and abs() of the positions we create a
        # spiral from the center out. The central pixel is the first element in
        # the kappa array.
        #---------------------------------------------------------------------
        self.pmap = array(lexsort(keys = (arctan2(self.int_ploc.imag, self.int_ploc.real), rkeys)),
                          dtype=np.int32)

        self.int_ploc  = self.int_ploc.take(self.pmap)
        rkeys          = rkeys.take(self.pmap)
        self.int_cell_size = self.int_cell_size.take(self.pmap)

        #---------------------------------------------------------------------
        # Make neighbor lists
        #---------------------------------------------------------------------
        #Log( 'Finding neighbors...' )
        #self.nbrs  = []
        #self.nbrs2 = []

#       if 0:
#           self.nbrs  = all_neighbors(self.int_ploc, 1.5*self.int_cell_size)
#           self.nbrs2 = all_neighbors(self.int_ploc,     self.int_cell_size*self.grad_rmax)

#           self.nbrs = [ [i, r, argwhere(logical_and(abs(self.int_ploc-r) <  s*1.5, abs(self.int_ploc-r) > 0)).ravel()] 
#                          for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size)) ]

#           self.nbrs = []
#           for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size)):
#               dist = abs(self.int_ploc-r)
#               self.nbrs.append([i, r, argwhere(logical_and(dist < 1.5*s, dist > 0)).ravel()]) 


#           print 'Finding neighbors...'
#           self.nbrs2 = [ [i,r, argwhere(abs(self.int_ploc[self.nbrs[i][2]]-r) < 1.5*s)] 
#                         for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size)) ]

#           print 'done!'
#           self.nbrs2 = [ [i, r, argwhere(logical_and(abs(self.int_ploc-r) <= s, abs(self.int_ploc-r) > 0)).ravel()] 
#                         for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size * self.grad_rmax)) ]

#       else:
#           Log( "SKIPPING NEIGHBOR FINDING!" )

        #print self.nbrs

        #---------------------------------------------------------------------
        #---------------------------------------------------------------------


        npix = len(self.int_ploc)
        #print "npix =", npix

        # Useful for some contraints
        #print argwhere(self.int_ploc == 0)
        self.central_pixel = argwhere(self.int_ploc == 0).item()

        #print "nbrs =", reduce(lambda s,x: s + len(x[0])+len(x[1]), self.nbrs, 0)

        #---------------------------------------------------------------------
        # Each pixel gets assigned a ring...
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        # Now make lists of all the pixels on a given ring
        #---------------------------------------------------------------------
        self.rings = [ argwhere(rkeys == i).ravel() for i in unique(rkeys) ]

        self.oppose = [ argwhere(self.int_ploc == -l).ravel() for l in self.int_ploc ]

        # XXX: Need these for the annular density prior
        #inner_image_ring = rmin // self.cell_size
        #outer_image_ring = rmax // self.cell_size + 1

        #---------------------------------------------------------------------
        #Log( "%i rings:" % len(self.rings) )
        #for r in self.rings:
        #    Log( "  % 4i ...% 4i" % (r[0], r[-1]) )
        #---------------------------------------------------------------------

        #---------------------------------------------------------------------
#       sp=subplot(122)
#       for i,r in enumerate(self.rings):
#           plot(self.int_ploc[r].real, self.int_ploc[r].imag)
#           #scatter(self.ploc[r].real, self.ploc[r].imag, s=(5*cell_size)**2, marker='s', c=pylab.cm.jet(i*20))
#       for r,s in izip(self.int_ploc, self.int_cell_size):
#           sp.add_artist(Rectangle([r.real-s/2, r.imag-s/2], s,s, fill=False))
#       sp.set_aspect('equal')
        #show()
        #---------------------------------------------------------------------
        if 0:
            f=pl.figure()
            sp=f.add_subplot(111, aspect='equal')
            xmin,xmax = np.inf, -np.inf
            ymin,ymax = np.inf, -np.inf
            for r,s in izip(self.int_ploc, self.int_cell_size):
                x,y = r.real-s/2, r.imag-s/2
                xmin,xmax = min([xmin,x]), max([xmax,x+s])
                ymin,ymax = min([ymin,y]), max([ymax,y+s])
                sp.add_artist(Rectangle([x,y], s,s, fill=False))
            pl.xlim(xmin,xmax)
            pl.ylim(ymin,ymax)

            pl.show()
            assert 0



        #---------------------------------------------------------------------
        # Now adjust the positions so that they are physical
        #---------------------------------------------------------------------
        self.rs                *= self.top_level_cell_size
        self.cell_size          = self.int_cell_size        * self.top_level_cell_size
        self.radial_cell_size   = self.int_radial_cell_size * self.top_level_cell_size
        self.ploc               = self.int_ploc * self.cell_size
        self.xy                *= self.top_level_cell_size


        #visual_neighbor_verification(self, self.nbrs)
        #visual_neighbor_verification(self, self.nbrs2)

        #visual_neighbor_verification(self, all_neighbors(self.ploc, 1.1*self.cell_size))
        #print self.top_level_cell_size
        #visual_neighbor_verification(self, all_neighbors(self.ploc, [.2]*npix))


#       for n in self.nbrs2[1][2]:
#           r = self.int_ploc[n]
#           s = self.int_cell_size[n]
#           sp.add_artist(Rectangle([r.real-s/2, r.imag-s/2], s,s, fill=False))
            #sp.add_artist(Circle([r.real, r.imag], radius=1.5*s, fill=False))

        #print "...", self.radial_cell_size

        #---------------------------------------------------------------------
        # Setup variable offsets for the constraint arrays
        #---------------------------------------------------------------------
        self.pix_start,    self.pix_end    = 0, npix
        self.shear_start,  self.shear_end  = self.pix_end,   self.pix_end+2*(obj.shear is not None)
        self.ptmass_start, self.ptmass_end = self.shear_end, self.shear_end

        #nimgs=0
        #for src in obj.sources:
            #nimgs += len(src.images)

        #print "nimgs =", nimgs

        self.srcpos_start,    self.srcpos_end    = self.ptmass_end, self.ptmass_end+2*len(obj.sources)
        self.H0 = self.srcpos_end

        self.nvar = self.H0+1

        H0inv_ref_as_nu = convert('H0^-1 in Gyr to nu', env().H0inv_ref)
        Log( 'Pixel basis' )
        Log( '    Pixel radius         = %i'  % self.pixrad )
        Log( '    Map radius           = %.4f [arcsec] %s' % (self.maprad, 'Distance to center of outer pixel.') )
        Log( '    Map Extent           = %.4f [arcsec] %s' % (self.mapextent, 'Distance to outer edge of outer pixel.') )
        Log( '    top_level_cell_size  = %.4f [arcsec]'  % self.top_level_cell_size )
        Log( '    Map radius           = %3.4f [kpc]    H0inv=%.1f' % (convert('arcsec to kpc', self.maprad, obj.dL, H0inv_ref_as_nu), env().H0inv_ref))
        Log( '    Map Extent           = %3.4f [kpc]    H0inv=%.1f' % (convert('arcsec to kpc', self.mapextent, obj.dL, H0inv_ref_as_nu), env().H0inv_ref ))
        Log( '    top_level_cell       = %3.4f [kpc]    H0inv=%.1f' % (convert('arcsec to kpc', self.top_level_cell_size, obj.dL, H0inv_ref_as_nu), env().H0inv_ref ))
        Log( '    Number of rings      = %i'    % len(self.rings) )
        Log( '    Number of pixels     = %i'    % npix )
        Log( '    Number of variables  = %i'    % self.nvar )
        Log( '    Central pixel offset = %i'    % self.central_pixel )
        Log( 'Offsets        % 5s  % 5s' % ('Start', 'End') )
        Log( '    pix        % 5i  % 5i' % (self.pix_start, self.pix_end) )
        if self.shear_start == self.shear_end:
            Log( '    shear           %s' % 'None')
        else:
            Log( '    shear      % 5i  % 5i' % (self.shear_start,  self.shear_end) )

        if self.ptmass_start == self.ptmass_end:
            Log( '    ptmass          %s' % 'None')
        else:
            Log( '    ptmass     % 5i  % 5i' % (self.ptmass_start, self.ptmass_end) )

        if self.srcpos_start == self.srcpos_end:
            Log( '   osrcpos          %s' % 'None')
        else:
            Log( '    srcpos     % 5i  % 5i' % (self.srcpos_start, self.srcpos_end) )

        Log( '    H0         % 5i'       % (self.H0) )

#       xy    = self.refined_xy_grid({})

#       print where(xy == obj.basis.ploc)
#       r,c=xy.shape
#       print xy[r//2]
#       print obj.basis.ploc

#       sys.exit(0)


    def solution_to_dict(self, sol):
        obj    = self.myobject
        o      = self.array_offset

        ps = {}


        #---------------------------------------------------------------------
        # These come directly from the solution
        #---------------------------------------------------------------------

        ps['kappa']  = sol[ o+self.pix_start    : o+self.pix_end      ]

        ps['shear']  = sol[ o+self.shear_start  : o+self.shear_end    ] \
                       if obj.shear else array([0,0])
        ps['ptmass'] = sol[ o+self.ptmass_start : o+self.ptmass_start ]
        ps['src']    = [complex(sol[o+i], sol[o+i+1]) / obj.sources[j].zcap - complex(self.map_shift, self.map_shift)
                        for j,i in enumerate(xrange(self.srcpos_start, self.srcpos_end,2))]
        ps['src'] = array(ps['src'])
 
        ps['nu']     = sol[o+self.H0]
        ps['H0']     = convert('nu to H0 in km/s/Mpc', ps['nu'])
        ps['1/H0']   = convert('nu to H0^-1 in Gyr',   ps['nu'])


        #---------------------------------------------------------------------
        # These are computed quantities based on the above solutions.
        #---------------------------------------------------------------------

        #print self.myobject.scales, self.H0, sol[o+self.H0], len(sol)
        #print self.srcpos_start
        #assert not isinf(ps['1/H0'])
        #if ps['1/H0'] == float('inf'): ps['1/H0'] = 1

        #rscale = Arcsec_to_Kpc(obj, 1, ps['1/H0'])

        return ps

    @memoize
    def time_delays(self, data):

        obj  = self.myobject

        pix_start,    pix_end    = self.pix_start,    self.pix_end
        shear_start,  shear_end  = self.shear_start,  self.shear_end
        ptmass_start, ptmass_end = self.ptmass_start, self.ptmass_end

        delays = []
        for i, src in enumerate(obj.sources):

            s = 2*i
            src_pos = data['src'][i] # + self.map_shift

            prev = 0
            d = []
            for j,img in enumerate(src.images):

                tau  = abs(img.pos-src_pos)**2 / 2  *  src.zcap
                tau -= dot(data['kappa'], poten(img.pos - self.ploc, self.cell_size))

                if obj.shear:
                    tau -= data['shear'][0] * obj.shear.poten(1, r) 
                    tau -= data['shear'][1] * obj.shear.poten(2, r)

                if j > 0:
                    d.append(time_to_physical(obj, tau-prev) * data['H0'])
                prev = tau

            delays.append(d)

        return delays

    @memoize
    def refined_xy_grid(self, data):
        L = self.pixrad
        S = self.subdivision

        Q = S*(2*L+1) // 2
        gx = linspace(-Q, Q, 2*Q+1) * (self.top_level_cell_size/S); assert (len(gx) % 2) == 1
        gy = atleast_2d(-gx).T
        return vectorize(complex)(gx, gy)

    def _to_grid(self, a, refinement=1):
        L = self.pixrad
        reorder = empty_like(a)
        reorder.put(self.pmap, a)
        grid = zeros((2*L+1)**2)
        grid[self.insideL] = reorder
        grid = grid.reshape((2*L+1,2*L+1))
        if refinement > 1:
            grid = repeat(grid, refinement, axis=1)
            grid = repeat(grid, refinement, axis=0)
        return grid

    def from_grid(self, a):
        print self.insideL
        print self.pmap
        return a.ravel()[self.insideL].take(self.pmap)

    def mass_grid(self, data):
        Log( "WARNING: use of mass_grid is deprecated. Use kappa_grid instead." )
        return self.kappa_grid(self, data)

    @memoize
    def kappa_grid(self, data):
        #print 'kappa_grid:', id(data)
        L = self.pixrad
        S = self.subdivision
        assert (S%2)==1

        grid = self._to_grid(data['kappa'], S)

        #self.refined_cell_size = repeat(self.cell_size, S) / S

        return grid

    def _lnr(self):
        """ Returns a grid of the indefinite integral of the potential. """
        if self.lnr is None:
            L, S = self.pixrad, self.subdivision
            assert (S%2)==1

            Q = S * (2*L+1)
            gx = linspace(-Q-0.5,Q+0.5, 2*Q+2) * (self.top_level_cell_size/S)
            assert (len(gx) % 2) == 0
            gy = atleast_2d(gx).T
            xy = vectorize(complex)(gx, gy)

            self.lnr = poten_indef(xy)

        return self.lnr

    @memoize
    def potential_grid(self, data):
        obj = self.myobject
        L = obj.basis.pixrad
        S = obj.basis.subdivision
        assert (S%2)==1

        kappa = self.kappa_grid(data)
        phi   = zeros_like(kappa)
        lnr   = self._lnr()
        lr,lc = lnr.shape
        mr,mc = kappa.shape

        r0 = lr // 2
        c0 = lc // 2

        o = S*(2*L+1)

        l = empty_like(kappa)
        for [r,c],m in ndenumerate(kappa):
            if not m: continue

            rs,re = r0-r, r0-r+o
            cs,ce = c0-c, c0-c+o

            s0 = s_[rs      : re    , cs      : ce    ]
            s1 = s_[rs - 1  : re - 1, cs - 1  : ce - 1]
            s2 = s_[rs - 1  : re - 1, cs      : ce    ]
            s3 = s_[rs      : re    , cs - 1  : ce - 1]

            add(        lnr[s0],
                        lnr[s1], l)
            subtract(l, lnr[s2], l)
            subtract(l, lnr[s3], l)
            multiply(l, m, l)
            subtract(phi, l, phi) #phi -= l
            
        if obj.shear:
            xy   = self.refined_xy_grid(data)
            s1,s2 = data['shear']
            phi -= s1*obj.shear.poten(1, xy) + s2*obj.shear.poten(2, xy)

        #print 'potential_grid: sum', sum(phi)

        return phi

    @memoize
    def potential_contour_levels(self, data):
        obj = self.myobject
        if obj.shear: s1,s2 = data['shear']
        lvls = []
        for i,src in enumerate(obj.sources):
            l = []
            for img in src.images:
                p  = -dot(data['kappa'], poten(img.pos - obj.basis.ploc, obj.basis.top_level_cell_size))
                if obj.shear:
                    p -= s1*obj.shear.poten(1, img.pos) + s2*obj.shear.poten(2, img.pos)
                l.append(p)
            if l: lvls.append(l)
        return lvls
        
    @memoize
    def arrival_grid(self, data):
        obj = self.myobject
        phi = self.potential_grid(data)
        xy  = self.refined_xy_grid(data)
        r   = data['src']
        return [ abs(xy - r[i])**2/2 * src.zcap + phi for i,src in enumerate(obj.sources)]
        #return [ (abs(xy - r[i])**2 - abs(r)**2)/2 * src.zcap + phi for i,src in enumerate(obj.sources)]

    @memoize
    def arrival_contour_levels(self, data):

        obj = self.myobject
        lvls = []
        if obj.shear: s1,s2 = data['shear']

        def _tau(img,r):
            geom  = abs(img.pos - r)**2 / 2  *  src.zcap

            p  = -dot(data['kappa'], poten(img.pos - obj.basis.ploc, obj.basis.top_level_cell_size))
            if obj.shear:
                p -= s1*obj.shear.poten(1, img.pos) + s2*obj.shear.poten(2, img.pos)

            return geom + p

        for i,src in enumerate(obj.sources):
            r = data['src'][i]
            l = [_tau(img,r) for img in src.images if img.parity_name == 'sad']
            if l: lvls.append(l)

        return lvls

    @memoize
    def maginv_grid(self, data):
        obj = self.myobject
        L = obj.basis.pixrad

        kappa   = data['kappa']
        maginv0 = empty_like(kappa)

        grid = []
        for i,src in enumerate(obj.sources):
            for i,theta in enumerate(self.ploc):
                angle = arctan2(theta.imag, theta.real) * 180/pi
                xy,xx,yy = maginv(theta - self.ploc, angle, self.cell_size)
                maginv0[i] = sum((xx + yy) * kappa)

            grid.append(self._to_grid(maginv0))

        return grid

    def deflect(self, theta, data):
        obj = self.myobject
        kappa = data['kappa']
        #dist  = theta - self.ploc
        #s = complex(dot(kappa, nan_to_num(poten_dx(dist,self.cell_size))),
        #            dot(kappa, nan_to_num(poten_dy(dist,self.cell_size))))
        #s = complex(dot(kappa, (poten_dx(dist,self.cell_size))),
                    #dot(kappa, (poten_dy(dist,self.cell_size))))
        s = potential.grad(kappa,theta, self.ploc, self.cell_size)
        if obj.shear:
            s1,s2 = data['shear']
            s += complex(s1*obj.shear.poten_dx(theta) + s2*obj.shear.poten_d2x(theta),
                         s1*obj.shear.poten_dy(theta) + s2*obj.shear.poten_d2y(theta))
        return s

    def magnification(self, r, theta, data):
        
        obj = self.myobject
        kappa = data['kappa']
        dist  = r - self.ploc
        e = sum(kappa * nan_to_num(maginv(dist,theta,self.cell_size)), axis=1)
        K = matrix([ [ e[1], e[0] ], 
                     [ e[0], e[2] ] ])

#       if obj.shear:
#           s1,s2 = data['shear']
#           s += complex(s1*obj.shear.poten_dx(theta) + s2*obj.shear.poten_d2x(theta),
#                        s1*obj.shear.poten_dy(theta) + s2*obj.shear.poten_d2y(theta))
        return K

    def srcdiff(self, data, src_index):
        if not data.has_key('srcdiff'):
            obj = self.myobject

            kappa   = data['kappa']
            deflect = empty_like(self.ploc)
            #dist    = empty_like(self.ploc)
            ploc    = self.ploc
            cell_size = self.cell_size

#           kappa = self.kappa_grid(data).ravel()
#           ploc    = self.refined_xy_grid(data).ravel()
#           deflect = empty_like(ploc)
#           dist    = empty_like(ploc)
#           cell_size = self.top_level_cell_size / self.subdivision


            if not data.has_key('deflect'):
                x = False
                _or = None 
                def pot_grad(args):
                    global x
                    i,theta = args
#                   if (i%100) == 0: 
#                       print 'Calculating srcdiff: % 5i/%5i\r' % (i+1, len(ploc)),;sys.stdout.flush(),
#                       x = True
                    deflect[i] = potential.grad(kappa,theta,ploc,cell_size)
                    if obj.shear:
                        s1,s2 = data['shear']
                        s = complex(s1*obj.shear.poten_dx(theta) + s2*obj.shear.poten_d2x(theta),
                                    s1*obj.shear.poten_dy(theta) + s2*obj.shear.poten_d2y(theta))
                        deflect[i] += s

                #parallel_map(pot_grad, enumerate(ploc), threads = 1, return_ = False)
                #parallel_map(pot_grad, enumerate(ploc), threads = env().ncpus, return_ = False)
                for i,theta in enumerate(ploc):
                    pot_grad((i,theta))
                    if i%100 == 0: 
                        print 'Calculating srcdiff: % 5i/%5i\r' % (i+1, len(ploc)),;sys.stdout.flush(),
                        x = True
                        #print 'Calculating srcdiff: % 6i/%6i\r' % (i+1, len(ploc)), ' '*40,;sys.stdout.flush() 

                if x: 
                    print ' '*40, '\r',

                data['deflect'] = deflect
    
            else:
                
                deflect = data['deflect']

            data['srcdiff'] = map(lambda s: abs(s[0] - ploc + deflect / s[1].zcap),
                                  izip(data['src'], obj.sources))

        return data['srcdiff'][src_index]

    #@memoize
    def deflect_grid(self, data, which, src_index):
        if not data.has_key('deflect'):
            self.srcdiff(data, src_index)

        if which == 'x':
            defl = data['deflect'].real
        elif which == 'y':
            defl = data['deflect'].imag
        else:
            assert False, "deflect_grid: 'which' must be one of 'x' or 'y'"

        return self._to_grid(defl)
        
    @memoize
    def srcdiff_grid(self, data):
        obj = self.myobject
        return [ self._to_grid(self.srcdiff(data, i)) for i,src in enumerate(obj.sources) ]

    def grid_to_grid(self, grid, grid_size, H0inv):
        o = self.myobject
        L = o.basis.pixrad
        r = grid.shape[0]
        J = r // 2
        #grid = congrid(grid, (2*L+1, 2*L+1), minusone=True)

        grid_cell_size = grid_size / r
        cell_size      = o.basis.top_level_cell_size

        #cell_size = grid_cell_size

        grid *= grid_cell_size**2

        pg = zeros((2*L+1, 2*L+1))
        for y in xrange(2*L+1):
            for x in xrange(2*L+1):
                rt = (L-y+0.5) * cell_size
                rb = (L-y-0.5) * cell_size
                cl = (x-L-0.5) * cell_size
                cr = (x-L+0.5) * cell_size
                for i in xrange(2*J+1):
                    for j in xrange(2*J+1):
                        it = (J-i+0.5) * grid_cell_size
                        ib = (J-i-0.5) * grid_cell_size
                        jl = (j-J-0.5) * grid_cell_size
                        jr = (j-J+0.5) * grid_cell_size
                        frac = intersect_frac([rt,rb,cl,cr], [it,ib,jl,jr])
                        pg[y,x] += frac * grid[i,j]
#                   if frac > 0:
#                       print [rt,rb,cl,cr], [it,ib,jl,jr]
#                       print frac, i,j, L,J, cell_size, grid_cell_size
                            #assert 0

                #break
            #break

        #pg *= 2
        pg /= cell_size**2
        pg *= convert('Msun/arcsec^2 to kappa',  1., o.dL, convert('H0^-1 in Gyr to nu', H0inv, o.dL))

        return pg

    def grid_mass(self, X,Y,M, H0inv, to_kappa=True):
        obj = self.myobject
        Rmap = self.mapextent
        Rpix = self.pixrad

        #return ones(len(self.insideL)+len(self.outsideL))

        cell_size      = self.top_level_cell_size
        #phys_cell_size = Arcsec_to_Kpc(obj, cell_size, H0inv)

        #-----------------------------------------------------------------------
        # Project the particles on to the 2d grid.
        # Note the order of X and Y in the histogram because Y maps to row
        # number and X to column number. Y is also inverted so that negative 
        # values are down.
        #-----------------------------------------------------------------------
        #rx = Arcsec_to_Kpc(obj, Rmap, H0inv)
        #ry = Arcsec_to_Kpc(obj, Rmap, H0inv)
        nu = convert('H0^-1 in Gyr to nu', H0inv)
        rx = convert('arcsec to kpc', Rmap, obj.dL, nu)
        ry = convert('arcsec to kpc', Rmap, obj.dL, nu)

        bins = 2*Rpix+1
        #binsX = self.ploc.real - cell_size
        #binsY = self.ploc.imag - cell_size
        #bins = [binsX, binsY]
        grid = histogram2d(-Y, X, bins=bins, weights=M, range=[[-ry,ry], [-rx,rx]])[0]

#       kernel = array([[1,4,7,4,1], 
#                       [4,16,26,16,4],
#                       [7,26,41,26,7],
#                       [4,16,26,16,4],
#                       [1,4,7,4,1]]
#                       ) / 273.
        
        #grid = convolve2d(grid, gauss_kernel(1), mode='same')
        #Log('Projected mass has been smooth with a 5x5 gaussian kernel.')

        #-----------------------------------------------------------------------
        # Convert physical units to internal units.
        #-----------------------------------------------------------------------
        phys_cell_size = convert('arcsec to ly', cell_size, obj.dL, nu)
        grid /= phys_cell_size**2
        #grid *= MsunKpc2_to_Kappa(obj, 1, nu)

        if to_kappa:
            grid *= convert('Msun/ly^2 to kappa', 1., obj.dL, nu)


        #figure()
        #matshow(grid, extent=[-Rmap,Rmap,-Rmap,Rmap])
        #over(contour, grid, 50, extent=[-Rmap,Rmap,-Rmap,Rmap])
        #show()

        return grid



    def solution_isothermal(self, theta_E, src=None, H0inv=None, shear=None, ptmass=None,
                            top_level_func_name='solution_isothermal()'):

        def gp(a,b):
            return b*log(hypot(a,b) + a) + a*log(hypot(a,b) + b) - b

        def g(a,b,j,k):
            return (gp(b,k) - gp(b,j)) - (gp(a,k) - gp(a,j))

#       def g(a,b, j,k):
#           return       log(a + sqrt(a**2 + j**2)) * j \
#                  + a * log(j + sqrt(a**2 + j**2))     \
#                  -     log(b + sqrt(b**2 + j**2)) * j \
#                  - b * log(j + sqrt(b**2 + j**2))     \
#                  -     log(a + sqrt(a**2 + k**2)) * k \
#                  - a * log(k + sqrt(a**2 + k**2))     \
#                  +     log(b + sqrt(b**2 + k**2)) * k \
#                  + b * log(k + sqrt(b**2 + k**2))

        nu = convert('H0^-1 in Gyr to nu', H0inv)

        dx = self.cell_size / 2.
        dy = dx
        gg = g(self.ploc.real-dx, self.ploc.real+dx, self.ploc.imag-dy, self.ploc.imag+dy)
        gg /= self.cell_size**2
        assert np.all(gg >= 0)
        return self.solution_from_array(0.5*theta_E * gg,
                                        src=src, H0inv=H0inv, shear=shear, ptmass=ptmass,
                                        top_level_func_name=top_level_func_name)

    def solution_from_array(self, a, src=None, H0inv=None, shear=None, ptmass=None,
                            top_level_func_name='solution_from_array()'):

        assert src   is not None, '%s: src keyword must be set.' % top_level_func_name
        assert len(src) != 0, '%s: src keyword must be set.' % top_level_func_name
        assert H0inv is not None, '%s: H0inv keyword must be set.' % top_level_func_name

        obj  = self.myobject

        #-----------------------------------------------------------------------
        # Now fill in the solution array.
        #-----------------------------------------------------------------------

        o = self.array_offset
        sol = zeros(obj.basis.nvar+1)

        sol[o+self.pix_start : o+self.pix_end] = a
        if shear:  sol[o+self.shear_start  : o+self.shear_end]  = shear
        if ptmass: sol[o+self.ptmass_start : o+self.ptmass_end] = ptmass

        for i,s in enumerate(src):
            assert isinstance(s, (list, tuple)) and len(s) == 2, \
                   "solution_from_grid(): Each element of src must be a 2 item list"
            offs = o+self.srcpos_start + 2*i
            #print '!'*80
            #print s
            sol[offs : offs+2] = s
            sol[offs : offs+2] += self.map_shift
            sol[offs : offs+2] *= obj.sources[i].zcap

        nu = convert('H0^-1 in Gyr to nu', H0inv)
        sol[o+self.H0] = nu
        #sol[o+self.H0] = time_to_internal(obj, H0inv)

        return sol
        #return self.solution_to_dict(sol)


    def solution_from_grid(self, grid, src=None, H0inv=None, shear=None, ptmass=None,
                           top_level_func_name='solution_from_grid'):
        return self.solution_from_array(grid.ravel()[self.insideL].take(self.pmap),
                                        src=src, H0inv=H0inv, shear=shear, ptmass=ptmass,
                                        top_level_func_name=top_level_func_name)

    def solution_from_data(self, X,Y,M, src=None, H0inv=None, shear=None, ptmass=None,
                           top_level_func_name='solution_from_data'):
        assert src   is not None, 'solution_from_data(): src keyword must be set.'
        assert H0inv is not None, 'solution_from_data(): H0inv keyword must be set.'

        grid = self.grid_mass(X,Y,M, H0inv)
        return self.solution_from_grid(grid, src=src, H0inv=H0inv, shear=shear, ptmass=ptmass,
                                       top_level_func_name=top_level_func_name)


if __name__ == "__main__":

    import sys
    sys.path.append('/Users/jonathan/GLASS/glass/')

    from pylab import contour, show

    pb = PixelBasis()
    
    pb.L = 10
    pb.cell_size = 1
    lnr = pb._lnr()
    print lnr.shape
    contour(lnr)
    show()

