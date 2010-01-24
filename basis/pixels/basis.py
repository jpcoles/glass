#
# Coordinates should be located at the centers of pixels
#
from __future__ import division
import sys
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jonathan/GLASS/glass/')

import numpy
from numpy import zeros, amin, amax, min, max, abs, vectorize, negative, array, take,   \
                  ndindex, empty, arange, empty_like, ogrid, round, where,  \
                  unique, round, argwhere, asarray, lexsort, angle, floor,  \
                  conj, arctan2, atleast_2d, linspace, cumsum, sum, repeat, \
                  zeros_like, ndenumerate, s_, isinf, where, dot, array, \
                  add, subtract, multiply, append, ceil, ones, sort, sign, diff, \
                  trunc, argmin, logical_and, logical_not, nan_to_num, histogram2d, \
                  sin, cos, pi, matrix, diag, average

import pylab
from pylab import plot, show, matshow, figure, contour, over, scatter, subplot
from matplotlib.patches import Circle, Rectangle

from scipy.ndimage.filters import correlate
from scipy.misc import central_diff_weights
from scipy.linalg import eig

import scipy.ndimage._ni_support as _ni_support
import scipy.ndimage._nd_image as _nd_image

from math import hypot, atan2, pi, sqrt
from itertools import izip

from environment import env
from potential import poten_indef, poten2d_indef, poten, poten_dx, poten_dy
from scales import density_to_physical, distance_to_physical, time_to_physical
from scales import density_to_internal, distance_to_internal, time_to_internal
from scales import MsunKpc2_to_Kappa, Kappa_to_MsunKpc2, Kappa_to_MsunArcsec2
from scales import Kpc_to_Arcsec, Arcsec_to_Kpc
from handythread import parallel_map2

from potential import poten_dxdx, poten_dydy, maginv, poten_dxdy

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

class PixelBasis: 

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

        self.map_shift = 10        # [arcsec]

        self.lnr = None
        self.subdivision = 5

        self.hiresR = 0
        self.hires_levels = None

        self.grad_rmax = 1
        #self.grad_rmax = sqrt(10.5)

    def init(self, obj):
        self.myobject = obj

        L = obj.basis.pixrad

        #---------------------------------------------------------------------
        # Get all image positions (except maximums near the center)
        #---------------------------------------------------------------------
        rs = [ abs(img.pos) for src in obj.sources for img in src.images if img.parity_name != 'max']

        print '=' * 80
        print 'PIXEL BASIS for %s' % obj.name
        print '=' * 80

        if rs:
            rmin, rmax = min(rs), max(rs)
            print "rmin =", rmin, "rmax =", rmax, " L =", L
        else:
            assert obj.maprad is not None, 'If no images are given, then maprad must be.'
            rmin, rmax = 0, obj.maprad

        del rs

        #---------------------------------------------------------------------
        # Select a maprad if not specified. Remember, coordinates are of
        # pixels centers.
        #---------------------------------------------------------------------
        self.maprad = obj.maprad
        if self.maprad is None:
            self.maprad = min([rmax+rmin, 2*rmax-rmin])

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
                          dtype=numpy.int32)

        self.int_ploc  = self.int_ploc.take(self.pmap)
        rkeys          = rkeys.take(self.pmap)
        self.int_cell_size = self.int_cell_size.take(self.pmap)

        #---------------------------------------------------------------------
        # Make neighbor lists
        #---------------------------------------------------------------------
        print 'Finding neighbors...'
        self.nbrs  = []
        self.nbrs2 = []

        if 1:
            self.nbrs = [ [i, r, argwhere(logical_and(abs(self.int_ploc-r) < 1.5*s, abs(self.int_ploc-r) > 0)).ravel()] 
                          for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size)) ]
#           self.nbrs = []
#           for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size)):
#               dist = abs(self.int_ploc-r)
#               self.nbrs.append([i, r, argwhere(logical_and(dist < 1.5*s, dist > 0)).ravel()]) 


#           print 'Finding neighbors...'
#           self.nbrs2 = [ [i,r, argwhere(abs(self.int_ploc[self.nbrs[i][2]]-r) < 1.5*s)] 
#                         for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size)) ]

#           print 'done!'
            self.nbrs2 = [ [i, r, argwhere(logical_and(abs(self.int_ploc-r) <= s, abs(self.int_ploc-r) > 0)).ravel()] 
                          for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size * self.grad_rmax)) ]
        else:
            print "SKIPPING NEIGHBOR FINDING!"

        #print self.nbrs

        #---------------------------------------------------------------------
        #---------------------------------------------------------------------

#       figure(figsize=(15,7))
#       sp=subplot(121)
        #plot(self.ploc.real, self.ploc.imag)
        #scatter(self.ploc.real, self.ploc.imag, s=(5)**2, marker='s')
#       scatter(self.int_ploc.real, self.int_ploc.imag, s=(5*self.int_cell_size)**2, marker='s')
#       for r,s in izip(self.ploc, self.cell_size):
#           sp.add_artist(Rectangle([r.real-s/2, r.imag-s/2], s,s, fill=False))
#           sp.add_artist(Circle([r.real, r.imag], radius=1.5*s, fill=False))
        #scatter(self.ploc.real, self.ploc.imag, s=(5*self.cell_size)**2, marker='s')
#       sp.set_aspect('equal')

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

        # XXX: Need these for the annular density prior
        #inner_image_ring = rmin // self.cell_size
        #outer_image_ring = rmax // self.cell_size + 1

        #---------------------------------------------------------------------
        #print self.rings
        print "%i rings:" % len(self.rings)
        for r in self.rings:
            print "  % 4i ...% 4i" % (r[0], r[-1])
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

        #---------------------------------------------------------------------
        # Now adjust the positions so that they are physical
        #---------------------------------------------------------------------
        self.rs                *= self.top_level_cell_size
        self.cell_size          = self.int_cell_size        * self.top_level_cell_size
        self.radial_cell_size   = self.int_radial_cell_size * self.top_level_cell_size
        self.ploc               = self.int_ploc * self.cell_size
        self.xy                *= self.top_level_cell_size


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

        print 'Pixel basis'
        print '    Pixel radius         = %i'  % self.pixrad
        print '    Map radius           = %.4f [arcsec]'  % self.maprad, 'Distance to center of outer pixel.'
        print '    Map Extent           = %.4f [arcsec]'  % self.mapextent, 'Distance to outer edge of outer pixel.'
        print '    top_level_cell_size  = %.4f [arcsec]'  % self.top_level_cell_size
        print '    Map radius g=14      = %.4f [kpc]'     % Arcsec_to_Kpc(obj,self.maprad,14)
        print '    Map Extent g=14      = %.4f [kpc]'     % Arcsec_to_Kpc(obj,self.mapextent,14)
        print '    top_level_cell g=14  = %.4f [kpc]'     % Arcsec_to_Kpc(obj, self.top_level_cell_size, 14)
        print '    Number of pixels     = %i'    % npix
        print '    Number of variables  = %i'    % self.nvar
        print '    Central pixel offset = %i'    % self.central_pixel
        print 'Offsets        % 5s  % 5s' % ('Start', 'End')
        print '    pix        % 5i  % 5i' % (self.pix_start, self.pix_end)
        print '    shear      % 5i  % 5i' % (self.shear_start,  self.shear_end)
        print '    ptmass     % 5i  % 5i' % (self.ptmass_start, self.ptmass_end)
        print '    srcpos     % 5i  % 5i' % (self.srcpos_start, self.srcpos_end)
        print '    H0         % 5i'       % (self.H0)

#       xy    = self.refined_xy_grid({})

#       print where(xy == obj.basis.ploc)
#       r,c=xy.shape
#       print xy[r//2]
#       print obj.basis.ploc

#       sys.exit(0)


    def packaged_solution(self, sol):
        obj    = self.myobject
        o      = self.array_offset
        scales = self.myobject.scales

        ps = {}
        ps['kappa']  = sol[ o+self.pix_start    : o+self.pix_end      ]

        #ps['kappa']  = zeros_like(sol[ o+self.pix_start    : o+self.pix_end      ])
        ps['shear']  = sol[ o+self.shear_start  : o+self.shear_end    ] \
                       if obj.shear else array([0,0])
        ps['ptmass'] = sol[ o+self.ptmass_start : o+self.ptmass_start ]
        ps['src'] = array([complex(sol[o+i], sol[o+i+1]) - complex(self.map_shift,self.map_shift)
                     for i in xrange(self.srcpos_start, self.srcpos_end,2)])

        Gyr = 1e9 * 365*60*60*24
        km = 1000
        Mpc = 3.086e22

        ps['H0']     = sol[o+self.H0] * (Mpc/km/Gyr)
        ps['1/H0']   = 1 / sol[o+self.H0]

        #print self.myobject.scales, self.H0, sol[o+self.H0], len(sol)
        #print self.srcpos_start
        #assert not isinf(ps['1/H0'])
        #if ps['1/H0'] == float('inf'): ps['1/H0'] = 1

        rscale = Arcsec_to_Kpc(obj, 1, ps['1/H0'])

        dscale1 = Kappa_to_MsunArcsec2(obj, 1, ps['1/H0'])
        dscale2 = Kappa_to_MsunKpc2   (obj, 1, ps['1/H0'])

        ps['R']     = self.rs + self.radial_cell_size / 2
        ps['R_kpc'] = ps['R'] * rscale

        ps['enckappa'] = cumsum([    sum(ps['kappa'][r])         for r in self.rings])
        #ps['enckappa'] = cumsum([    sum(ps['kappa'][r]*self.cell_size[r]**2)         for r in self.rings])
        ps['encmass']  = cumsum([    sum(ps['kappa'][r]*self.cell_size[r]**2)*dscale1 for r in self.rings])
        ps['sigma']    =  array([average(ps['kappa'][r]                     )*dscale2 for r in self.rings])

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
                    d.append(time_to_physical(tau-prev) * data['H0'])
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

    def _to_grid(self, a):
        L = self.pixrad
        reorder = empty_like(a)
        reorder.put(self.pmap, a)
        grid = zeros((2*L+1)**2)
        grid[self.insideL] = reorder
        grid = grid.reshape((2*L+1,2*L+1))
        return grid

    def mass_grid(self, data):
        print "WARNING: use of mass_grid is deprecated. Use kappa_grid instead."
        return self.kappa_grid(self, data)

    @memoize
    def kappa_grid(self, data):
        #print 'kappa_grid:', id(data)
        L = self.pixrad
        S = self.subdivision
        assert (S%2)==1

        grid = self._to_grid(data['kappa'])
        grid = repeat(grid, S, axis=1)
        grid = repeat(grid, S, axis=0)

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

    def XXpotential_grid(self, data):
        if not data.has_key('potential_grid'):
            obj = self.myobject
            L = obj.basis.pixrad
            S = obj.basis.subdivision
            assert (S%2)==1

            kappa  = self.kappa_grid(data)
            phi   = zeros_like(kappa)
            lnr   = self._lnr()
            lr,lc = lnr.shape
            mr,mc = kappa.shape

#           rscale = distance_to_physical([obj,data], 1) * self.top_level_cell_size
#           dscale = density_to_physical([obj,data], 1) * self.top_level_cell_size**2
#           print sum(kappa) * dscale
#           print 2.1*rscale
            print "kappa is", kappa.shape
            #print "phi is", phi.shape
            #print "lnr is", lnr.shape

            r0 = (lr + S) // 2
            c0 = (lc + S) // 2
            o = S*(2*L+1)

            w = zeros((S+1,S+1))
            w[0, 0] = w[-1,-1] =  1
            w[0,-1] = w[-1, 0] = -1

            if S == 1:
                crop = s_[1:,1:]
            else:
                crop = s_[(S+1)//2:-(S//2), (S+1)//2:-(S//2)]

            l = empty((o+S,o+S))
            _or = -1
            for [r,c],m in ndenumerate(kappa):
                if r != _or: 
                    _or=r; 
                    print '\rCalculating potential: %i/%i' \
                        % (r+1, kappa.shape[0]), ' '*40,;sys.stdout.flush()

                if not m: continue

                rs,re = r0-r, r0-r+o
                cs,ce = c0-c, c0-c+o

                s = lnr[rs - S : re, cs - S : ce]
                correlate(s, w, mode='constant', output=l)
                multiply(l, m, l)

                subtract(phi, l[crop], phi)
            print

            def f(r):
                print r
                rs,re = r0-r, r0-r+o

                phi_part = zeros_like(kappa)
                l        = empty((o+S,o+S))
                for c,m in enumerate(kappa[r]):
                    if not m: continue

                    cs,ce = c0-c, c0-c+o

                    w0 = w*m
                    s = lnr[rs - S : re, cs - S : ce]
                    fast_correlate(s, w0, mode='constant', output=l)
                    #multiply(l, m, l)

                    add(phi_part, l[crop], phi_part)

                del l
                return phi_part

            if obj.shear:
                xy   = self.refined_xy_grid(data)
                s1,s2 = data['shear']
                phi -= s1*obj.shear.poten(1, xy) + s2*obj.shear.poten(2, xy)
                
            data['potential_grid'] = phi
            print 'sum', sum(phi)

        return data['potential_grid']

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
        dist  = theta - self.ploc
        s = complex(sum(kappa * nan_to_num(poten_dx(dist,self.cell_size))),
                    sum(kappa * nan_to_num(poten_dy(dist,self.cell_size))))
        if obj.shear:
            s1,s2 = data['shear']
            s += complex(s1*obj.shear.poten_dx(theta) + s2*obj.shear.poten_d2x(theta),
                         s1*obj.shear.poten_dy(theta) + s2*obj.shear.poten_d2y(theta))
        return s

    def srcdiff(self, data, src_index):
        if not data.has_key('srcdiff'):
            obj = self.myobject

            kappa   = data['kappa']
            deflect = empty_like(self.ploc)
            dist    = empty_like(self.ploc)
            ploc    = self.ploc

#           kappa = self.kappa_grid(data).ravel()
#           ploc    = self.refined_xy_grid(data).ravel()
#           dist    = empty_like(ploc)

            _or = None 
            for i,theta in enumerate(ploc):
                subtract(theta, ploc, dist)
                deflect[i] = complex(sum(kappa * nan_to_num(poten_dx(dist,self.cell_size))),
                                     sum(kappa * nan_to_num(poten_dy(dist,self.cell_size))))
                if obj.shear:
                    s1,s2 = data['shear']
                    s = complex(s1*obj.shear.poten_dx(theta) + s2*obj.shear.poten_d2x(theta),
                                s1*obj.shear.poten_dy(theta) + s2*obj.shear.poten_d2y(theta))
                    deflect[i] += s

                if i%100 == 0: 
                    print '\rCalculating srcdiff: %i/%i' % (i+1, len(ploc)), ' '*40,;sys.stdout.flush()
            print

            data['srcdiff'] = map(lambda s: abs(s[0] - ploc + deflect / s[1].zcap),
                                  izip(data['src'], obj.sources))

            data.setdefault('deflect', deflect)

        return data['srcdiff'][src_index]

    @memoize
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
        return map(lambda i,src: self._to_grid(self.srcdiff(data, i)), 
                   enumerate(obj.sources))

    @memoize
    def maginv(self, data):
        mag_inv = []
        obj = self.myobject

# B1115 Iterations
        M = [[ (-0.93447255300632492414, -0.35603517758480146549),
               (0.74714385909724179147, -0.66466236076167328761),
               (0.98996250756997195008, -0.14133022891714641656),
               (0.79116942416136204486,  0.6115970423997967309)]]

        M = [[ (-0.99973532389059382997,  0.02300613321202587061),
               (0.84458748009647599542, -0.53541758326028543458),
               (0.97789726566482826353, -0.20908595793417703024),
               (0.87275959025376470546,  0.48815028179862884894) ]]

        M = [[ (-0.93454685931096703744, -0.35584008733138494662),
               (0.74792424441611915498, -0.66378409487918377252),
               (0.98990206260351198964, -0.14175297687637017363),
               (0.78633424730469037023,  0.61780130423605139622) ]]

        M = [[ (-0.93403178581837331684, -0.35718989778679405278),
               (0.74726696353443322618, -0.66452395382711970662),
               (0.98991661792316720803, -0.14165129565082729135),
               (0.7862352941716893584,  0.6179272305043345126) ]]

# D0_0211 iterations

        if 0:
            M = [[ (0.8817900298449217189,   0.47164217714925804614),
                   (0.95420449518838390546, -0.29915511254578502909) ],
                 [ (-0.99975063472780190121 ,-0.02233088357761452841),
                   (-0.96364089936189545327, -0.26720070560722203323) ]]

            M = [[ (0.99840037595348163357,  0.05653927215614497348),
                   (0.99783543748806946549, -0.06576047211655966507) ],
                 [ (-0.99814953668160477651,  0.06080709187502369995),
                   (0.88598184776783817096, -0.46371992131661438075) ]]

#               M = [[ (0.99998158084853316918, -0.00606942861138636339),
#                      (0.99811561515346303164, -0.06136137862551745148) ],
#                    [ (-0.99987787876445743951, -0.01562778159205616135),
#                      (-0.77811691937027627208, -0.62811946299228049728) ]]

        for i,src in enumerate(obj.sources):
            l = []
            for j,img in enumerate(src.images):
                d = img.pos - self.ploc
                c,s = M[i][j]

                #xy,xx,yy = maginv(d, img.angle, self.cell_size)
                xx = src.zcap - sum(data['kappa'] * poten_dxdx(d,self.cell_size))
                yy = src.zcap - sum(data['kappa'] * poten_dydy(d,self.cell_size))
                xy =        0 - sum(data['kappa'] * poten_dxdy(d,self.cell_size))
                #theta = img.angle * pi/180
                #c = cos(theta)
                #s = sin(theta)
                P = matrix([[c,s], [-s,c]])
                A = matrix([[xx, xy], [xy, yy]])
                D = P.I*A*P

                if 1:
                    c2  = c ** 2
                    s2  = s ** 2

                    xx = sum(data['kappa'] * poten_dxdx(d,self.cell_size))
                    yy = sum(data['kappa'] * poten_dydy(d,self.cell_size))
                    xy = sum(data['kappa'] * poten_dxdy(d,self.cell_size))
                    alpha = src.zcap*(c**2+s**2) - c2*xx - s2*yy + 2*s*c*xy
                    beta  = src.zcap*(c**2+s**2) - s2*xx - c2*yy - 2*s*c*xy
                    delta =                        s*c*(yy-xx) - xy*(c2-s2)

                if 0:
                    c_2 = cos(2*theta)
                    s_2 = sin(2*theta)
                    c2  = cos(theta) ** 2
                    s2  = sin(theta) ** 2

                    xx = sum(data['kappa'] * poten_dxdx(d,self.cell_size))
                    yy = sum(data['kappa'] * poten_dydy(d,self.cell_size))
                    xy = sum(data['kappa'] * poten_dxdy(d,self.cell_size))
                    
                    alpha = src.zcap +  s_2*xy - c2*xx - s2*yy
                    beta  = src.zcap +  -s_2*xy - c2*yy - s2*xx
                    delta =             s_2*(yy-xx)/2 - c_2*xy

                eval,evec = eig(A)
                fs = [0,0,0, [A,D,alpha,beta,delta, eval, evec,P, (delta**2/(alpha*beta))]]


#                   P=matrix(evec)
#                   D=diag(eval)
#                   A = P * D * P.I
#                   fs = [0,0,0, [m,A]]
#                   fs = [src.zcap*(1+sin(2*img.angle*pi/180)) - sum(data['kappa'] * xx),
#                         src.zcap*(1+sin(2*img.angle*pi/180)) - sum(data['kappa'] * yy),
#                         src.zcap*(0+cos(2*img.angle*pi/180)) - sum(data['kappa'] * xy)]
                l.append(fs)
            mag_inv.append(l)
        return mag_inv

    def grid_mass(self, X,Y,M, H0inv):
        obj = self.myobject
        Rmap = self.mapextent
        Rpix = self.pixrad

        cell_size      = self.top_level_cell_size
        phys_cell_size = distance_to_physical(obj, cell_size, H0inv)

        #-----------------------------------------------------------------------
        # Project the particles on to the 2d grid.
        # Note the order of X and Y in the histogram because Y maps to row
        # number and X to column number. Y is also inverted so that negative 
        # values are down.
        #-----------------------------------------------------------------------
        rx = distance_to_physical(obj, Rmap, H0inv)
        ry = distance_to_physical(obj, Rmap, H0inv)

        grid = histogram2d(-Y, X, bins=2*Rpix+1, weights=M, range=[[-ry,ry], [-rx,rx]])[0]

        #-----------------------------------------------------------------------
        # Convert physical units to internal units.
        #-----------------------------------------------------------------------
        grid /= phys_cell_size**2
        grid *= MsunKpc2_to_Kappa(obj, 1, H0inv)
        #matshow(grid, extent=[-Rmap,Rmap,-Rmap,Rmap])
        #over(contour, grid, 50, extent=[-Rmap,Rmap,-Rmap,Rmap])

        return grid

    def solution_from_array(self, a, src=None, H0inv=None, shear=None, ptmass=None):

        assert src   is not None, 'solution_from_grid(): src keyword must be set.'
        assert H0inv is not None, 'solution_from_grid(): H0inv keyword must be set.'

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
            sol[offs : offs+2] = s
            sol[offs : offs+2] += self.map_shift

        sol[o+self.H0] = 1 / H0inv
        #sol[o+self.H0] = time_to_internal(obj, H0inv)

        return self.packaged_solution(sol)


    def solution_from_grid(self, grid, src=None, H0inv=None, shear=None, ptmass=None):
        return self.solution_from_array(grid.ravel()[self.insideL].take(self.pmap),
                                        src=src, H0inv=H0inv, shear=shear, ptmass=ptmass)

    def solution_from_data(self, X,Y,M, src=None, H0inv=None, shear=None, ptmass=None):
        assert src   is not None, 'solution_from_data(): src keyword must be set.'
        assert H0inv is not None, 'solution_from_data(): H0inv keyword must be set.'

        grid = self.grid_mass(X,Y,M, H0inv)
        return self.solution_from_grid(grid, src=src, H0inv=H0inv, shear=shear, ptmass=ptmass)


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

