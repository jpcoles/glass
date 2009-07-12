#
# Coordinates should be located at the centers of pixels
#
from __future__ import division
import sys
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jonathan/GLASS/glass/')

import numpy
from numpy import zeros, min, max, abs, vectorize, negative, array, take,   \
                  ndindex, empty, arange, empty_like, ogrid, round, where,  \
                  unique, round, argwhere, asarray, lexsort, angle, floor,  \
                  conj, arctan2, atleast_2d, linspace, cumsum, sum, repeat, \
                  zeros_like, ndenumerate, s_, isinf, where, dot, array, \
                  add, subtract, multiply, append, ceil, ones, sort, sign, diff, trunc, argmin, logical_and, nan_to_num
import pylab
from pylab import plot, show, matshow, figure, contour, over, scatter, subplot
from matplotlib.patches import Circle, Rectangle

from scipy.ndimage.filters import correlate
from scipy.misc import central_diff_weights

import scipy.ndimage._ni_support as _ni_support
import scipy.ndimage._nd_image as _nd_image

from math import hypot, atan2, pi, sqrt
from itertools import izip

from environment import env
from potential import poten_indef, poten2d_indef, poten
from scales import density_to_physical, distance_to_physical, time_to_physical
from scales import density_to_internal, distance_to_internal, time_to_internal
from scales import MsunKpc2_to_KappaArcsec2, KappaArcsec2_to_MsunKpc2, KappaArcsec2_to_MsunArcsec2
from scales import Kpc_to_Arcsec, Arcsec_to_Kpc
from handythread import parallel_map2

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

        self.map_shift = 100        # [arcsec]

        self.lnr = None
        self.subdivision = 5

        self.hiresR = 0
        self.hires_levels = None

    def init(self, obj):
        self.myobject = obj

        L = obj.basis.pixrad

        #---------------------------------------------------------------------
        # Get all image positions
        #---------------------------------------------------------------------
        rs = [ abs(img.pos) for src in obj.sources for img in src.images ]

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
        insideL = self.insideL = argwhere(mags < (2*L+1)/2).T[0]
        self.int_ploc      = self.xy[insideL]
        self.int_cell_size = self.int_cell_size[insideL]

        rkeys = array([self.rs[argmin(abs(abs(p)-self.rs))] for p in self.int_ploc])

        #---------------------------------------------------------------------
        # By sorting by the arctan2() and abs() of the positions we create a
        # spiral from the center out. The central pixel is the first element in
        # the mass array.
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

        if 0:
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
                          for i,[r,s] in enumerate(izip(self.int_ploc, self.int_cell_size)) ]

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


#       for n in self.nbrs2[1][2]:
#           r = self.int_ploc[n]
#           s = self.int_cell_size[n]
#           sp.add_artist(Rectangle([r.real-s/2, r.imag-s/2], s,s, fill=False))
            #sp.add_artist(Circle([r.real, r.imag], radius=1.5*s, fill=False))

        #print "...", self.radial_cell_size

        #---------------------------------------------------------------------
        # Setup variable offsets for the constraint arrays
        #---------------------------------------------------------------------
        self.pix_start,    self.pix_end    = 1, npix+1
        self.shear_start,  self.shear_end  = self.pix_end,   self.pix_end+2*(obj.shear is not None)
        self.ptmass_start, self.ptmass_end = self.shear_end, self.shear_end

        #nimgs=0
        #for src in obj.sources:
            #nimgs += len(src.images)

        #print "nimgs =", nimgs

        self.srcpos_start,    self.srcpos_end    = self.ptmass_end, self.ptmass_end+2*len(obj.sources)
        self.H0 = self.srcpos_end

        self.nvar = self.H0 + 1

        print 'Pixel basis'
        print '    Pixel radius         = %i'  % self.pixrad
        print '    Map radius           = %.4f [arcsec]'  % self.maprad, 'Distance to center of outer pixel.'
        print '    Map Extent           = %.4f [arcsec]'  % self.mapextent, 'Distance to outer edge of outer pixel.'
        print '    top_level_cell_size  = %.4f [arcsec]'  % self.top_level_cell_size
        print '    Map radius g=14      = %.4f [kpc]'     % Arcsec_to_Kpc(obj,self.maprad,14)
        print '    Map Extent g=14      = %.4f [kpc]'     % Arcsec_to_Kpc(obj,self.mapextent,14)
        print '    top_level_cell g=14  = %.4f [kpc]'     % Arcsec_to_Kpc(obj, self.top_level_cell_size, 14)
        print '    Number of pixels     = %i'    % npix
        print '    Number of variables  = %i'    % (self.nvar-1)
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
        obj = self.myobject
        o = obj.basis.array_offset
        scales = self.myobject.scales

        ps = {}
        ps['kappa']  = sol[ o+self.pix_start    : o+self.pix_end      ]
        ps['shear']  = sol[ o+self.shear_start  : o+self.shear_end    ]
        ps['ptmass'] = sol[ o+self.ptmass_start : o+self.ptmass_start ]
        ps['src']    = sol[ o+self.srcpos_start : o+self.srcpos_end   ] - self.map_shift
        ps['H0']     = sol[o+self.H0]
        ps['1/H0']   = time_to_physical(obj, sol[o+self.H0])

        #print self.myobject.scales, self.H0, sol[o+self.H0], len(sol)
        #print self.srcpos_start
        #assert not isinf(ps['1/H0'])
        #if ps['1/H0'] == float('inf'): ps['1/H0'] = 1

        #rscale = distance_to_physical([obj,ps], 1)
        rscale = Arcsec_to_Kpc([obj,ps], 1)
        #dscale1 =  density_to_physical([obj,ps], 1)

        dscale1 = KappaArcsec2_to_MsunArcsec2([obj,ps], 1)
        dscale2 = KappaArcsec2_to_MsunKpc2([obj,ps], 1)

#       print ps['1/H0']
#       print rscale, dscale 
        
        ps['R'] = self.rs + self.radial_cell_size / 2
        ps['enckappa'] = cumsum([sum(ps['kappa'][r]) for r in self.rings])
        #ps['sigma']    =  array([sum(ps['kappa'][r]/self.cell_size[r]**2) / len(r) for r in self.rings])

#       print ps['R']
#       print ps['encmass']
#       print ps['sigma']

        ps['R_kpc']    = ps['R'] * rscale
        ps['encmass']  = cumsum([sum(ps['kappa'][r]*dscale1*self.cell_size[r]**2) for r in self.rings])
        ps['sigma']    =  array([sum(ps['kappa'][r]*dscale2) / len(r)             for r in self.rings])

        #print len(ps['R_phys']), len(ps['encmass_phys']), len(ps['sigma_phys'])

#       print ps['R_phys']
#       print ps['encmass_phys']
#       print ps['sigma_phys']

#       print '****************************************'

        return ps

    def time_delays(self, data):

        if not data.has_key('time_delays'):

            obj  = self.myobject

            pix_start,    pix_end    = self.pix_start,    self.pix_end
            shear_start,  shear_end  = self.shear_start,  self.shear_end
            ptmass_start, ptmass_end = self.ptmass_start, self.ptmass_end

            data['time_delays'] = []
            for i, src in enumerate(obj.sources):

                s = 2*i
                srcx, srcy = data['src'][s:s+2] # + self.map_shift

                prev = 0
                for j,img in enumerate(src.images):

                    r    = img.pos
                    x, y = r.real, r.imag

                    # The constant term
                    tau  = abs(r)**2 / 2 #+ (x + y)*self.map_shift
                    tau -= x * srcx
                    tau -= y * srcy
                    tau *= src.zcap

                    # The ln terms
                    tau -= sum(data['kappa'] * poten(img.pos - self.ploc, self.cell_size))

                    if obj.shear:
                        tau -= data['shear'][0] * obj.shear.poten(1, r) 
                        tau -= data['shear'][1] * obj.shear.poten(2, r)

                    if j > 0:
                        data['time_delays'].append((tau-prev) / data['H0'])
                    prev = tau

        return data['time_delays']

    def refined_xy_grid(self, data):
        if not data.has_key('refined_xy_grid'):
            L = self.pixrad
            S = self.subdivision
#           xy = xy_grid(L+S//2, S, self.cell_size)

            Q = S*(2*L+1) // 2
            gx = arange(-Q, Q+1, 1) * (self.top_level_cell_size/S)
            assert (len(gx) % 2) == 1

#           print gx
#           print self.ploc
#           print len(gx)
#           sys.exit(0)
            gy = atleast_2d(-gx).T
            xy = vectorize(complex)(gx, gy)
            data['refined_xy_grid'] = xy
        return data['refined_xy_grid']

    def mass_grid(self, data):
        if not data.has_key('mass_grid'):
            obj = self.myobject
            mass = data['kappa']

            L = obj.basis.pixrad
            S = obj.basis.subdivision
            assert (S%2)==1

            reorder = empty_like(mass)
            #print "********"
            #print max(obj.basis.pmap), len(mass)
            reorder.put(obj.basis.pmap, mass)

            grid = zeros((2*L+1)**2)
            grid[obj.basis.insideL] = reorder
            grid = grid.reshape((2*L+1,2*L+1))
            grid = repeat(grid, S, axis=1)
            grid = repeat(grid, S, axis=0)

#           sx, sy = data['src'][0*2:0*2+2]
#           #sx , sy = 10,10
#           r = int(-S * (sy / self.cell_size) + S*(2*L+1)/2)
#           c = int(S * (sx / self.cell_size) + S*(2*L+1)/2)

#           print "!", c, sx, sx / self.cell_size
#           print "!", r, sy, sy / self.cell_size

            #grid[r,c] = -5

            grid /= S**2
            #grid[grid == 0] = 1000
            #grid[grid != 1000] = 0
            #grid[grid != 0] = 10
            data['mass_grid'] = grid

        return data['mass_grid']

    def _lnr(self):
        """ Returns a grid of the indefinite integral of the potential. """
        if self.lnr is None:
            L, S = self.pixrad, self.subdivision
            assert (S%2)==1

            Q = 2 * S * (L + 1)
            gx = arange(-Q+0.5,Q+0.5, 1) * (self.top_level_cell_size/S)
            assert (len(gx) % 2) == 0
            gy = atleast_2d(gx).T
            xy = vectorize(complex)(gx, gy)
            #print xy

            self.lnr = poten_indef(xy)
            #print 'lnr', self.lnr[-1]

        return self.lnr

    def XXXpotential_grid(self, data):
        if not data.has_key('potential_grid'):
            obj = self.myobject
            L = obj.basis.pixrad
            S = obj.basis.subdivision
            assert (S%2)==1

            mass  = self.mass_grid(data)
            phi   = zeros_like(mass)
            lnr   = self._lnr()
            lr,lc = lnr.shape
            mr,mc = mass.shape

            #print "mass is", mass.shape
            #print "phi is", phi.shape
            #print "lnr is", lnr.shape

            r0 = (lr + S) // 2
            c0 = (lc + S) // 2
            o = S*(2*L+1)

            l = empty_like(mass)
            for [r,c],m in ndenumerate(mass):
                if not m: continue

                rs,re = r0-r, r0-r+o
                cs,ce = c0-c, c0-c+o

                s0 = s_[rs      : re    , cs      : ce    ]
                s1 = s_[rs - S  : re - S, cs - S  : ce - S]
                s2 = s_[rs - S  : re - S, cs      : ce    ]
                s3 = s_[rs      : re    , cs - S  : ce - S]

                #print rs,cs
                #print s0, s1, s2, s3
                add(lnr[s0],lnr[s1], l)
                subtract(l, lnr[s2], l)
                subtract(l, lnr[s3], l)
                multiply(l, m, l)
                subtract(phi, l, phi) #phi -= l
                
            # TODO: need xy grid for this
            #if obj.shear: phi -= obj.shear.poten(1, img.pos) + obj.shear.poten(2, img.pos)

            data['potential_grid'] = phi
            print 'sum', sum(phi)

            print phi

        return data['potential_grid']

    def potential_grid(self, data):
        if not data.has_key('potential_grid'):
            obj = self.myobject
            L = obj.basis.pixrad
            S = obj.basis.subdivision
            assert (S%2)==1

            mass  = self.mass_grid(data)
            phi   = zeros_like(mass)
            lnr   = self._lnr()
            lr,lc = lnr.shape
            mr,mc = mass.shape

#           rscale = distance_to_physical([obj,data], 1) * self.top_level_cell_size
#           dscale = density_to_physical([obj,data], 1) * self.top_level_cell_size**2
#           print sum(mass) * dscale
#           print 2.1*rscale
            print "mass is", mass.shape
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
            for [r,c],m in ndenumerate(mass):
                if r != _or: 
                    _or=r; 
                    print '\rCalculating potential: %i/%i' \
                        % (r+1, mass.shape[0]), ' '*40,;sys.stdout.flush()

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

                phi_part = zeros_like(mass)
                l        = empty((o+S,o+S))
                for c,m in enumerate(mass[r]):
                    if not m: continue

                    cs,ce = c0-c, c0-c+o

                    w0 = w*m
                    s = lnr[rs - S : re, cs - S : ce]
                    fast_correlate(s, w0, mode='constant', output=l)
                    #multiply(l, m, l)

                    add(phi_part, l[crop], phi_part)

                del l
                return phi_part

#           for phi_part in parallel_map2(f, arange(mass.shape[0]), threads=env().ncpus):
#               subtract(phi, phi_part, phi)
                
            if obj.shear:
                phi -= obj.shear.poten(1, img.pos) + obj.shear.poten(2, img.pos)

            data['potential_grid'] = phi
            print 'sum', sum(phi)

        return data['potential_grid']

    def potential_contour_levels(self, data):
        if not data.has_key('potential_contour_levels'):
            obj = self.myobject
            data['potential_contour_levels'] = []
            for i,src in enumerate(obj.sources):
                l = []
                for img in src.images:
                    p  = dot(data['kappa'], poten(img.pos - obj.basis.ploc, obj.basis.top_level_cell_size))
                    if obj.shear:
                        p += obj.shear.poten(1, img.pos) + obj.shear.poten(2, img.pos)
                    l.append(-p)
                if l: data['potential_contour_levels'].append(l)

        return data['potential_contour_levels']

    def arrival_grid(self, data):
        if not data.has_key('arrival_grid'):
            obj = self.myobject
            L = obj.basis.pixrad
            S = obj.basis.subdivision

            phi  = self.potential_grid(data)
            xy   = self.refined_xy_grid(data)
            r2_2 = abs(xy)**2 / 2
            
            data['arrival_grid'] = []
            for i,src in enumerate(obj.sources):
                srcx,srcy = data['src'][i*2:i*2+2]
                geom  = r2_2 - xy.real * srcx - xy.imag * srcy
                grid  = geom * src.zcap + phi
                #print phi
                #print geom


                data['arrival_grid'].append(grid)

            #print 'arrival_grid:', sum(grid)

        return data['arrival_grid']

    def srcdiff_grid(self, data):
        if not data.has_key('srcdiff_grid'):
            obj = self.myobject
            L = obj.basis.pixrad
            S = obj.basis.subdivision

            kappa   = data['kappa']
            deflect = zeros_like(self.ploc)
            dist    = empty_like(self.ploc)

            data['srcdiff_grid'] = []
            _or = None 
            for i,xy1 in enumerate(self.ploc):
                print i
                print '\rCalculating srcdiff: %i/%i' \
                    % (i+1, len(self.ploc)), ' '*40,;sys.stdout.flush()
                #dist = self.ploc-xy1
                #subtract(self.ploc, xy1, dist)
                subtract(xy1, self.ploc, dist)
                deflect[i] = sum(kappa * nan_to_num(dist/abs(dist)**2))
                print deflect[i]


            for i,src in enumerate(obj.sources):
                srcx,srcy = data['src'][i*2:i*2+2]
                #grid = abs(complex(srcx,srcy) - (self.ploc - deflect/src.zcap))
                grid = abs(complex(srcx,srcy) - (self.ploc - deflect/src.zcap))

                reorder = empty_like(grid)
                reorder.put(self.pmap, grid)

                sd = zeros((2*L+1)**2)
                sd[self.insideL] = reorder
                sd = sd.reshape((2*L+1,2*L+1))

                data['srcdiff_grid'].append(sd)
                #data['srcdiff_grid'].append(deflect)

        return data['srcdiff_grid']

    def arrival_contour_levels(self, data):

        if not data.has_key('arrival_contour_levels'):
            obj = self.myobject
            data['arrival_contour_levels'] = []
            for i,src in enumerate(obj.sources):
                srcx,srcy = data['src'][i*2:i*2+2]
                l = []
                for img in src.images:
                    if img.parity_name != 'sad': continue

                    geom  = abs(img.pos)**2 / 2 - img.pos.real * srcx - img.pos.imag * srcy

                    p  = dot(data['kappa'], poten(img.pos - obj.basis.ploc, obj.basis.top_level_cell_size))

                    # TODO:
                    #p += obj.shear and obj.shear.poten(1, img.pos) + obj.shear.poten(2, img.pos)

                    l.append(geom * src.zcap - p)
                if l: data['arrival_contour_levels'].append(l)

        return data['arrival_contour_levels']

    def solution_from_data(self, X,Y,M, src=None, H0inv=None, shear=None, ptmass=None):

        assert src   is not None, 'solution_from_data(): src keyword must be set.'
        assert H0inv is not None, 'solution_from_data(): H0inv keyword must be set.'

        obj = self.myobject
        Rmap = self.mapextent
        Rpix = self.pixrad

        cell_size = self.top_level_cell_size
        phys_cell_size = distance_to_physical(obj, cell_size, H0inv)
        grid = zeros((2*Rpix+1, 2*Rpix+1))

        #-----------------------------------------------------------------------
        # Project the particles on to the 2d grid.
        #-----------------------------------------------------------------------
        gxmax = distance_to_physical(obj, Rmap, H0inv)
        gxmin = -gxmax
        gymax = distance_to_physical(obj, Rmap, H0inv)
        gymin = -gymax

        for x,y,m in izip(X,Y,M):

            if not gxmin < x < gxmax: continue
            if not gymin < y < gymax: continue

            i = grid.shape[0]-1 - int((y - gymin) / phys_cell_size) # invert y XXX:really?
            j = int((x - gxmin) / phys_cell_size)
            grid[i,j] += m

        #-----------------------------------------------------------------------
        # Convert physical units to internal units.
        #-----------------------------------------------------------------------
        grid /= phys_cell_size**2
        grid *= MsunKpc2_to_KappaArcsec2(obj, 1, H0inv)

        #matshow(grid, extent=[-Rmap,Rmap,-Rmap,Rmap])
        #over(contour, grid, 50, extent=[-Rmap,Rmap,-Rmap,Rmap])

        #-----------------------------------------------------------------------
        # Now fill in the solution array.
        #-----------------------------------------------------------------------
        o = self.array_offset
        sol = zeros(obj.basis.nvar+1)

        sol[o+self.pix_start : o+self.pix_end] = grid.ravel()[self.insideL].take(self.pmap)
        if shear:  sol[o+self.shear_start  : o+self.shear_end]  = shear
        if ptmass: sol[o+self.ptmass_start : o+self.ptmass_end] = ptmass

        for i,s in enumerate(src):
            offs = o+self.srcpos_start + 2*i
            sol[offs : offs+2] = s
            sol[offs : offs+2] += self.map_shift

        sol[o+self.H0] = time_to_internal(obj, H0inv)

        return self.packaged_solution(sol)


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

