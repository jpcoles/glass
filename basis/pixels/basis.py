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
                  add, subtract, multiply
from pylab import plot, show, matshow, figure, contour, over

from scipy.ndimage.filters import correlate
from scipy.misc import central_diff_weights

from math import hypot, atan2, pi
from itertools import izip

from environment import env
from potential import poten_indef, poten2d_indef, poten
from scales import density_to_physical, distance_to_physical, time_to_physical
from scales import density_to_internal, time_to_internal

def xy_grid(L, S=1, scale=1):
    """Return a grid with radius L (i.e. diameter=2*L+1) where each
       of the cells are subdivided into S parts. The result is scaled
       by 'scale'."""
    gx = linspace(-L,L, (2*L+1) * S) * scale
    gy = atleast_2d(-gx).T
    return vectorize(complex)(gx, gy)

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

    def init(self, obj):
        self.myobject = obj

        L = obj.basis.pixrad

        #---------------------------------------------------------------------
        # Get all image positions
        #---------------------------------------------------------------------
        rs = [ abs(img.pos) for sys in obj.systems for img in sys.images ]

        if rs:
            rmin, rmax = min(rs), max(rs)
            print "rmin =", rmin, "rmax =", rmax, " L =", L
        else:
            assert obj.maprad is not None, 'If no images are given, then maprad must be.'
            rmin, rmax = 0, obj.maprad

        #---------------------------------------------------------------------
        # Select a maprad if not specified. Remember, coordinates are of
        # pixels centers.
        #---------------------------------------------------------------------
        self.maprad = obj.maprad
        if self.maprad is None:
            self.maprad = min([rmax+rmin, 2*rmax-rmin])

        #self.cell_size = (2*self.maprad) / (2*L+1)
        self.cell_size = self.maprad / L
        #self.cell_size = (L+1) * self.maprad / (L*L)
        #self.maprad = self.cell_size * L # + self.cell_size / 2
        #self.cell_size = 1

        self.mapextent = self.cell_size * (2*L + 1)/2

        print "cell_size =", self.cell_size

        #---------------------------------------------------------------------
        # Create pixel map -- symm not supported
        #---------------------------------------------------------------------
        #gy,gx = ogrid[-(L+1):L+2, -(L+1):L+2]
        #gy,gx = ogrid[-L:L+1, -L:L+1]

#       gx = linspace(-L,L, 2*L+1)
#       gy = atleast_2d(linspace(-L,L, 2*L+1)).T
#       xy = self.xy = vectorize(complex)(gx, gy)

        xy = self.xy = xy_grid(L)

        #print xy
        rs = round(abs(xy))
        insideL = self.insideL = argwhere(rs.ravel() <= L).ravel()
        #print insideL
        self.ploc = xy.ravel()[insideL]
        #print self.ploc

        #---------------------------------------------------------------------
        # By sorting by the arctan2() and abs() of the positions we create
        # a spiral from the center out. The central pixel is the first element
        # in the mass array.
        #---------------------------------------------------------------------
        self.pmap = array(lexsort(keys = (arctan2(self.ploc.imag, self.ploc.real), round(abs(self.ploc)))),
                          copy = False).astype(numpy.int32)
        #print self.pmap
        #sys.exit(0)
        self.ploc = self.ploc.take(self.pmap)
        #print self.ploc

#       plot(self.pmap, arange(len(self.pmap)))
#       show()

#       for i,p in enumerate(self.ploc):
#           print "I", i+1, int(p.real), int(p.imag)

#       plot(self.ploc.real, -self.ploc.imag)
#       show()

        npix = len(self.ploc)
        print "npix =", npix

        #---------------------------------------------------------------------
        # Make neighbor lists and pixel map
        #---------------------------------------------------------------------
        def the_nbrs(r):
            """Helper function"""
            # This can't be the best way...
            n = [] #zeros(len(self.ploc), 'bool')
            nmask = []
            for rr in \
                [ 1 + 0j,   # right
                 -1 + 0j,   # left
                  0 + 1j,   # up
                  0 - 1j]:  # down
                w = argwhere(self.ploc == r+rr).ravel().tolist()
                n += w
                nmask.append(len(w) != 0)

            n2 = []
            for rr in \
                [ 1 + 1j,   # upper right
                 -1 - 1j,   # upper left
                  1 - 1j,   # lower right
                 -1 + 1j]:  # lower left
                w = argwhere(self.ploc == r+rr).ravel().tolist()
                n2 += w

                #n |= self.ploc == r + rr

            #n = argwhere(n).ravel()
            assert len(n)+len(n2) > 0, "r=%s has no neighbors!" % r
#            print r
            ri = argwhere(self.ploc == r)
            return [array(n,  dtype=numpy.int32), 
                    array(n2, dtype=numpy.int32), nmask,r,ri]
            
        #self.nbrs = [ the_nbrs(r) for r in self.ploc ]
        self.nbrs = [ the_nbrs(r) for r in xy.ravel()[insideL] ]

        # Useful for some contraints
        self.central_pixel = argwhere(self.ploc == 0).item()
        print "central_pixel =", self.central_pixel

        print "nbrs =", reduce(lambda s,x: s + len(x[0])+len(x[1]), self.nbrs, 0)

        #---------------------------------------------------------------------
        # Each pixel gets assigned a ring...
        #---------------------------------------------------------------------
        ringlvl = round(abs(self.ploc))

        #---------------------------------------------------------------------
        # Now make lists of all the pixels on a given ring
        #---------------------------------------------------------------------
        self.rings = [ argwhere(ringlvl == i).ravel() for i in unique(ringlvl) ]
        print "%i rings:" % len(self.rings)
        for r in self.rings:
            print "  % 4i ...% 4i" % (r[0], r[-1])

        inner_image_ring = rmin // self.cell_size
        outer_image_ring = rmax // self.cell_size + 1

        #---------------------------------------------------------------------
        # Now adjust the positions so that they are physical
        #---------------------------------------------------------------------
        self.ploc *= self.cell_size
        #print self.ploc

        #---------------------------------------------------------------------
        # Setup variable offsets for the constraint arrays
        #---------------------------------------------------------------------
        self.pix_start,    self.pix_end    = 1, npix+1
        self.shear_start,  self.shear_end  = self.pix_end,   self.pix_end+2*(obj.shear is not None)
        self.ptmass_start, self.ptmass_end = self.shear_end, self.shear_end

        #nimgs=0
        #for sys in obj.systems:
            #nimgs += len(sys.images)

        #print "nimgs =", nimgs

        self.srcpos_start,    self.srcpos_end    = self.ptmass_end, self.ptmass_end+2*len(obj.systems)
        self.H0 = self.srcpos_end

        self.nvar = self.H0 + 1

        print "Pixel basis"
        print "    npix=%i  maprad=% .4f nvars=%i" % (npix, self.maprad, self.nvar-1)
        print "Offsets        % 5s  % 5s" % ("Start", "End")
        print "    pix        % 5i  % 5i" % (self.pix_start, self.pix_end)
        print "    shear      % 5i  % 5i" % (self.shear_start,  self.shear_end)
        print "    ptmass     % 5i  % 5i" % (self.ptmass_start, self.ptmass_end)
        print "    srcpos     % 5i  % 5i" % (self.srcpos_start, self.srcpos_end)
        print "    H0         % 5i"       % (self.H0)

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
        ps['mass']   = sol[ o+self.pix_start    : o+self.pix_end      ]
        ps['shear']  = sol[ o+self.shear_start  : o+self.shear_end    ]
        ps['ptmass'] = sol[ o+self.ptmass_start : o+self.ptmass_start ]
        ps['src']    = sol[ o+self.srcpos_start : o+self.srcpos_end   ] - self.map_shift
        ps['1/H0']   = time_to_physical(obj, sol[o+self.H0])

        #print self.myobject.scales, self.H0, sol[o+self.H0], len(sol)
        #print self.srcpos_start
        #assert not isinf(ps['1/H0'])
        #if ps['1/H0'] == float('inf'): ps['1/H0'] = 1

        rscale = distance_to_physical([obj,ps], 1) * self.cell_size
        mscale = density_to_physical([obj,ps], 1) * self.cell_size**2

        ps['R']     = (arange(len(self.rings)) + 0.5) * self.cell_size
        ps['encmass'] = cumsum([sum(ps['mass'][r]) for r in self.rings])
        ps['sigma'] = array([sum(ps['mass'][r]) / len(r) for r in self.rings]) \
                    / self.cell_size**2

        ps['R_phys'] = (arange(len(self.rings)) + 0.5) * rscale
        ps['encmass_phys'] = ps['encmass'] * mscale
        ps['sigma_phys'] = array([sum(ps['mass'][r]) / len(r) for r in self.rings]) \
                         * mscale/rscale**2

        return ps

    def refined_xy_grid(self, data):
        if not data.has_key('refined_xy_grid'):
            L = self.pixrad
            S = self.subdivision
#           xy = xy_grid(L+S//2, S, self.cell_size)

            Q = S*(2*L+1) // 2
            gx = arange(-Q, Q+1, 1) * (self.cell_size/S)
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
            mass = data['mass']

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
            gx = arange(-Q+0.5,Q+0.5, 1) * (self.cell_size/S)
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

            rscale = distance_to_physical([obj,data], 1) * self.cell_size
            mscale = density_to_physical([obj,data], 1) * self.cell_size**2
            print sum(mass) * mscale
            print 2.1*rscale
            print "mass is", mass.shape
            #print "phi is", phi.shape
            #print "lnr is", lnr.shape

            r0 = (lr + S) // 2
            c0 = (lc + S) // 2
            o = S*(2*L+1)

            w = zeros((S+1,S+1))
            w[0, 0] = w[-1,-1] =  1
            w[0,-1] = w[-1, 0] = -1

            crop = s_[(S+1)//2:-(S//2), (S+1)//2:-(S//2)]

            l = empty((o+S,o+S))
            _or = -1
            for [r,c],m in ndenumerate(mass):
                if r != _or: _or=r; print r
                if not m: continue

                rs,re = r0-r, r0-r+o
                cs,ce = c0-c, c0-c+o

                s = lnr[rs - S : re, cs - S : ce]
                correlate(s, w, mode='constant', output=l)
                multiply(l, m, l)

                subtract(phi, l[crop], phi)
                
            # TODO: need xy grid for this
            #if obj.shear: phi -= obj.shear.poten(1, img.pos) + obj.shear.poten(2, img.pos)

            print phi

            data['potential_grid'] = phi
            print 'sum', sum(phi)

        return data['potential_grid']

    def potential_contour_levels(self, data):
        if not data.has_key('potential_contour_levels'):
            obj = self.myobject
            data['potential_contour_levels'] = []
            for i,sys in enumerate(obj.systems):
                l = []
                for img in sys.images:
                    p  = dot(data['mass'], poten(img.pos - obj.basis.ploc, obj.basis.cell_size))
                    # TODO:
                    #p += obj.shear and obj.shear.poten(1, img.pos) + obj.shear.poten(2, img.pos)
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
            for i,sys in enumerate(obj.systems):
                srcx,srcy = data['src'][i*2:i*2+2]
                geom  = r2_2 - xy.real * srcx - xy.imag * srcy
                grid  = geom * sys.zcap + phi
                #print phi
                #print geom
                data['arrival_grid'].append(grid)

            #print 'arrival_grid:', sum(grid)

        return data['arrival_grid']

    def arrival_contour_levels(self, data):

        if not data.has_key('arrival_contour_levels'):
            obj = self.myobject
            data['arrival_contour_levels'] = []
            for i,sys in enumerate(obj.systems):
                srcx,srcy = data['src'][i*2:i*2+2]
                l = []
                for img in sys.images:
                    if img.parity_name != 'sad': continue

                    geom  = abs(img.pos)**2 / 2 - img.pos.real * srcx - img.pos.imag * srcy

                    p  = dot(data['mass'], poten(img.pos - obj.basis.ploc, obj.basis.cell_size))

                    # TODO:
                    #p += obj.shear and obj.shear.poten(1, img.pos) + obj.shear.poten(2, img.pos)

                    l.append(geom * sys.zcap - p)
                if l: data['arrival_contour_levels'].append(l)

        return data['arrival_contour_levels']

    def solution_from_data(self, X,Y,M, src=None, shear=None, ptmass=None, H0inv=None):

        obj = self.myobject
        Rmap = obj.maprad
        Rpix = self.pixrad

        if H0inv is None: H0inv = 1/env.h_spec

        phys_cell_size = distance_to_physical(obj, self.cell_size, H0inv)
        grid = zeros((2*Rpix+1, 2*Rpix+1))

        gxmin = distance_to_physical(obj, -(Rmap+self.cell_size/2), H0inv)
        gxmax = distance_to_physical(obj,  (Rmap+self.cell_size/2), H0inv)
        gymin = distance_to_physical(obj, -(Rmap+self.cell_size/2), H0inv)
        gymax = distance_to_physical(obj,  (Rmap+self.cell_size/2), H0inv)

        for x,y,m in izip(X,Y,M):

            if not gxmin <= x < gxmax: continue
            if not gymin <= y < gymax: continue

            i = grid.shape[0]-1 - int((y - gymin) / phys_cell_size) # invert y
            j = int((x - gxmin) / phys_cell_size)
            grid[i,j] += m

        grid /= phys_cell_size**2
        grid *= density_to_internal(obj, 1, H0inv)

        #matshow(grid, extent=[-Rmap,Rmap,-Rmap,Rmap])
        #over(contour, grid, 50, extent=[-Rmap,Rmap,-Rmap,Rmap])

        o = self.array_offset
        sol = zeros(obj.basis.nvar+1)

        sol[o+self.pix_start : o+self.pix_end] = grid.ravel()[self.insideL].take(self.pmap)
        if shear:  sol[o+self.shear_start  : o+self.shear_end]  = shear
        if ptmass: sol[o+self.ptmass_start : o+self.ptmass_end] = ptmass
        if src:   
             sol[o+self.srcpos_start : o+self.srcpos_end] = src
             sol[o+self.srcpos_start : o+self.srcpos_end] += self.map_shift
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

