from __future__ import division
import numpy
from numpy import zeros, min, max, abs, vectorize, negative, array, take,   \
                  ndindex, empty, arange, empty_like, ogrid, round, where,  \
                  unique, round, argwhere, asarray, lexsort, angle, floor,  \
                  conj, arctan2, atleast_2d, linspace, cumsum, sum
from math import hypot, atan2, pi
from pylab import plot, show

class PixelBasis: 

    myobject = None

    L         = 0
    maprad    = None
    cell_size = 0

    samplex_row_offset = 0

    nvar = 0
    pix_start,    pix_end    = 0,0
    shear_start,  shear_end  = 0,0
    ptmass_start, ptmass_end = 0,0
    srcpos_start, srcpos_end = 0,0
    timedelay_start, timedelay_end = 0,0
    H0 = 0

    inner_image_ring, outer_image_ring = 0,0

    map_shift = 10

    def init(self, obj):
        self.myobject = obj

        param = obj.basis
        L = param.L

        #---------------------------------------------------------------------
        # Get all image positions
        #---------------------------------------------------------------------
        rs = [ abs(img.pos) for sys in obj.systems for img in sys.images ]
        rmin, rmax = min(rs), max(rs)

        print "rmin =", rmin, "rmax =", rmax, " L =", L

        #---------------------------------------------------------------------
        # Select a maprad if not specified
        #---------------------------------------------------------------------
        self.maprad = obj.maprad
        if self.maprad is None:
            r              = min([rmax+rmin, 2*rmax-rmin])
            self.cell_size = (L+1) * r / L**2
            self.maprad    = self.cell_size * L
        else:
            self.cell_size = (L+1) * self.maprad / L**2

        print "cell_size =", self.cell_size

        #---------------------------------------------------------------------
        # Create pixel map -- symm not supported
        #---------------------------------------------------------------------
        #gy,gx = ogrid[-(L+1):L+2, -(L+1):L+2]
        #gy,gx = ogrid[-L:L+1, -L:L+1]

        gx = linspace(-L,L, 2*L+1)
        gy = atleast_2d(linspace(-L,L, 2*L+1)).T

        xy = self.xy = vectorize(complex)(gx, gy)
        #print xy
        rs = round(abs(xy))
        insideL = self.insideL = argwhere(rs.ravel() <= L).ravel()
        print insideL
        self.ploc = xy.ravel()[insideL]

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
                [
                         1,
                  -1,     
                           1j,       
                          -1j,
                                    ]:
                w = argwhere(self.ploc == r+rr).ravel().tolist()
                n += w
                nmask.append(len(w) != 0)

            n2 = []
            for rr in \
                [ 
                  1+1j,
                -1-1j,         
                1-1j,
                  -1+1j
                  ]:

                w = argwhere(self.ploc == r+rr).ravel().tolist()
                n2 += w

                #n |= self.ploc == r + rr

            #n = argwhere(n).ravel()
            assert len(n)+len(n2) > 0, "r=%s has no neighbors!" % r
#            print r
            ri = argwhere(self.ploc == r)
            return [array(n), array(n2), nmask,r,ri]
            
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
        self.timedelay_start, self.timedelay_end = self.srcpos_end, self.srcpos_end

        self.H0 = self.timedelay_end

        self.nvar = self.H0 + 1

        print "Pixel basis"
        print "    npix=%i  maprad=% .4f nvars=%i" % (npix, self.maprad, self.nvar-1)
        print "Offsets        % 5s  % 5s" % ("Start", "End")
        print "    pix        % 5i  % 5i" % (self.pix_start, self.pix_end)
        print "    shear      % 5i  % 5i" % (self.shear_start,  self.shear_end)
        print "    ptmass     % 5i  % 5i" % (self.ptmass_start, self.ptmass_end)
        print "    srcpos     % 5i  % 5i" % (self.srcpos_start, self.srcpos_end)
        print "    timedelay  % 5i  % 5i" % (self.timedelay_start, self.timedelay_end)
        print "    H0         % 5i"       % (self.H0)

    def packaged_solution(self, obj, sol):
        o = obj.basis.array_offset
        ps = {}
        ps['mass']   = sol[ o+self.pix_start    : o+self.pix_end      ]
        ps['shear']  = sol[ o+self.shear_start  : o+self.shear_end    ]
        ps['ptmass'] = sol[ o+self.ptmass_start : o+self.ptmass_start ]
        ps['src']    = sol[ o+self.srcpos_start : o+self.srcpos_end   ] - self.map_shift
        ps['H0']     = sol[ o+self.H0]

        #ps['sigma']   = array([len(r) * self.cell_size**2 for r in self.rings], numpy.float32)
        ps['encmass'] = cumsum([sum(ps['mass'][r]) for r in self.rings])
        #ps['sigma'] = cumsum([sum(ps['mass'][r]) / (len(r) * self.cell_size**2)) for r in self.rings])
        ps['sigma'] = [sum(ps['mass'][r]) / (len(r) * self.cell_size**2) for r in self.rings]

        return ps

    def mass_to_grid(self, mass):
        obj = self.myobject

        L = obj.basis.L

        reorder = empty_like(mass)
        print "********"
        print max(obj.basis.pmap), len(mass)
        reorder.put(obj.basis.pmap, mass)

        grid = zeros((2*L+1)**2)
        grid[obj.basis.insideL] = reorder
        grid = grid.reshape((2*L+1,2*L+1))

        return 2*L+1, 2*L+1, grid

