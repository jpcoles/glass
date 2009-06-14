from __future__ import division
from environment import env
import numpy
from numpy import zeros, array, empty, cos, sin, compress
from potential import poten, poten_dx, poten_dy, maginv
from itertools import izip

all_priors = []
inc_priors = []
exc_priors = []

def _make_prior(f, where):
    class P: 
        def __init__(self, f, where): self.f, self.where = f, where
        def __eq__(self, f):
            return self.f == f
    all_priors.append(P(f, where))
    return all_priors[-1]

def object_prior(f):   return _make_prior(f, 'object_prior')
def ensemble_prior(f): return _make_prior(f, 'ensemble_prior')

def include_prior(f):
    assert not exc_priors, 'Cannot both include and exclude priors.'
    i = all_priors.index(f)
    inc_priors.append(all_priors[i])

def exclude_prior(f):
    assert not inc_priors, 'Cannot both include and exclude priors.'
    i = all_priors.index(f)
    exc_priors.append(all_priors[i])


##############################################################################

@object_prior
def image_pos(o, leq, eq, geq):
    print "Image Position"

    pix_start, pix_end = o.basis.pix_start, o.basis.pix_end
    srcpos_start, srcpos_end = o.basis.srcpos_start, o.basis.srcpos_end
    shear_start,  shear_end  = o.basis.shear_start,  o.basis.shear_end
    ptmass_start, ptmass_end = o.basis.ptmass_start, o.basis.ptmass_end

    for i,sys in enumerate(o.systems):
        for img in sys.images:
            rows = zeros((2, o.basis.nvar))
            print "\tposition", img.pos, o.basis.cell_size
            rows[0,0] = (img.pos.real + o.basis.map_shift) * sys.zcap
            rows[1,0] = (img.pos.imag + o.basis.map_shift) * sys.zcap
            positions = img.pos - o.basis.ploc
            rows[0,pix_start:pix_end] = -poten_dx(positions, o.basis.cell_size)
            rows[1,pix_start:pix_end] = -poten_dy(positions, o.basis.cell_size)

#           for n,offs in enumerate(xrange(shear_start, shear_end)):
#               rows[0,offs] = -o.shear.poten_dx(n+1, img.pos)
#               rows[1,offs] = -o.shear.poten_dy(n+1, img.pos)

            rows[0,shear_start] = -o.shear.poten_dx(img.pos)
            rows[1,shear_start] = -o.shear.poten_dy(img.pos)

            rows[0,shear_start+1] = -o.shear.poten_d2x(img.pos)
            rows[1,shear_start+1] = -o.shear.poten_d2y(img.pos)

#           for n,offs in enumerate(xrange(ptmass_start, ptmass_end)):
#               rows[0,offs] = -o.ptmass.poten_dx(n+1, img.pos)
#               rows[1,offs] = -o.ptmass.poten_dy(n+1, img.pos)

            srcpos = o.basis.srcpos_start + 2*i
            rows[0,srcpos:srcpos+2] = -sys.zcap,     0
            rows[1,srcpos:srcpos+2] =     0,     -sys.zcap

            eq(rows[0])
            eq(rows[1])


@object_prior
def time_delay(o, leq, eq, geq):
    print "Time Delay"

    H0   = o.basis.H0

    pix_start, pix_end = o.basis.pix_start, o.basis.pix_end
    shear_start,  shear_end  = o.basis.shear_start,  o.basis.shear_end
    ptmass_start, ptmass_end = o.basis.ptmass_start, o.basis.ptmass_end

    for i, sys in enumerate(o.systems):
        for img0,img1,[lower_delay,upper_delay] in sys.time_delays:

            row = zeros(o.basis.nvar)

            srcpos = o.basis.srcpos_start + 2*i

            x0, y0 = img0.pos.real, img0.pos.imag
            x1, y1 = img1.pos.real, img1.pos.imag

            # The constant term
            row[0] = (abs(img1.pos)**2 - abs(img0.pos)**2) / 2 + (x1-x0 + y1-y0)*o.basis.map_shift
            row[0] *= sys.zcap

            # The beta term
            row[srcpos:srcpos+2]  = x0-x1, y0-y1
            row[srcpos:srcpos+2] *= sys.zcap

            # The ln terms
            row[pix_start:pix_end] -= poten(img1.pos - o.basis.ploc, o.basis.cell_size)
            row[pix_start:pix_end] += poten(img0.pos - o.basis.ploc, o.basis.cell_size)

            for n,offs in enumerate(xrange(shear_start, shear_end)):
                row[offs] -= o.shear.poten(n+1, img1.pos)
                row[offs] += o.shear.poten(n+1, img0.pos)

#           for n,offs in enumerate(xrange(ptmass_start, ptmass_end)):
#               row[offs] -= o.ptmass.poten(n+1, img1.pos, o.basis.cell_size)
#               row[offs] += o.ptmass.poten(n+1, img0.pos, o.basis.cell_size)

            if [lower_delay, upper_delay] == [None, None]:
                row[H0] = 0
                eq(row)
            elif lower_delay == upper_delay:
                row[H0] = -lower_delay
                eq(row)
            elif lower_delay is None:
                row[H0] = -upper_delay
                leq(row)
            elif upper_delay is None:
                row[H0] = -lower_delay
                geq(row)
            else:
                row2 = row.copy()
                row2[H0] = -upper_delay
                row[H0]  = -lower_delay
                leq(row2)
                geq(row)

            #print row

@object_prior
def hubble_constant(o, leq, eq, geq):
    """This requires a particular hubble constant for the object."""
    print "Hubble Constant"
    on = False
    if env.h_spec is not None:
        row = zeros(o.basis.nvar)
        row[0] = env.h_spec / o.scales['time']
        row[o.basis.H0] = -1
        eq(row)
        on = True
    print "\t", on

@object_prior
def magnification(o, leq, eq, geq):

    print "Magnification"

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,     pix_end = o.basis.pix_start,   o.basis.pix_end
    shear_start, shear_end = o.basis.shear_start, o.basis.shear_end

    for sys in o.systems:
        for img in sys.images:
            k1, k2, eps = img.elongation
            parity = img.parity
            k2 = 1/k2

            rows = zeros((6, o.basis.nvar))
            rows[0,0] = [k1-1,  k1+1, -k1+1][parity] * sys.zcap
            rows[1,0] = [k2-1, -k2-1, -k2+1][parity] * sys.zcap
            rows[2,0] = [-eps,  -eps,   eps][parity] * sys.zcap
            rows[3,0] = [-eps,  -eps,   eps][parity] * sys.zcap
            rows[4,0] = [-eps,   eps,   eps][parity] * sys.zcap
            rows[5,0] = [-eps,   eps,   eps][parity] * sys.zcap

            xy,xx,yy = maginv(img.pos - o.basis.ploc, img.angle, o.basis.cell_size)
            if parity == MINIMUM:
                rows[0, pix_start:pix_end] = -k1*xx + yy
                rows[1, pix_start:pix_end] = -k2*yy + xx
                rows[2, pix_start:pix_end] =     xy + xx*eps
                rows[3, pix_start:pix_end] =    -xy + xx*eps
                rows[4, pix_start:pix_end] =     xy + yy*eps
                rows[5, pix_start:pix_end] =    -xy + yy*eps

            if parity == SADDLE:
                rows[0, pix_start:pix_end] = -k1*xx - yy
                rows[1, pix_start:pix_end] =  k2*yy + xx
                rows[2, pix_start:pix_end] =     xy + xx*eps
                rows[3, pix_start:pix_end] =    -xy + xx*eps
                rows[4, pix_start:pix_end] =     xy - yy*eps
                rows[5, pix_start:pix_end] =    -xy - yy*eps

            if parity == MAXIMUM:
                rows[0, pix_start:pix_end] =  k1*xx - yy
                rows[1, pix_start:pix_end] =  k2*yy - xx
                rows[2, pix_start:pix_end] =     xy - xx*eps
                rows[3, pix_start:pix_end] =    -xy - xx*eps
                rows[4, pix_start:pix_end] =     xy - yy*eps
                rows[5, pix_start:pix_end] =    -xy - yy*eps

            for n,offs in enumerate(xrange(shear_start, shear_end)):
                xy,xx,yy = o.shear.maginv(n+1, img.pos, img.angle)
                if parity == MINIMUM: 
                    rows[0, offs] = -k1*xx + yy
                    rows[1, offs] = -k2*yy + xx
                    rows[2, offs] =     xy + xx*eps
                    rows[3, offs] =    -xy + xx*eps
                    rows[4, offs] =     xy + yy*eps
                    rows[5, offs] =    -xy + yy*eps

                if parity == SADDLE:  
                    rows[0, offs] = -k1*xx - yy
                    rows[1, offs] =  k2*yy + xx
                    rows[2, offs] =     xy + xx*eps
                    rows[3, offs] =    -xy + xx*eps
                    rows[4, offs] =     xy - yy*eps
                    rows[5, offs] =    -xy - yy*eps

                if parity == MAXIMUM: 
                    rows[0, offs] =  k1*xx - yy
                    rows[1, offs] =  k2*yy - xx
                    rows[2, offs] =     xy - xx*eps
                    rows[3, offs] =    -xy - xx*eps
                    rows[4, offs] =     xy - yy*eps
                    rows[5, offs] =    -xy - yy*eps

            #print pix_start, pix_end
            #print "rows[:,0]", rows[:,0]

            for r in rows: leq(r)

@object_prior
def annular_density(o, leq, eq, geq):
    print "Annular density"
    on = False
    if o.kann_spec != 0:
        row = zeros(o.basis.nvar)
        for r in xrange(o.basis.inner_image_ring, o.basis.outer_image_ring):
            row[o.basis.rings[r]] = -1
            row[0] = kann_spec * len(o.basis.rings[r])
        eq(row)
        on = True

    print "\t", on
        

##############################################################################

@object_prior
def steepness(o, leq, eq, geq):
    print "Steepness" 

    if o.minsteep is None: return
    if o.maxsteep is None: return

    assert o.maxsteep >= o.minsteep

    pix_start, pix_end = o.basis.pix_start, o.basis.pix_end

    nrings = len(o.basis.rings)
    row = zeros(o.basis.nvar)

    #---------------------------------------------------------------------------
    # First handle the central pixel
    #---------------------------------------------------------------------------
    r0,r1 = o.basis.rings[0:2]
    row[pix_start+r0] =  1.0 / len(r0)
    row[pix_start+r1] = -1.0 / len(r1)
    #print r0,r1
    #print row
    c=1
    geq(row)


    #---------------------------------------------------------------------------
    # Now the rest of the rings.
    #---------------------------------------------------------------------------
    for l in xrange(1,nrings-1):
        r0 = o.basis.rings[l]
        r1 = o.basis.rings[l+1]

        if o.minsteep == o.maxsteep:
            row = zeros(o.basis.nvar)
            lc  = l ** o.minsteep
            lpc = -((l+1) ** o.minsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            #print r0,r1
            #print row
            eq(row)
        else:
            row = zeros(o.basis.nvar)
            lc  = l ** o.minsteep
            lpc = -((l+1) ** o.minsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            #print r0,r1
            #print row
            geq(row)

            row = zeros(o.basis.nvar)
            lc  = l ** o.maxsteep
            lpc = -((l+1) ** o.maxsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            leq(row)
            c += 1


#   print "\tmaxsteep=", o.maxsteep, "minsteep=",o.minsteep
#   if o.maxsteep > o.minsteep:
#       row = zeros(o.basis.nvar)
#       r0 = o.basis.rings[1]
#       r1 = o.basis.rings[-2]
#       lc  = -1
#       lpc =  nrings ** o.maxsteep
#       row[pix_start+r0] = lc  / len(r0)
#       row[pix_start+r1] = lpc / len(r1)
#       geq(row)
#       c += 1
    print "\tsteepness eqs =", c
        

@object_prior
def gradient(o, leq, eq, geq):
    print "Gradient"
    pix_start, pix_end = o.basis.pix_start, o.basis.pix_end

    cs = cos(o.cen_ang)
    sn = sin(o.cen_ang)
    c = 0
    for nbrs1,nbrs2,nmask,r,ri in o.basis.nbrs:
        if ri == o.basis.central_pixel: continue

        px = r.real
        py = r.imag
        x = cs*px - sn*py
        y = sn*px + cs*py

        row = zeros(o.basis.nvar)
        row[pix_start + nbrs1] = compress(nmask, [-x,x,-y,y])
        #print ": ", px,py,x,y,nbrs1#, nbrs2
        geq(row)
        c += 1
        if sn != 0:
            x =  cs*px + sn*py
            y = -sn*px + cs*py
            row = zeros(o.basis.nvar)
            row[pix_start + nbrs1] = compress(nmask, [-x,x,-y,y])
            geq(row)
            c += 1

    print "\tgradient eqs =", c
    print "\tsn=", sn

#@object_prior
def central_pixel_as_maximum(o, leq, eq, geq):
    cp = o.basis.central_pixel + o.basis.pix_start
    print "Central pixel as maximum", cp

    for i in xrange(o.basis.pix_start, o.basis.pix_end):
        if i == cp: continue
        row = zeros(o.basis.nvar, dtype=numpy.float32)
        row[cp] = 1
        row[i]  = -1
        geq(row)


@object_prior
def smoothness(o, leq, eq, geq):
    """The average of the neighbouring pixels can't be more than twice
       the value of a given pixel."""

    print "Smoothness"
    pix_start, pix_end = o.basis.pix_start, o.basis.pix_end

    c=0
    for nbrs1,nbrs2,nmask,r, ri in o.basis.nbrs:
        #-----------------------------------------------------------------------
        # Skip the central pixel. This allows any value of mass.
        # XXX: Some versions of PixeLens don't.
        #-----------------------------------------------------------------------
        #if ri == o.basis.central_pixel: continue

        row = zeros(o.basis.nvar)
        #print "N",
        #for n in nbrs1: print n,
        #print "|",
        #for n in nbrs2: print n,
        #print
        #if len(nbrs1): row[pix_start + nbrs1] = 1
        #if len(nbrs2): row[pix_start + nbrs2] = 1

        row[pix_start + nbrs1] = 1
        row[pix_start + nbrs2] = 1

        #-----------------------------------------------------------------------
        # FIXME: This line should really be
        #        row[pix_start + ri] = -(len(nbrs1)+len(nbrs2))/2.0
        # since there aren't always 8 neighbors (e.g. on the edge).
        #-----------------------------------------------------------------------
        #row[pix_start + ri] = -4
        row[pix_start + ri] = -(len(nbrs1)+len(nbrs2)) / 2

        geq(row)
        c += 1

    print "\tsmoothness eqs =", c

@object_prior
def external_shear(o, leq, eq, geq):
    print "External Shear"
    on = False
    for s in xrange(o.basis.shear_start, o.basis.shear_end):
        row = zeros(o.basis.nvar)
        row[0] =  0.1
        row[s] = -1
        geq(row)
        on = True

    print "\t", on

@ensemble_prior
def shared_h(objs, nvars, leq, eq, geq):
    """This requires that all objects have the same hubble constant."""
    print "Shared h"
    on = False
    for o1,o2 in izip(objs[:-1], objs[1:]):
        offs1 = o1.basis.array_offset
        offs2 = o2.basis.array_offset
        row = zeros(nvars)
        row[offs1 + o1.basis.H0] =  o1.scales['time']
        row[offs2 + o2.basis.H0] = -o2.scales['time']
        eq(row) 
        on = True
    print "\t", on

