from __future__ import division
from environment import env
import numpy
from numpy import zeros, array, empty, cos, sin, compress, sign, logical_or, sort
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

    for i,src in enumerate(o.sources):
        for img in src.images:
            rows = zeros((2, o.basis.nvar))
            print "\tposition", img.pos #, o.basis.cell_size
            rows[0,0] = (img.pos.real + o.basis.map_shift) * src.zcap
            rows[1,0] = (img.pos.imag + o.basis.map_shift) * src.zcap
            positions = img.pos - o.basis.ploc
            rows[0,pix_start:pix_end] = -poten_dx(positions, o.basis.cell_size)
            rows[1,pix_start:pix_end] = -poten_dy(positions, o.basis.cell_size)

#           for n,offs in enumerate(xrange(shear_start, shear_end)):
#               rows[0,offs] = -o.shear.poten_dx(n+1, img.pos)
#               rows[1,offs] = -o.shear.poten_dy(n+1, img.pos)

            if o.shear:
                rows[0,shear_start] = -o.shear.poten_dx(img.pos)
                rows[1,shear_start] = -o.shear.poten_dy(img.pos)

                rows[0,shear_start+1] = -o.shear.poten_d2x(img.pos)
                rows[1,shear_start+1] = -o.shear.poten_d2y(img.pos)

#           for n,offs in enumerate(xrange(ptmass_start, ptmass_end)):
#               rows[0,offs] = -o.ptmass.poten_dx(n+1, img.pos)
#               rows[1,offs] = -o.ptmass.poten_dy(n+1, img.pos)

            srcpos = o.basis.srcpos_start + 2*i
            rows[0,srcpos:srcpos+2] = -src.zcap,     0
            rows[1,srcpos:srcpos+2] =     0,     -src.zcap

            eq(rows[0])
            eq(rows[1])


@object_prior
def time_delay(o, leq, eq, geq):
    print "Time Delay"

    H0   = o.basis.H0

    pix_start, pix_end = o.basis.pix_start, o.basis.pix_end
    shear_start,  shear_end  = o.basis.shear_start,  o.basis.shear_end
    ptmass_start, ptmass_end = o.basis.ptmass_start, o.basis.ptmass_end

    for i, src in enumerate(o.sources):
        for img0,img1,delay in src.time_delays:

            row = zeros(o.basis.nvar)

            srcpos = o.basis.srcpos_start + 2*i

            x0, y0 = img0.pos.real, img0.pos.imag
            x1, y1 = img1.pos.real, img1.pos.imag

            # The constant term
            row[0] = (abs(img1.pos)**2 - abs(img0.pos)**2) / 2 + (x1-x0 + y1-y0)*o.basis.map_shift
            row[0] *= src.zcap

            # The beta term
            row[srcpos:srcpos+2]  = x0-x1, y0-y1
            row[srcpos:srcpos+2] *= src.zcap

            # The ln terms
            row[pix_start:pix_end] -= poten(img1.pos - o.basis.ploc, o.basis.cell_size)
            row[pix_start:pix_end] += poten(img0.pos - o.basis.ploc, o.basis.cell_size)

            for n,offs in enumerate(xrange(shear_start, shear_end)):
                row[offs] -= o.shear.poten(n+1, img1.pos)
                row[offs] += o.shear.poten(n+1, img0.pos)

#           for n,offs in enumerate(xrange(ptmass_start, ptmass_end)):
#               row[offs] -= o.ptmass.poten(n+1, img1.pos, o.basis.cell_size)
#               row[offs] += o.ptmass.poten(n+1, img0.pos, o.basis.cell_size)

            if len(delay) == 1:
                d = delay[0]
                if d is None: row[H0] =  0; geq(row)
                else:         row[H0] = -d;  eq(row)
            else:
                l,u = delay
                if   l is None: row[H0] = -u; leq(row)
                elif u is None: row[H0] = -l; geq(row)
                else:
                    row2 = row.copy()
                    row[H0]  = -l; geq(row)
                    row2[H0] = -u; leq(row2)

            #print row

@object_prior
def hubble_constant(o, leq, eq, geq):
    """This requires a particular hubble constant for the object."""
    print "Hubble Constant"
    on = False

    if len(env().h_spec) == 1:
        h = env().h_spec[0]
        row = zeros(o.basis.nvar)
        row[0] = h / o.scales['time']
        row[o.basis.H0] = -1
        eq(row)
    else:
        lb, ub = env().h_spec
        if ub is not None:
            row = zeros(o.basis.nvar)
            row[0] = ub / o.scales['time']
            row[o.basis.H0] = -1
            leq(row)

        if lb is not None:
            row = zeros(o.basis.nvar)
            row[0] = lb / o.scales['time']
            row[o.basis.H0] = -1
            geq(row)

#       row = zeros(o.basis.nvar)
#       row[0] = env().h_spec / o.scales['time']
#       row[o.basis.H0] = -1
#       eq(row)
        on = True
    print "\t", on

@object_prior
def magnification(o, leq, eq, geq):

    print "Magnification"

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,     pix_end = o.basis.pix_start,   o.basis.pix_end
    shear_start, shear_end = o.basis.shear_start, o.basis.shear_end

    for src in o.sources:
        for img in src.images:
            k1, k2, eps = img.elongation
            parity = img.parity
            k2 = 1/k2

            rows = zeros((6, o.basis.nvar))
            rows[0,0] = [k1-1,  k1+1, -k1+1][parity] * src.zcap
            rows[1,0] = [k2-1, -k2-1, -k2+1][parity] * src.zcap
            rows[2,0] = [-eps,  -eps,   eps][parity] * src.zcap
            rows[3,0] = [-eps,  -eps,   eps][parity] * src.zcap
            rows[4,0] = [-eps,   eps,   eps][parity] * src.zcap
            rows[5,0] = [-eps,   eps,   eps][parity] * src.zcap

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

    if o.steep is None: return

    minsteep, maxsteep = o.steep

    assert maxsteep >= minsteep

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

        if minsteep == maxsteep:
            row = zeros(o.basis.nvar)
            lc  = l ** minsteep
            lpc = -((l+1) ** minsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            #print r0,r1
            #print row
            eq(row)
        else:
            row = zeros(o.basis.nvar)
            lc  = l ** minsteep
            lpc = -((l+1) ** minsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            #print r0,r1
            #print row
            geq(row)

            row = zeros(o.basis.nvar)
            lc  = l ** maxsteep
            lpc = -((l+1) ** maxsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            leq(row)
            c += 1


#   print "\tmaxsteep=", maxsteep, "minsteep=",minsteep
#   if maxsteep > minsteep:
#       row = zeros(o.basis.nvar)
#       r0 = o.basis.rings[1]
#       r1 = o.basis.rings[-2]
#       lc  = -1
#       lpc =  nrings ** maxsteep
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
    for ri,r,nbrs in o.basis.nbrs2:
        if ri == o.basis.central_pixel: continue

        px = r.real
        py = r.imag
        x = cs*px - sn*py
        y = sn*px + cs*py

        xx = r - o.basis.int_ploc[nbrs]
        yy = sign(xx.real) + 1j*sign(xx.imag)
        xx = x*sign(xx.real) + y*sign(xx.imag)
        assert logical_or(abs(yy) == 1, yy == 0).all(), yy

        #minxy = min(x,y)
        #xx[abs(xx) < abs(minxy)] = 0

        #print xx

        row = zeros(o.basis.nvar)
        row[pix_start + nbrs] = xx.astype(float)

        geq(row)
        c += 1
        if sn != 0:
            x =  cs*px + sn*py
            y = -sn*px + cs*py
            #minxy = min(x,y)
            #xx[abs(xx) < abs(minxy)] = 0
            xx = r - o.basis.int_ploc[nbrs]
            yy = sign(xx.real) + 1j*sign(xx.imag)
            xx = x*sign(xx.real) + y*sign(xx.imag)
            assert logical_or(abs(yy) == 1, yy == 0).all(), yy
            row = zeros(o.basis.nvar)
            row[pix_start + nbrs] = xx
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
    for i,r,nbrs in o.basis.nbrs:
        #-----------------------------------------------------------------------
        # Skip the central pixel. This allows any value of mass.
        # XXX: Some versions of PixeLens don't.
        #-----------------------------------------------------------------------
        if i == o.basis.central_pixel: continue

        row = zeros(o.basis.nvar)
        row[pix_start + nbrs] = 1
        row[pix_start + i]    = -len(nbrs) / 2

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

