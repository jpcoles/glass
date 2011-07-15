from __future__ import division
from environment import env, command
import numpy
from numpy import zeros, array, empty, cos, sin, compress, sign, logical_or, sort, pi, log10, radians, argwhere, all, dot, sum
from potential import poten, poten_dx, poten_dy, poten_dxdx, poten_dydy, maginv, maginv_new, poten_dxdy, maginv_new4, maginv_new5
from itertools import izip
from log import log as Log
from scales import convert

from basis import neighbors

all_priors = []
def_priors = []
inc_priors = []
exc_priors = []
acc_objpriors = []
acc_enspriors = []

def _make_prior(f, where):
    class P: 
        def __init__(self, f, where): 
            self.f, self.check, self.where = f, None, where
        def __eq__(self, f):
            return f in [self.f, self.f.__name__]
        def __call__(self, *args, **kwargs):
            return f(*args, **kwargs)
    #print f
    all_priors.append(P(f, where))
    return all_priors[-1]

def object_prior(f):   return  _make_prior(f, 'object_prior')
def ensemble_prior(f): return  _make_prior(f, 'ensemble_prior')
def default_prior(p):  
    def_priors.append(p)
    return p

def object_prior_check(check): 
    def x(f):
        all_priors[all_priors.index(check)].check = f
    return x

@command
def include_prior(*f):
    #assert not exc_priors, 'Cannot both include and exclude priors.'
    for p in f:
        i = all_priors.index(p)
        inc_priors.append(all_priors[i])

@command
def exclude_prior(*f):
    #assert not inc_priors, 'Cannot both include and exclude priors.'
    for p in f:
        i = all_priors.index(p)
        exc_priors.append(all_priors[i])


##############################################################################

def new_row(obj, n=1):
    if n == 1: 
        return zeros(1+obj.basis.nvar)
    else:
        return zeros((n,1+obj.basis.nvar))

##############################################################################


@default_prior
@object_prior
def image_pos(o, leq, eq, geq):
    Log( "Image Position" )

    b = o.basis

    pix_start, pix_end = 1+b.pix_start, 1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    for i,src in enumerate(o.sources):
        for img in src.images:
            rows = new_row(o, 2)
            Log( "\tposition %s" % img.pos ) #, b.cell_size
            rows[0,0] = (img.pos.real + b.map_shift) * src.zcap
            rows[1,0] = (img.pos.imag + b.map_shift) * src.zcap
            positions = img.pos - b.ploc
            rows[0,pix_start:pix_end] = -poten_dx(positions, b.cell_size)
            rows[1,pix_start:pix_end] = -poten_dy(positions, b.cell_size)

            if o.shear:
                rows[0,shear_start+0] = -o.shear.poten_dx(img.pos)
                rows[0,shear_start+1] = -o.shear.poten_d2x(img.pos)

                rows[1,shear_start+0] = -o.shear.poten_dy(img.pos)
                rows[1,shear_start+1] = -o.shear.poten_d2y(img.pos)

#           for n,offs in enumerate(xrange(ptmass_start, ptmass_end)):
#               rows[0,offs] = -o.ptmass.poten_dx(n+1, img.pos)
#               rows[1,offs] = -o.ptmass.poten_dy(n+1, img.pos)

            srcpos = srcpos_start + 2*i
            rows[0,srcpos:srcpos+2] = -1,  0
            rows[1,srcpos:srcpos+2] =  0, -1

            eq(rows[0])
            eq(rows[1])

@object_prior_check(image_pos)
def check_image_pos(o, sol):
    Log( "Check Image Position (' ':-12  '.':-11  '-':-10  '*':-9)" )
    b    = o.basis
    offs = b.array_offset

    pix_start,    pix_end    = offs+b.pix_start,    offs+b.pix_end
    srcpos_start, srcpos_end = offs+b.srcpos_start, offs+b.srcpos_end
    shear_start,  shear_end  = offs+b.shear_start,  offs+b.shear_end
    ptmass_start, ptmass_end = offs+b.ptmass_start, offs+b.ptmass_end

    for i,src in enumerate(o.sources):
        res = ''
        for img in src.images:
            r0 = (img.pos.real + b.map_shift) * src.zcap
            r1 = (img.pos.imag + b.map_shift) * src.zcap
            positions = img.pos - b.ploc
            r0 -= sum(sol[pix_start:pix_end] * poten_dx(positions, b.cell_size))
            r1 -= sum(sol[pix_start:pix_end] * poten_dy(positions, b.cell_size))

            if o.shear:
                r0 -= sol[shear_start] * o.shear.poten_dx(img.pos)
                r1 -= sol[shear_start] * o.shear.poten_dy(img.pos)

                r0 -= sol[shear_start+1] * o.shear.poten_d2x(img.pos)
                r1 -= sol[shear_start+1] * o.shear.poten_d2y(img.pos)

            srcpos = srcpos_start + 2*i
            r0 -= sol[srcpos+0]
            r1 -= sol[srcpos+1]

            l0 = log10(abs(r0)) if r0 else -13
            l1 = log10(abs(r1)) if r1 else -13

            res += '['
            res += ' ' if l0 <= -12 else '.' if l0 <= -11 else '-' if l0 <= -10 else '*' if l0 <= -9 else '%-3i' % l0
            res += ' '
            res += ' ' if l1 <= -12 else '.' if l1 <= -11 else '-' if l1 <= -10 else '*' if l1 <= -9 else '% 3i' % l1
            res += ']'
            #res += '[%-4i %-4i]' % (r0, r1)

        Log( '    %s  src.zcap=%6.4f %s' % (o.name, src.zcap, res) )


@default_prior
@object_prior
def time_delay(o, leq, eq, geq):
    Log( "Time Delay" )

    b  = o.basis
    nu = 1+b.H0

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    tscale = o.scales['time']

    zLp1 = o.z + 1

    shft = [b.map_shift, b.map_shift]

    for i, src in enumerate(o.sources):
        for img0,img1,delay in src.time_delays:

            delay = [d / zLp1 if d else d for d in delay]

            row = new_row(o)

            srcpos = srcpos_start + 2*i

            r0, r1 = img0.pos, img1.pos

            x0, y0 = r0.real, r0.imag
            x1, y1 = r1.real, r1.imag

            # The constant term
            row[0]  = (abs(r1)**2 - abs(r0)**2) / 2
            row[0] += dot([x1-x0, y1-y0], shft)
            row[0] *= src.zcap

            # The beta term
            row[srcpos:srcpos+2]  = x0-x1, y0-y1
            #row[srcpos:srcpos+2] *= src.zcap

            # The ln terms
            row[pix_start:pix_end] -= poten(img1.pos - o.basis.ploc, b.cell_size)
            row[pix_start:pix_end] += poten(img0.pos - o.basis.ploc, b.cell_size)

            if o.shear:
                row[shear_start+0] -= o.shear.poten(1, img1.pos)
                row[shear_start+0] += o.shear.poten(1, img0.pos)
                row[shear_start+1] -= o.shear.poten(2, img1.pos)
                row[shear_start+1] += o.shear.poten(2, img0.pos)

#           for n,offs in enumerate(xrange(ptmass_start, ptmass_end)):
#               row[offs] -= o.ptmass.poten(n+1, img1.pos, o.basis.cell_size)
#               row[offs] += o.ptmass.poten(n+1, img0.pos, o.basis.cell_size)

            if len(delay) == 1:
                d = delay[0]
                if d is None: row[nu] =  0; geq(row)
                else:         row[nu] = -d;  eq(row)
            else:
                l,u = delay
                if   l is None: row[nu] = -u; leq(row)
                elif u is None: row[nu] = -l; geq(row)
                else:
                    row2 = row.copy()
                    row[nu]  = -l; geq(row)
                    row2[nu] = -u; leq(row2)

            #print 'row:', len(row), row

            #print row

@object_prior_check(time_delay)
def check_time_delay(o, sol):

    Log( "Check Time Delay (' ':-12  '.':-11  '-':-10  '*':-9)" )

##  ls = []
##  f = lambda x: ls.append(x)
##  time_delay(o, f,f,f)

##  res = []
##  for l in ls:
##      s = l[0] + dot(sol[1:], l[1:])

##      l0 = log10(abs(s)) if s else -13

##      res.append(' ' if l0 <= -12 else '.' if l0 <= -11 else '-' if l0 <= -10 else '*' if l0 <= -9 else '%-3i' % l0)

##  Log( '    [%s]' % ' '.join(res) )
##  return


    b  = o.basis
    nu = 1+b.H0

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    zLp1 = o.z + 1
    shft = [b.map_shift, b.map_shift]

    for i, src in enumerate(o.sources):
        res = ''
        for img0,img1,delay in src.time_delays:

            delay = [d / zLp1 if d else d for d in delay]

            row = new_row(o)

            srcpos = srcpos_start + 2*i

            r0, r1 = img0.pos, img1.pos
            x0, y0 = img0.pos.real, img0.pos.imag
            x1, y1 = img1.pos.real, img1.pos.imag

            # The constant term
            row[0]  = (abs(r1)**2 - abs(r0)**2) / 2
            row[0] += dot([x1-x0, y1-y0], shft)
            row[0] *= src.zcap

            # The beta term
            row[srcpos:srcpos+2]  = x0-x1, y0-y1
            #row[srcpos:srcpos+2] *= src.zcap

            # The ln terms
            row[pix_start:pix_end] -= poten(img1.pos - o.basis.ploc, b.cell_size)
            row[pix_start:pix_end] += poten(img0.pos - o.basis.ploc, b.cell_size)

            if o.shear:
                row[shear_start+0] -= o.shear.poten(1, img1.pos)
                row[shear_start+0] += o.shear.poten(1, img0.pos)
                row[shear_start+1] -= o.shear.poten(2, img1.pos)
                row[shear_start+1] += o.shear.poten(2, img0.pos)

#           for n,offs in enumerate(xrange(ptmass_start, ptmass_end)):
#               row[offs] -= o.ptmass.poten(n+1, img1.pos, o.basis.cell_size)
#               row[offs] += o.ptmass.poten(n+1, img0.pos, o.basis.cell_size)

            row2 = None

            if len(delay) == 1:
                d = delay[0]
                if d is None: row[nu] =  0
                else:         row[nu] = -d
            else:
                l,u = delay
                if   l is None: row[nu] = -u
                elif u is None: row[nu] = -l
                else:
                    row2 = row.copy()
                    row[nu]  = -l
                    row2[nu] = -u

            r0 = dot(row[1:], sol[1:]) + row[0]
            if row2:
                r1 = dot(row2[1:], sol[1:]) + row2[0]

            l0 = log10(abs(r0)) if r0 else -13
            if row2:
                l1 = log10(abs(r1)) if r1 else -13

            #res += '[%i %i]' % (r0,r1)

            res += '['
            res += ' ' if l0 <= -12 else '.' if l0 <= -11 else '-' if l0 <= -10 else '*' if l0 <= -9 else '%-3i' % l0
            if row2:
                res += ' '
                res += ' ' if l1 <= -12 else '.' if l1 <= -11 else '-' if l1 <= -10 else '*' if l1 <= -9 else '% 3i' % l1
            res += ']'

        Log( '    %s  src.zcap=%6.4f %s' % (o.name, src.zcap, res) )

#@object_prior
def JPC1time_delay(o, leq, eq, geq):
    Log( "JPC Time Delay" )

    b  = o.basis
    nu = 1+b.H0

    pix_start, pix_end = 1+b.pix_start, 1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    for i, src in enumerate(o.sources):

        srcpos = srcpos_start + 2*i

        img0      = src.images[-1].pos
        x0, y0    = img0.real, img0.imag
        img0poten = poten(img0 - b.ploc, b.cell_size)

        for j,img1 in enumerate(b.ploc):
            if abs(img0-img1) <= b.cell_size[j]: continue

            row = new_row(o)

            x1, y1 = img1.real, img1.imag

            # The constant term
            row[0] = (abs(img1)**2 - abs(img0)**2) / 2 + (x1-x0 + y1-y0)*b.map_shift
            row[0] *= src.zcap

            # The beta term
            row[srcpos:srcpos+2]  = x0-x1, y0-y1
            #row[srcpos:srcpos+2] *= src.zcap

            # The ln terms
            row[pix_start:pix_end] -= poten(img1 - b.ploc, b.cell_size)
            row[pix_start:pix_end] += img0poten

            if o.shear:
                row[shear_start+0] -= o.shear.poten(1, img1)
                row[shear_start+0] += o.shear.poten(1, img0)
                row[shear_start+1] -= o.shear.poten(2, img1)
                row[shear_start+1] += o.shear.poten(2, img0)

            # TODO: Ptmass

            row[nu] = 0
            leq(row)

            #print row

@default_prior
@object_prior
def hubble_constant(o, leq, eq, geq):
    """This requires a particular hubble constant for the object."""

    if env().nu is None:
        Log( "Hubble Constant DISABLED")
        return

    Log( "Hubble Constant" + str(env().nu))

    nu = 1+o.basis.H0

    lb, ub = env().nu[0], env().nu[-1]

    if lb is not None and ub is not None:
        assert ub >= lb, 'Hubble constant contraints must be given as (lower_bound, upper_bound)'

    if lb == ub:
        row = new_row(o)
        row[ [0,nu] ] = lb, -1
        eq(row)
    else:
        if ub is not None:
            row = new_row(o)
            row[ [0,nu] ] = ub, -1
            geq(row)

        if lb is not None:
            row = new_row(o)
            row[ [0,nu] ] = lb, -1
            leq(row)

#@object_prior
def parity(o, leq, eq, geq):
    Log( "Parity" )

    b = o.basis

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    for i,src in enumerate(o.sources):
        for img in src.images:
            parity = img.parity

            xy,xx,yy = maginv_new4(img.pos - o.basis.ploc, img.angle, o.basis.cell_size)

            rows = new_row(o,4)
            rows[0,0] = src.zcap
            rows[1,0] = src.zcap

            # TODO: Shear term

            rows[0,pix_start:pix_end] = xx
            rows[1,pix_start:pix_end] = yy

            if parity == MINIMUM:
                #rows[0,0] -= 1e-4
                #rows[1,0] -= 1e-4
                geq(rows[0])
                geq(rows[1])

            if parity == MAXIMUM:
                #rows[0,0] += 1e-4
                #rows[1,0] += 1e-4
                leq(rows[0])
                leq(rows[1])

            if parity == SADDLE:
                #rows[0,0] -= 1e-4
                #rows[1,0] += 1e-4
                geq(rows[0])
                leq(rows[1])

@default_prior
@object_prior
def J1parity(o, leq, eq, geq):
    Log( "J1 Parity" )

    b = o.basis

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

#   print '*' * 80
#   print 'TURNED OFF PARITY MAXIMUM CONDITION'
#   print '*' * 80

    for i,src in enumerate(o.sources):
        for img in src.images:
            parity = img.parity

            rows      = new_row(o,2)
            rows[0,0] = src.zcap
            rows[1,0] = src.zcap

            positions = img.pos - b.ploc
            rows[0,pix_start:pix_end] = -poten_dxdx(positions, o.basis.cell_size)
            rows[1,pix_start:pix_end] = -poten_dydy(positions, o.basis.cell_size)

            if o.shear:
                rows[0,shear_start:shear_end] = [ 2, 0]
                rows[1,shear_start:shear_end] = [-2, 0]

            if parity == MINIMUM:
                geq(rows[0])
                geq(rows[1])

            if parity == MAXIMUM:
                leq(rows[0])
                leq(rows[1])

            #if parity == SADDLE:
            # Parity contraint only valid if Hessian is diagonal


#@object_prior
def magnification(o, leq, eq, geq):

    Log( "Magnification" )

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,     pix_end = 1+o.basis.pix_start,   1+o.basis.pix_end
    shear_start, shear_end = 1+o.basis.shear_start, 1+o.basis.shear_end

    for src in o.sources:
        for img in src.images:
            k1, k2, eps = img.elongation
            parity = img.parity
            k2 = 1/k2

            rows = new_row(o,6)
            rows[0,0] = [k1-1,  k1+1, -k1+1][parity] * src.zcap
            rows[1,0] = [k2-1, -k2-1, -k2+1][parity] * src.zcap
            rows[2,0] = [-eps,  -eps,   eps][parity] * src.zcap
            rows[3,0] = [-eps,  -eps,   eps][parity] * src.zcap
            rows[4,0] = [-eps,   eps,   eps][parity] * src.zcap
            rows[5,0] = [-eps,   eps,   eps][parity] * src.zcap

            print "42", img.angle, k1, k2, eps
            xy,xx,yy = maginv(img.pos - o.basis.ploc, img.angle, o.basis.cell_size)
            #print "MAG", xx, yy, xy
            if parity == MINIMUM:
                rows[0, pix_start:pix_end] = -k1*xx + yy
                rows[1, pix_start:pix_end] = -k2*yy + xx
                rows[2, pix_start:pix_end] =     xy + xx*eps
                rows[3, pix_start:pix_end] =    -xy + xx*eps
                rows[4, pix_start:pix_end] =     xy + yy*eps
                rows[5, pix_start:pix_end] =    -xy + yy*eps
                print "43", rows[0, pix_start], img.pos-o.basis.ploc[0], xx[0], yy[0], xy[0], src.zcap

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

@default_prior
@object_prior
def annular_density(o, leq, eq, geq):
    theta = o.prior_options.get('annular_density', None)
    if theta is None:
        Log( "Annular density DISABLED" )
        return

    Log( "Annular density %s" % theta )

    if theta is not None and theta != 0:
        row = new_row(o)
        for r in xrange(o.basis.inner_image_ring, o.basis.outer_image_ring):
            row[o.basis.rings[r]] = -1
            row[0] = theta * len(o.basis.rings[r])
        eq(row)
        on = True
        
@default_prior
@object_prior
def external_shear(o, leq, eq, geq):
    on = None
    for s in xrange(1+o.basis.shear_start, 1+o.basis.shear_end):
        row = new_row(o)
        row[ [0,s] ] = 0.1, -1
        geq(row)
        on = [0.1, -1]

    if on is None:
        on = 'DISABLED'

    Log( "External Shear %s" % on)

##############################################################################

@default_prior
@object_prior
def profile_steepness(o, leq, eq, geq):
    steep = o.prior_options.get('steepness', None)

    if steep is None: 
        Log( "Profile Steepness DISABLED" )
        return

    Log( "Profile Steepness %s" % steep )

    minsteep, maxsteep = steep
    assert maxsteep is None or maxsteep >= minsteep

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    nrings = len(o.basis.rings)
    row = new_row(o)

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
            row = new_row(o)
            lc  = l ** minsteep
            lpc = -((l+1) ** minsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            eq(row)
        else:
            row = new_row(o)
            lc  = l ** minsteep
            lpc = -((l+1) ** minsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            geq(row)

            if maxsteep is not None:
                row = new_row(o)
                lc  = l ** maxsteep
                lpc = -((l+1) ** maxsteep)
                row[pix_start+r0] = lc  / len(r0)
                row[pix_start+r1] = lpc / len(r1)
                leq(row)
                c += 1


#   print "\tmaxsteep=", maxsteep, "minsteep=",minsteep
#   if maxsteep > minsteep:
#       row = zeros(1+o.basis.nvar)
#       r0 = o.basis.rings[1]
#       r1 = o.basis.rings[-2]
#       lc  = -1
#       lpc =  nrings ** maxsteep
#       row[pix_start+r0] = lc  / len(r0)
#       row[pix_start+r1] = lpc / len(r1)
#       geq(row)
#       c += 1
    Log( "\t# eqs = %i" % c )
        
#@default_prior
#@object_prior
def gradient(o, leq, eq, geq):
    cen_ang = o.prior_options.get('gradient', pi/4)

    Log( "Gradient %s" % cen_ang )
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    cs = cos(cen_ang)
    sn = sin(cen_ang)
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

        row = new_row(o)
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
            row = new_row(o)
            row[pix_start + nbrs] = xx
            geq(row)
            c += 1

    Log( "\t# eqs = %i" % c )
    Log( "\tsn=%g" % sn )

#@default_prior
@object_prior
def Pgradient(o, leq, eq, geq):
    cen_ang = o.prior_options.get('gradient', pi/4)

    Log( "PGradient %s" % cen_ang )

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    cs = cos(cen_ang)
    sn = sin(cen_ang)
    c = 0
    for ri,r,nbrs in o.basis.nbrs2:
        if ri == o.basis.central_pixel: continue

        px = r.real
        py = r.imag
        x = cs*px - sn*py
        y = sn*px + cs*py

        dr = r - o.basis.int_ploc[nbrs]
        dir = dr / abs(dr)

        dW = abs(dr)*(o.basis.grad_rmax + 1 - abs(dr))

        row = new_row(o)
        row[pix_start + nbrs] = dW * (dir * complex(x,-y)).real
        geq(row)
        c += 1

        if sn != 0:
            x =  cs*px + sn*py
            y = -sn*px + cs*py
            row = new_row(o)
            row[pix_start + nbrs] = dW * (dir * complex(x,-y)).real
            geq(row)
            c += 1

    Log( "\tgradient eqs = %i" % c )
    Log( "\tsn=%g" % sn )

@default_prior
@object_prior
def J2gradient(o, leq, eq, geq):

    opts = o.prior_options.get('J2Gradient', None)

    if not opts:
        opts = {}
        #Log( 'J2Gradient NOT ACTIVE' )
        #return

    Lmin = 1.1*o.basis.top_level_cell_size

    theta = opts.get('theta', 45)
    L     = opts.get('size',  Lmin)

    assert (L >= Lmin), 'size=%f < %f is too small' % (L, Lmin)

    Log( "J2Gradient (theta=%.2f  size=%.2f)" % (theta, L) )

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    theta = radians(90-theta)
    cs = cos(theta)
    sn = sin(theta)
    c = 0
    for i,[ri,r] in enumerate(izip(o.basis.int_ploc, o.basis.ploc)):
        if i == o.basis.central_pixel: continue

        nbrs = neighbors(r,L, o.basis.ploc)

        px = r.real
        py = r.imag
        x = cs*px - sn*py
        y = sn*px + cs*py

        #dr = ri - o.basis.int_ploc[nbrs]
        dr = r - o.basis.ploc[nbrs]
        dir = dr / abs(dr)

        dW = abs(dr)*(L + o.basis.top_level_cell_size - abs(dr))
        #dW = abs(dr)*(o.basis.grad_rmax + 1 - abs(dr))

        row = new_row(o)
        row[pix_start + nbrs] = dW * (dir * complex(x,-y)).real
        geq(row)
        c += 1

        if sn != 0:
            x =  cs*px + sn*py
            y = -sn*px + cs*py
            row = new_row(o)
            row[pix_start + nbrs] = dW * (dir * complex(x,-y)).real
            geq(row)
            c += 1

    Log( "\tgradient eqs = %i" % c )
    Log( "\tsn=%g" % sn )

#@default_prior
@object_prior
def central_pixel_as_maximum(o, leq, eq, geq):
    cp = o.basis.central_pixel + 1+o.basis.pix_start
    Log( "Central pixel as maximum %i" % cp )

    for i in xrange(1+o.basis.pix_start, 1+o.basis.pix_end):
        if i == cp: continue
        row = new_row(o)
        row[ [cp,i] ] = 1, -1
        geq(row)

@object_prior
def central_pixel_max(o, leq, eq, geq):

    cp = o.basis.central_pixel + 1+o.basis.pix_start
    Log( "Central pixel maximum %i" % cp )

    g = o.prior_options.get('central_pixel_maximum')
    M     = g['M']
    H0inv = g['H0inv']
    nu    = g['nu']

    max_nu = env().nu[-1] # should this be min?

    M = convert('Msun/kpc^2 to kappa',  M, o.dL, max_nu)

    row = new_row(o)
    row[ [0,cp] ] = -M, 1
    leq(row)

@object_prior
def PLsmoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "Smoothness [None]" )
        return

    pix_start, pix_end    = 1+o.basis.pix_start, 1+o.basis.pix_end
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( "Smoothness (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    for i,r,nbrs in o.basis.nbrs:
        if not include_central_pixel and i == o.basis.central_pixel: continue

        row = new_row(o)
        row[pix_start + nbrs] = 1
        row[pix_start + i]    = -len(nbrs) / smoothness_factor

        geq(row)
        c += 1

    Log( "\t# eqs = %i" % c )

@object_prior
def JCsmoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "JCSmoothness DISABLED" )
        return

    pix_start, pix_end    = 1+o.basis.pix_start, 1+o.basis.pix_end
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( "JCSmoothness (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    for i,r,nbrs in o.basis.nbrs:
        if not include_central_pixel and i == o.basis.central_pixel: continue

        row = new_row(o)
        row[pix_start + nbrs] = 1
        row[pix_start + i]    = -len(nbrs) / ((smoothness_factor-1) * (1 - abs(r) / o.basis.pixrad) + 1)

        geq(row)
        c += 1

    Log( "\t# eqs = %i" % c )

@default_prior
@object_prior
def smoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {})

    Lmin = 1.5*o.basis.top_level_cell_size

    pix_start, pix_end    = 1+o.basis.pix_start, 1+o.basis.pix_end
    smoothness_factor     = smth.get('factor', 2)
    L                     = smth.get('L', Lmin)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( "Smoothness (factor=%.1f L=%.1f include_central_pixel=%s)" % (smoothness_factor, L, include_central_pixel) )

    c=0
    for i,[ri,r] in enumerate(izip(o.basis.int_ploc, o.basis.ploc)):
        if not include_central_pixel and i == o.basis.central_pixel: continue

        nbrs = neighbors(r,L, o.basis.ploc)

        row = new_row(o)
        row[pix_start + nbrs] = 1
        row[pix_start + i]    = -len(nbrs) / ((smoothness_factor-1) * (1 - abs(ri) / o.basis.pixrad) + 1)

        geq(row)
        c += 1

    Log( "\t# eqs = %i" % c )

@default_prior
@ensemble_prior
def shared_h(objs, nvars, leq, eq, geq):
    """This requires that all objects have the same hubble constant."""
    on = False
    for o1,o2 in izip(objs[:-1], objs[1:]):
        offs1 = o1.basis.array_offset
        offs2 = o2.basis.array_offset
        row = zeros(1+nvars)
        row[offs1 + o1.basis.H0] =  1
        row[offs2 + o2.basis.H0] = -1
        eq(row) 
        on = True

    if not on: 
        Log( "Shared h DISABLED" )
    else:
        Log( "Shared h" )


@object_prior
def max_kappa(o, leq, eq, geq):
    Log( "Maximum Kappa" )
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    for i in xrange(pix_start, pix_end):
        row = new_row(o)
        row[ [0,i] ] = -6, 1
        leq(row)

    
@object_prior
def min_kappa(o, leq, eq, geq):
    Log( "Minimum Kappa" )
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    for i in xrange(pix_start, pix_end):
        row = new_row(o)
        row[ [0,i] ] = -0.6, 1
        geq(row)

@object_prior
def min_kappa_grid(o, leq, eq, geq):

    g = o.prior_options.get('minkappa')

    if not g: 
        Log( "Minimum Kappa Grid DISABLED" )
        return

    Log( "Minimum Kappa Grid" )

    X,Y,M = g['grid']
    H0inv = g['H0inv']
    nu    = g['nu']

    max_nu = env().nu[-1]

    g  = o.basis.grid_mass(X,Y,M, H0inv, to_kappa=False)
    #g *= convert('den_stel to den_lum', 1., o.dL, nu) 
    g *= convert('Msun/ly^2 to kappa',  1., o.dL, max_nu)

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    a = g.ravel()[o.basis.insideL].take(o.basis.pmap)

    #nu = 1+o.basis.H0

    for i,j in enumerate(xrange(pix_start, pix_end)):
        row = new_row(o)
        #row[ [nu,j] ] = -a[i], 1
        row[ [0,j] ] = -a[i], 1
        geq(row)

#@object_prior
def Xmin_kappa_grid(o, leq, eq, geq):

    g = o.prior_options.get('minkappa')

    if not g: 
        Log( "Minimum Kappa Grid DISABLED" )
        return

    Log( "Minimum Kappa Grid" )

    X,Y,M = g['grid']
    H0inv = g['H0inv']

    g = o.basis.grid_mass(X,Y,M, H0inv) * H0inv

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    a = g.ravel()[o.basis.insideL].take(o.basis.pmap)

    nu = 1+o.basis.H0

    for i,j in enumerate(xrange(pix_start, pix_end)):
        row = new_row(o)
        row[ [nu,j] ] = -a[i], 1
        #row[ [0,j] ] = -a[i], 1
        geq(row)

@object_prior
def min_kappa_model(o, leq, eq, geq):
    Log( "Minimum Kappa Model" )

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    a = o.basis.min_kappa_model

    for i,j in enumerate(xrange(pix_start, pix_end)):
        row = new_row(o)
        row[ [0,j] ] = -a[i], 1
        geq(row)

@object_prior
def smoothness2(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness2', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "Smoothness [None]" )
        return

    pix_start, pix_end    = 1+o.basis.pix_start, 1+o.basis.pix_end
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( "Smoothness (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    for i,r,nbrs in o.basis.nbrs:
        if not include_central_pixel and i == o.basis.central_pixel: continue

        for n in nbrs:
            row = new_row(o)
            row[pix_start + i] = 1
            row[pix_start + n] = -smoothness_factor

            leq(row)
            c += 1

    Log( "\t# eqs = %i" % c )

@default_prior
@object_prior
def symmetry(o, leq, eq, geq):

    if not o.symm:
        Log( "Symmetry DISABLED" )
        return

    Log( "Symmetry" )

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    c = 0
    done = zeros(o.basis.int_ploc.size)
    for i,ri in enumerate(o.basis.int_ploc):
        if i == o.basis.central_pixel: continue
        if done[i]: continue

        j = argwhere(o.basis.int_ploc == -ri).flatten()
        assert j.size == 1
        j = j[0]

        done[i] = 1
        done[j] = 1

        row = new_row(o)
        row[pix_start + i] = 1
        row[pix_start + j] = -1

        eq(row)
        c+=1

    Log( "\t# eqs = %i" % c )


#@default_prior
@object_prior
def max_kappa(o, leq, eq, geq):

    Log( "Maximum Kappa" )

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    for i,j in enumerate(xrange(pix_start, pix_end)):
        row = new_row(o)
        row[ [0,j] ] = 100, -1
        leq(row)
