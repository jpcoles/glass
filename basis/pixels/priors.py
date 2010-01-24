from __future__ import division
from environment import env
import numpy
from numpy import zeros, array, empty, cos, sin, compress, sign, logical_or, sort, pi
from potential import poten, poten_dx, poten_dy, poten_dxdx, poten_dydy, maginv, maginv_new, poten_dxdy, maginv_new4, maginv_new5
from itertools import izip

all_priors = []
inc_priors = []
exc_priors = []
acc_objpriors = []
acc_enspriors = []

def _make_prior(f, where):
    class P: 
        def __init__(self, f, where): 
            self.f, self.check, self.where = f, None, where
        def __eq__(self, f):
            return self.f == f
    print f
    all_priors.append(P(f, where))
    return all_priors[-1]

def object_prior(f):   return _make_prior(f, 'object_prior')
def ensemble_prior(f): return  _make_prior(f, 'ensemble_prior')

def object_prior_check(check): 
    def x(f):
        all_priors[all_priors.index(check)].check = f
    return x

def include_prior(f):
    assert not exc_priors, 'Cannot both include and exclude priors.'
    i = all_priors.index(f)
    inc_priors.append(all_priors[i])

def exclude_prior(f):
    assert not inc_priors, 'Cannot both include and exclude priors.'
    i = all_priors.index(f)
    exc_priors.append(all_priors[i])


##############################################################################

def new_row(obj, n=1):
    if n == 1: 
        return zeros(1+obj.basis.nvar)
    else:
        return zeros((n,1+obj.basis.nvar))

##############################################################################


@object_prior
def image_pos(o, leq, eq, geq):
    print "Image Position"

    b = o.basis

    pix_start, pix_end = 1+b.pix_start, 1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    for i,src in enumerate(o.sources):
        for img in src.images:
            rows = new_row(o, 2)
            print "\tposition", img.pos #, b.cell_size
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
            rows[0,srcpos:srcpos+2] = -src.zcap,     0
            rows[1,srcpos:srcpos+2] =     0,     -src.zcap

            eq(rows[0])
            eq(rows[1])

@object_prior_check(image_pos)
def check_image_pos(o, sol):
    print "Check Image Position"
    b    = o.basis
    offs = b.array_offset
    print offs

    pix_start,    pix_end    = offs+b.pix_start,    offs+b.pix_end
    srcpos_start, srcpos_end = offs+b.srcpos_start, offs+b.srcpos_end
    shear_start,  shear_end  = offs+b.shear_start,  offs+b.shear_end
    ptmass_start, ptmass_end = offs+b.ptmass_start, offs+b.ptmass_end

    rows = new_row(o, 2)
    for i,src in enumerate(o.sources):
        print 'src.zcap=',src.zcap
        for img in src.images:
            rows[:,:] = 0
            r0 = (img.pos.real + b.map_shift) * src.zcap
            r1 = (img.pos.imag + b.map_shift) * src.zcap
            positions = img.pos - b.ploc
            r0 += sum(sol[pix_start:pix_end] * -poten_dx(positions, b.cell_size))
            r1 += sum(sol[pix_start:pix_end] * -poten_dy(positions, b.cell_size))

            if o.shear:
                r0 += sol[shear_start] * -o.shear.poten_dx(img.pos)
                r1 += sol[shear_start] * -o.shear.poten_dy(img.pos)

                r0 += sol[shear_start+1] * -o.shear.poten_d2x(img.pos)
                r1 += sol[shear_start+1] * -o.shear.poten_d2y(img.pos)

            srcpos = srcpos_start + 2*i
            r0 += sol[srcpos+0] * -src.zcap
            r1 += sol[srcpos+1] * -src.zcap

            print r0, r1


@object_prior
def time_delay(o, leq, eq, geq):
    print "Time Delay"

    b  = o.basis
    H0 = 1+b.H0

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    tscale = o.scales['time']

    for i, src in enumerate(o.sources):
        for img0,img1,delay in src.time_delays:

            row = new_row(o)

            srcpos = srcpos_start + 2*i

            x0, y0 = img0.pos.real, img0.pos.imag
            x1, y1 = img1.pos.real, img1.pos.imag

            # The constant term
            row[0] = (abs(img1.pos)**2 - abs(img0.pos)**2) / 2 + (x1-x0 + y1-y0)*b.map_shift
            row[0] *= src.zcap

            # The beta term
            row[srcpos:srcpos+2]  = x0-x1, y0-y1
            row[srcpos:srcpos+2] *= src.zcap

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
                if d is None: row[H0] =  0 / tscale; geq(row)
                else:         row[H0] = -d / tscale;  eq(row)
            else:
                l,u = delay
                if   l is None: row[H0] = -u / tscale; leq(row)
                elif u is None: row[H0] = -l / tscale; geq(row)
                else:
                    row2 = row.copy()
                    row[H0]  = -l / tscale; geq(row)
                    row2[H0] = -u / tscale; leq(row2)

            #print 'row:', len(row), row

            #print row

#@object_prior
def JPC1time_delay(o, leq, eq, geq):
    print "JPC Time Delay"


    b  = o.basis
    H0 = 1+b.H0

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
            row[srcpos:srcpos+2] *= src.zcap

            # The ln terms
            row[pix_start:pix_end] -= poten(img1 - b.ploc, b.cell_size)
            row[pix_start:pix_end] += img0poten

            if o.shear:
                row[shear_start+0] -= o.shear.poten(1, img1)
                row[shear_start+0] += o.shear.poten(1, img0)
                row[shear_start+1] -= o.shear.poten(2, img1)
                row[shear_start+1] += o.shear.poten(2, img0)

            # TODO: Ptmass

            row[H0] = 0
            leq(row)

            #print row

@object_prior
def hubble_constant(o, leq, eq, geq):
    """This requires a particular hubble constant for the object."""
    print "Hubble Constant"
    on = False

    H0 = 1+o.basis.H0

#   if not env().h_spec:
#       row = zeros(o.basis.nvar)
#       row[0] = 0
#       row[o.basis.H0] = -1
#       geq(row)
    if len(env().h_spec) == 1:
        h, = env().h_spec
        row = new_row(o)
        row[0] = h
        row[H0] = -1
        eq(row)
    else:
        lb, ub = env().h_spec
        if ub is not None:
            row = new_row(o)
            row[0] = ub
            row[H0] = -1
            leq(row)

        if lb is not None:
            row = new_row(o)
            row[0] = lb
            row[H0] = -1
            geq(row)

#       row = zeros(1+o.basis.nvar)
#       row[0] = env().h_spec / o.scales['time']
#       row[H0] = -1
#       eq(row)
        on = True
    print "\t", on

#@object_prior
def parity(o, leq, eq, geq):
    print "Parity"

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

@object_prior
def J1parity(o, leq, eq, geq):
    print "J1 Parity"

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
def J2parity(o, leq, eq, geq):
    print "Parity"

    b = o.basis

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

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

#       M = [[ (0.99998158084853316918, -0.00606942861138636339),
#              (0.99811561515346303164, -0.06136137862551745148) ],
#            [ (-0.99987787876445743951, -0.01562778159205616135),
#              (-0.77811691937027627208, -0.62811946299228049728) ]]

    for i,src in enumerate(o.sources):
        for j,img in enumerate(src.images):
            parity = img.parity

            c,s = M[i][j]

            xy,xx,yy = maginv_new5(img.pos - o.basis.ploc, o.basis.cell_size, c,s)

            rows = new_row(o,2)
            rows[0,0] = src.zcap * (c**2 + s**2)
            rows[1,0] = src.zcap * (c**2 + s**2)

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

#@object_prior
def magnification(o, leq, eq, geq):

    print "Magnification"

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

@object_prior
def annular_density(o, leq, eq, geq):
    print "Annular density"
    on = False
    if o.kann_spec != 0:
        row = new_row(o)
        for r in xrange(o.basis.inner_image_ring, o.basis.outer_image_ring):
            row[o.basis.rings[r]] = -1
            row[0] = kann_spec * len(o.basis.rings[r])
        eq(row)
        on = True

    print "\t", on
        
@object_prior
def external_shear(o, leq, eq, geq):
    print "External Shear"
    on = False
    for s in xrange(1+o.basis.shear_start, 1+o.basis.shear_end):
        row = new_row(o)
        row[0] =  0.1
        row[s] = -1
        geq(row)
        on = True

    print "\t", on

##############################################################################

@object_prior
def steepness(o, leq, eq, geq):
    print "Steepness" 

    if o.steep is None: return

    minsteep, maxsteep = o.steep

    assert maxsteep >= minsteep

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
            #print r0,r1
            #print row
            eq(row)
        else:
            row = new_row(o)
            lc  = l ** minsteep
            lpc = -((l+1) ** minsteep)
            row[pix_start+r0] = lc  / len(r0)
            row[pix_start+r1] = lpc / len(r1)
            #print r0,r1
            #print row
            geq(row)

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
    print "\tsteepness eqs =", c
        
#@object_prior
def gradient(o, leq, eq, geq):
    print "Gradient"
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

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

    print "\tgradient eqs =", c
    print "\tsn=", sn

@object_prior
def Pgradient(o, leq, eq, geq):
    print "PGradient"
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end

    cs = cos(o.cen_ang)
    sn = sin(o.cen_ang)
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

    print "\tgradient eqs =", c
    print "\tsn=", sn

#@object_prior
def central_pixel_as_maximum(o, leq, eq, geq):
    cp = o.basis.central_pixel + 1+o.basis.pix_start
    print "Central pixel as maximum", cp

    for i in xrange(1+o.basis.pix_start, 1+o.basis.pix_end):
        if i == cp: continue
        row = new_row(o)
        row[cp] = 1
        row[i]  = -1
        geq(row)

@object_prior
def smoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    smoothness_factor = 2

    print "Smoothness (factor=%.1f)" % smoothness_factor

    c=0
    for i,r,nbrs in o.basis.nbrs:
        #-----------------------------------------------------------------------
        # Skip the central pixel. This allows any value of mass.
        # XXX: Some versions of PixeLens don't.
        #-----------------------------------------------------------------------
        #if i == o.basis.central_pixel: continue

        row = new_row(o)
        row[pix_start + nbrs] = 1
        row[pix_start + i]    = -len(nbrs) / smoothness_factor
        # XXX: row[pix_start + i]    = -len(nbrs) / 2

        geq(row)
        c += 1

    print "\tsmoothness eqs =", c

#@object_prior
def more_smoothness(o, leq, eq, geq):
    """The average of the neighbouring pixels can't be more than twice
       the value of a given pixel."""

    print "More Smoothness"
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    smoothness_factor = 3

    c=0
    for i,r,nbrs in o.basis.nbrs:
        #-----------------------------------------------------------------------
        # Skip the central pixel. This allows any value of mass.
        # XXX: Some versions of PixeLens don't.
        #-----------------------------------------------------------------------
        #if i == o.basis.central_pixel: continue

        for n in nbrs:
            row = new_row(o)
            row[pix_start + i] = 1
            row[pix_start + n]  = -smoothness_factor
            leq(row)
            c += 1

    print "\tsmoothness factor =", smoothness_factor
    print "\tsmoothness eqs =", c

@ensemble_prior
def shared_h(objs, nvars, leq, eq, geq):
    """This requires that all objects have the same hubble constant."""
    print "Shared h"
    on = False
    for o1,o2 in izip(objs[:-1], objs[1:]):
        offs1 = o1.basis.array_offset
        offs2 = o2.basis.array_offset
        row = zeros(1+nvars)
        row[offs1 + o1.basis.H0] =  1 #o1.scales['time']
        row[offs2 + o2.basis.H0] = -1 #o2.scales['time']
        eq(row) 
        on = True
    print "\t", on


#@object_prior
def max_kappa(o, leq, eq, geq):
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    for i in xrange(pix_start, pix_end):
        row = new_row(o)
        row[i] = 1
        row[0] = -6
        leq(row)

    
#@object_prior
def min_kappa(o, leq, eq, geq):
    pix_start, pix_end = 1+o.basis.pix_start, 1+o.basis.pix_end
    for i in xrange(pix_start, pix_end):
        row = new_row(o)
        row[i] = 1
        row[0] = -0.6
        geq(row)

