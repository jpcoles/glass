from __future__ import division
if __name__ != '__main__':
    from glass.environment import env
    from glass.command import command
    from glass.shear import Shear
    from glass.exmass import ExternalMass
    from itertools import izip
    from glass.log import log as Log
    from glass.scales import convert
    from . basis import neighbors, irrhistogram2d
    from . potential import poten, poten_dx, poten_dy, poten_dxdx, poten_dydy, maginv, maginv_new, poten_dxdy, maginv_new4, maginv_new5
else:
    def command(x): pass

import numpy as np
from numpy import amin,amax,zeros, array, empty, cos, sin, compress, sign, logical_or, sort, pi, log10, radians, argwhere, all, dot, sum, loadtxt, amax, flipud


all_priors = []
def_priors = []
inc_priors = []
exc_priors = []
acc_objpriors = []
acc_enspriors = []

indent=11*' '

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
def include_prior(env, *f):
    #assert not exc_priors, 'Cannot both include and exclude priors.'
    for p in f:
        try:
            i = all_priors.index(p)
            inc_priors.append(all_priors[i])
        except ValueError:
            raise Exception("Can't find '%s' from available priors" % p)

@command
def include_all_priors(env):
    del inc_priors[:], exc_priors[:]
    for p in all_priors:
        inc_priors.append(p)

@command
def exclude_prior(env, *f):
    #assert not inc_priors, 'Cannot both include and exclude priors.'
    for p in f:
        i = all_priors.index(p)
        exc_priors.append(all_priors[i])

@command
def exclude_all_priors(env):
    del inc_priors[:], exc_priors[:]
    for p in all_priors:
        exc_priors.append(p)

@command
def prior_included(env, f):
    try:
        i = inc_priors.index(f)
        return True
    except ValueError:
        return False

##############################################################################

def new_row(obj, n=1):
    if n == 1: 
        return zeros(1+obj.basis.nvar)
    else:
        return zeros((n,1+obj.basis.nvar))

##############################################################################

def find_stellar_mass(o):
    if not isinstance(o.stellar_mass,type(0)):
        return o.stellar_mass
    assert o.stellar_mass == 0

    s = o.prior_options.has_key('minkappa Leier grid') \
      + o.prior_options.has_key('min_kappa_particles')

    assert s <= 1, 'Only one stellar mass source is allowed.'

    stellar_mass = 0

    g = o.prior_options.get('minkappa Leier grid')
    if g: 
        fname = g['filename']
        grid_size = g['grid size']
        units = g['grid size units']
        scale = g['scale']
        Log( 'Loading stellar mass from %s.' % fname )
        H0inv = convert('nu to H0^-1 in Gyr', env().nu[-1], o.dL)
        load_leier_grid(o, fname, grid_size, units, H0inv, scale)
        stellar_mass = o.stellar_mass

    g = o.prior_options.get('min_kappa_particles')
    if g: 
        X,Y,M = g['particles']
        H0inv = g['H0inv']
        nu    = g['nu']

        Log( 'Projecting mass onto grid.' % fname )
        phys_cell_size = convert('arcsec to kpc', o.basis.cell_size, o.dL, nu)
        phys_ploc = convert('arcsec to kpc', o.basis.ploc, o.dL, nu)
        h = irrhistogram2d(-Y, X, phys_ploc, phys_cell_size, weights=M) / phys_cell_size**2
        h *= convert('Msun/kpc^2 to kappa',  1., o.dL, nu)
        stellar_mass = o.stellar_mass = h
        g['particles'] = None

    return stellar_mass


@default_prior
@object_prior
def lens_eq(o, leq, eq, geq):
    Log( indent + "Lens Equation" )

    b = o.basis

    pix_start, pix_end = 1+b.offs_pix
    srcpos_start, srcpos_end = 1+b.offs_srcpos
    sm_err = 1+b.offs_sm_err
    #shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    #ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

    stellar_mass = find_stellar_mass(o)


    for i,src in enumerate(o.sources):
        for j,img in enumerate(src.images):
            rows = new_row(o, 2)
            Log( 2*indent+"Source %i,Image %i: (% 8.4f, % 8.4f)" % (i,j,img.pos.real, img.pos.imag) ) #, b.cell_size
            rows[0,0] = (img.pos.real + b.map_shift) * src.zcap
            rows[1,0] = (img.pos.imag + b.map_shift) * src.zcap
            positions = img.pos - b.ploc
            rows[0,pix_start:pix_end] = -poten_dx(positions, b.cell_size, o.basis.maprad)
            rows[1,pix_start:pix_end] = -poten_dy(positions, b.cell_size, o.basis.maprad)

            srcpos = srcpos_start + 2*i
            rows[0,srcpos:srcpos+2] = -1,  0
            rows[1,srcpos:srcpos+2] =  0, -1

            sm_x = np.sum(poten_dx(positions, b.cell_size, o.basis.maprad) * stellar_mass)
            sm_y = np.sum(poten_dy(positions, b.cell_size, o.basis.maprad) * stellar_mass)

            if o.stellar_mass_error != 0:
                rows[0,sm_err] = -sm_x
                rows[1,sm_err] = -sm_y
            else:
                rows[0,0] -= sm_x
                rows[1,0] -= sm_y

            for e,[start,end] in izip(o.extra_potentials, b.extra_potentials_array_offsets):
                start += 1
                end += 1

                coeff = e.poten_dx(img.pos)
                rows[0,start:end] -= coeff
                rows[0,0] += e.shift * sum(coeff)

                coeff = e.poten_dy(img.pos)
                rows[1,start:end] -= coeff
                rows[1,0] += e.shift * sum(coeff)

            eq(rows[0])
            eq(rows[1])

@object_prior_check(lens_eq)
def check_lens_eq(o, sol):
    b    = o.basis
    offs = 0 #b.array_offset

    pix_start,    pix_end    = offs+b.offs_pix
    srcpos_start, srcpos_end = offs+b.offs_srcpos
    sm_err = offs+b.offs_sm_err
    #shear_start,  shear_end  = offs+b.shear_start,  offs+b.shear_end
    #ptmass_start, ptmass_end = offs+b.ptmass_start, offs+b.ptmass_end

    assert np.all(sol[pix_start:pix_end] >= 0)

    report = ''
    for i,src in enumerate(o.sources):
        res = ''
        for img in src.images:
            positions = img.pos - b.ploc
            srcpos = srcpos_start + 2*i

            # r0 = img.pos.real * src.zcap
            # r1 = img.pos.imag * src.zcap
            # r0 -= sol[srcpos+0] - b.map_shift * src.zcap
            # r1 -= sol[srcpos+1] - b.map_shift * src.zcap

            r0 = (img.pos.real + b.map_shift) * src.zcap - sol[srcpos+0]
            r1 = (img.pos.imag + b.map_shift) * src.zcap - sol[srcpos+1]

            sm = o.stellar_mass

            if o.stellar_mass_error != 0:
                sm = sm * sol[sm_err]

            r0 -= sum((sm + sol[pix_start:pix_end]) * poten_dx(positions, b.cell_size, o.basis.maprad))
            r1 -= sum((sm + sol[pix_start:pix_end]) * poten_dy(positions, b.cell_size, o.basis.maprad))

            for e,[start,end] in izip(o.extra_potentials, b.extra_potentials_array_offsets):
                r0 -= np.sum((sol[start:end] - e.shift) * e.poten_dx(img.pos))
                r1 -= np.sum((sol[start:end] - e.shift) * e.poten_dy(img.pos))

            #print img.pos, r0,r1, sol[srcpos:srcpos+2]

            l0 = log10(abs(r0)) if r0 else -15
            l1 = log10(abs(r1)) if r1 else -15

            if l0 >= -12 or l1 >= -12:
                res += '['
                res += ' ' if l0 <= -15 else '.' if l0 <= -14 else '-' if l0 <= -13 else '*' if l0 <= -12 else '%-3i' % l0
                res += ' '
                res += ' ' if l1 <= -15 else '.' if l1 <= -14 else '-' if l1 <= -13 else '*' if l1 <= -12 else '% 3i' % l1
                res += ']'
            #res += '[%-4i %-4i]' % (r0, r1)

        if res:
            report += '\n%s%s  src.zcap=%6.4f %s' % (indent, o.name, src.zcap, res)

    if report:
        Log( "POSSIBLE NUMERICAL ERROR:\nCheck Lens Equation (' ':-15  '.':-14  '-':-13  '*':-12) %s" % report )


@default_prior
@object_prior
def time_delay(o, leq, eq, geq):
    Log( indent + "Time Delay" )

    b  = o.basis
    nu = 1+b.H0

    pix_start,    pix_end    = 1+b.offs_pix
    srcpos_start, srcpos_end = 1+b.offs_srcpos

    zLp1 = (1 + o.z) * o.dL

    shft = [b.map_shift, b.map_shift]

    stellar_mass = find_stellar_mass(o)

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
            #row[0] += dot([x1-x0, y1-y0], shft)
            row[0] *= src.zcap

            # The beta term - this is not scaled by src.zcap because this 
            # source position already contains it. Later it will be factored
            # out in basis.py.
            row[srcpos:srcpos+2]  = x0-x1, y0-y1
            row[0] += dot([x1-x0, y1-y0], shft) * src.zcap
#           row[0] -= dot([x0-x1, y0-y1], shft)

            # The ln terms
            row[pix_start:pix_end] -= poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad)
            row[pix_start:pix_end] += poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad)

            sm = stellar_mass

            if o.stellar_mass_error != 0:
                sm = sm * sol[sm_err] # don't modified the original array with *=

            row[0] -= sum(sm * poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad))
            row[0] += sum(sm * poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad))

            for e,[start,end] in izip(o.extra_potentials, b.extra_potentials_array_offsets):
                start += 1
                end += 1

                p0 = e.poten(img0.pos)
                p1 = e.poten(img1.pos)
                coeff  = p1-p0
                row[start:end] -= coeff
                row[0] += e.shift * sum(coeff)

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
            #assert 0

@object_prior_check(time_delay)
def check_time_delay(o, sol):

    #return

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
    nu = b.H0

    pix_start,    pix_end    = b.offs_pix
    srcpos_start, srcpos_end = b.offs_srcpos

    zLp1 = (o.z + 1) * o.dL
    shft = [b.map_shift, b.map_shift]

    report = ''
    for i, src in enumerate(o.sources):
        res = ''
        for img0,img1,delay_years in src.time_delays:

            delay = [d / zLp1 if d else d for d in delay_years]

            srcpos = srcpos_start + 2*i

            r0, r1 = img0.pos, img1.pos
            x0, y0 = img0.pos.real, img0.pos.imag
            x1, y1 = img1.pos.real, img1.pos.imag

            # The constant term
            S  = (abs(r1)**2 - abs(r0)**2) / 2
            #S += dot([x1-x0, y1-y0], shft)
            S *= src.zcap

            # The beta term
            S += (x0-x1) * (sol[srcpos+0] - b.map_shift * src.zcap)
            S += (y0-y1) * (sol[srcpos+1] - b.map_shift * src.zcap)

            sm = o.stellar_mass

            if o.stellar_mass_error != 0:
                sm *= sol[sm_err]

            S -= sum((sm + sol[pix_start:pix_end]) * poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad))
            S += sum((sm + sol[pix_start:pix_end]) * poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad))

            for e,[start,end] in izip(o.extra_potentials, b.extra_potentials_array_offsets):
                p0 = e.poten(img0.pos)
                p1 = e.poten(img1.pos)
                coeff  = p1-p0
                S -= sum((sol[start:end] - e.shift) * coeff)

            S2 = None
            if len(delay) == 1:
                d = delay[0]
                if d is None: continue
                else:         S -= sol[nu] * d
            else:
                # Only report an error if the sum lies outside the desired time delay range
                l,u = delay
                if   l is None: S -= sol[nu] * u
                elif u is None: S -= sol[nu] * l
                else:
                    S2 = S
                    S  -= sol[nu] * l
                    S2 -= sol[nu] * u


            l0 = log10(abs(S))  if S  < 0 else -15
            l1 = log10(abs(S2)) if S2 > 0 else -15

            #res += '[%i %i]' % (r0,r1)

            if l0 >= -12 or l1 >= -12:
                res += '['
                res += ' ' if l0 <= -15 else '.' if l0 <= -14 else '-' if l0 <= -13 else '*' if l0 <= -12 else '%-3i' % l0
                res += ' '
                res += ' ' if l1 <= -15 else '.' if l1 <= -14 else '-' if l1 <= -13 else '*' if l1 <= -12 else '% 3i' % l1
                res += ']'

        if res:
            report += '\n%s%s  src.zcap=%6.4f %s' % (indent, o.name, src.zcap, res)

    if report:
        msg = "POSSIBLE NUMERICAL ERROR:\nCheck Time Delay (' ':-15  '.':-14  '-':-13  '*':-12) %s" % report
        Log(msg)
        print msg

#@object_prior
def JPC1time_delay(o, leq, eq, geq):
    Log( "JPC Time Delay" )

    b  = o.basis
    nu = 1+b.H0

    pix_start, pix_end = 1+b.pix_start, 1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end

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
                row[shear_start+0] -= shear.poten(0, img1)
                row[shear_start+0] += shear.poten(0, img0)
                row[shear_start+1] -= shear.poten(1, img1)
                row[shear_start+1] += shear.poten(1, img0)

            # TODO: Ptmass

            row[nu] = 0
            leq(row)

            #print row

@default_prior
@object_prior
def hubble_constant(o, leq, eq, geq):
    """This requires a particular hubble constant for the object."""

    if env().nu is None:
        Log( "[DISABLED] Hubble Constant")
        return

    Log( indent + "Hubble Constant" + str(env().nu))

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
    Log( indent + "J1 Image Parity" )

    b = o.basis

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end
    #shear_start,  shear_end  = 1+b.shear_start,  1+b.shear_end
    #ptmass_start, ptmass_end = 1+b.ptmass_start, 1+b.ptmass_end

#   print '*' * 80
#   print 'TURNED OFF PARITY MAXIMUM CONDITION'
#   print '*' * 80

    for i,src in enumerate(o.sources):
        for j,img in enumerate(src.images):
            parity = img.parity

            Log( 2*indent + "Source %i,Image %i: %s" % (i,j,img.parity_name) )

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


#@default_prior
@object_prior
def L1parity(o, leq, eq, geq):
    Log( "L1 Parity with saddle constraint" )

    b = o.basis

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,    pix_end    = 1+b.pix_start,    1+b.pix_end
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end

#   print '*' * 80
#   print 'TURNED OFF PARITY MAXIMUM CONDITION'
#   print '*' * 80

    for i,src in enumerate(o.sources):
        for img in src.images:
            parity = img.parity

            rows      = new_row(o,2)

            x, y = img.pos.real, img.pos.imag

            rows[0,0] = src.zcap * (x**2 + y**2)
            rows[1,0] = src.zcap * (y**2 + x**2)

            positions = img.pos - b.ploc
            #radial
            rows[0,pix_start:pix_end] = - ( x**2 * poten_dxdx(positions, o.basis.cell_size) \
                                      + 2 * x**2 * poten_dxdy(positions, o.basis.cell_size) \
                                          + y**2 * poten_dydy(positions, o.basis.cell_size) )
            #tangential
            rows[1,pix_start:pix_end] = - ( y**2 * poten_dxdx(positions, o.basis.cell_size) \
                                      - 2 * x**2 * poten_dxdy(positions, o.basis.cell_size) \
                                          + x**2 * poten_dydy(positions, o.basis.cell_size) )

            if o.shear:
                rows[0,shear_start:shear_end] = [ 2, 0]
                rows[1,shear_start:shear_end] = [-2, 0]

            if parity == MINIMUM:
                geq(rows[0])
                geq(rows[1])

            if parity == MAXIMUM:
                leq(rows[0])
                leq(rows[1])

            if parity == SADDLE:
                geq(rows[0])
                leq(rows[1])

@object_prior
def magnification(o, leq, eq, geq):

    Log( indent + "Magnification" )

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    pix_start,     pix_end = 1+o.basis.offs_pix

    coeffs = zeros(6)

    stellar_mass = find_stellar_mass(o)

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

            for i,r in enumerate(rows):
                r[0] += np.sum(r[pix_start:pix_end] * stellar_mass)

            for e,[start,end] in izip(o.extra_potentials, o.basis.extra_potentials_array_offsets):
                start += 1
                end += 1

                if not hasattr(e, 'maginv'): continue

                minv = e.maginv(img.pos, img.angle)
                for offs,[xy,xx,yy] in izip(xrange(start, end), minv):
                    if parity == MINIMUM: 
                        coeffs[0] = -k1*xx + yy
                        coeffs[1] = -k2*yy + xx
                        coeffs[2] =     xy + xx*eps
                        coeffs[3] =    -xy + xx*eps
                        coeffs[4] =     xy + yy*eps
                        coeffs[5] =    -xy + yy*eps

                    if parity == SADDLE:  
                        coeffs[0] = -k1*xx - yy
                        coeffs[1] =  k2*yy + xx
                        coeffs[2] =     xy + xx*eps
                        coeffs[3] =    -xy + xx*eps
                        coeffs[4] =     xy - yy*eps
                        coeffs[5] =    -xy - yy*eps

                    if parity == MAXIMUM: 
                        coeffs[0] =  k1*xx - yy
                        coeffs[1] =  k2*yy - xx
                        coeffs[2] =     xy - xx*eps
                        coeffs[3] =    -xy - xx*eps
                        coeffs[4] =     xy - yy*eps
                        coeffs[5] =    -xy - yy*eps

                    for i,r in enumerate(rows):
                        #rows[i, offs] += coeffs[i]
                        r[offs] += coeffs[i]
                        r[0] -= coeffs[i] * e.shift

            #print pix_start, pix_end
            #print "rows[:,0]", rows[:,0]

            for r in rows: leq(r)

@default_prior
@object_prior
def annular_density(o, leq, eq, geq):
    theta = o.prior_options.get('annular_density', None)
    if theta is None:
        Log( "[DISABLED] Annular density" )
        return

    Log( indent + "Annular density %s" % theta )

    pix_start, pix_end = 1+o.basis.offs_pix

    if theta is not None and theta != 0:
        for r in xrange(o.basis.inner_image_ring, o.basis.outer_image_ring):
            row = new_row(o)
            row[pix_start + o.basis.rings[r]] = -1
            row[0] = theta * len(o.basis.rings[r])
            eq(row)
        on = True

@default_prior
@object_prior
def external_shear(o, leq, eq, geq):
    on = None
    v = o.prior_options.get('shear', None)
    if v is not None:
        v = o.prior_options['shear']['strength']

        for e,[start, end] in izip(o.extra_potentials, o.basis.extra_potentials_array_offsets):
            if not isinstance(e, Shear): continue
            for s in xrange(1+start, 1+end):
                row = new_row(o)
                row[ [0,s] ] = v - e.shift, 1
                geq(row)
                row = new_row(o)
                row[ [0,s] ] = -v - e.shift, 1
                leq(row)
                #row = new_row(o)
                #row[ [0,s] ] = shear.shift + v, -1
                #leq(row)
                on = [v, -1]

    if on is None:
        on = '[DISABLED]'
    else:
        on = ''

    Log( "%10s External Shear" % on)

@default_prior
@object_prior
def external_mass(o, leq, eq, geq):
    on = None
    v = o.prior_options.get('external mass', None)
    if v is None:
        return

    #assert v < shear.shift

    for e,[start, end] in izip(o.extra_potentials, o.basis.extra_potentials_array_offsets):
        if not (hasattr(e.__class__, '__bases__') and ExternalMass in e.__class__.__bases__): continue
        min,max = o.prior_options['external mass'][e]
        print indent + 'External mass %s  %g-%g' % (e.name, min, max)
        for s in xrange(1+start, 1+end):
            if min == max and min is not None:
                row = new_row(o)
                row[ [0,s] ] = min, -1
                eq(row)
            else:
                if max is not None:
                    row = new_row(o)
                    row[ [0,s] ] = max, -1
                    geq(row)
                if min is not None:
                    row = new_row(o)
                    row[ [0,s] ] = min, -1
                    leq(row)
            on = [v, -1]

    if on is None:
        on = '[DISABLED]'
    else:
        on = ''

    Log( "%10s External Mass" % on)

##############################################################################

#@default_prior
@object_prior
def profile_steepness2(o, leq, eq, geq):
    steep = o.prior_options.get('steepness', None)

    if steep is None: 
        Log( "[DISABLED] Profile Steepness" )
        return

    Log( indent + "Profile Steepness %s" % steep )

    minsteep, maxsteep = steep
    assert maxsteep is None or maxsteep >= minsteep

    pix_start, pix_end = 1+o.basis.offs_pix

    nrings = len(o.basis.rings)
    row = new_row(o)

    #---------------------------------------------------------------------------
    # First handle the central pixel
    #---------------------------------------------------------------------------
#   r0,r1 = o.basis.rings[0:2]
#   lc = (0.5) ** minsteep
#   lpc = (1.5) ** minsteep
#   row[pix_start+r0] = lc / len(r0)
#   row[pix_start+r1] = -lpc / len(r1)
#   #print r0,r1
#   #print row
#   c=1
#   geq(row)

    c = 0

    #---------------------------------------------------------------------------
    # Now the rest of the rings.
    #---------------------------------------------------------------------------
    for l in xrange(0,nrings-1):
        r0 = o.basis.rings[l]

        r0 = o.basis.rings[0]
        r1 = o.basis.rings[l+1]

        row = new_row(o)
        #lc  = (l+0.5) ** minsteep
        #lpc = (l+1.5) ** minsteep
        row[pix_start+r0] =  1  / len(r0)
        row[pix_start+r1] = -minsteep / len(r1)

        if minsteep == maxsteep:
            eq(row)
        else:
            geq(row)
            if maxsteep is not None:
                row = new_row(o)
                #lc  = (l+0.5) ** maxsteep
                #lpc = (l+1.5) ** maxsteep
                row[pix_start+r0] =  1  / len(r0)
                row[pix_start+r1] = -maxsteep / len(r1)
                leq(row)

@default_prior
@object_prior
def profile_steepness(o, leq, eq, geq):
    steep = o.prior_options.get('steepness', None)

    if steep is None: 
        Log( "[DISABLED] Profile Steepness" )
        return

    Log( indent + "Profile Steepness %s" % steep )

    minsteep, maxsteep = steep

    assert maxsteep is None or maxsteep >= minsteep

    pix_start, pix_end = 1+o.basis.offs_pix

    nrings = len(o.basis.rings)

    c = 0

    for l,[r0,r1] in enumerate(izip(o.basis.rings[:-1], o.basis.rings[1:])):

        if 1:
        #if not o.basis.hires_levels \
        #or (o.basis.hires_levels and l < (o.basis.hiresR*o.basis.hires_levels - o.basis.hires_levels//2)):

            R0 = (o.basis.rs[l]   + o.basis.radial_cell_size[l]/2)
            R1 = (o.basis.rs[l+1] + o.basis.radial_cell_size[l+1]/2)

            #print R0,R1, o.basis.radial_cell_size[l], o.basis.radial_cell_size[l+1]

            #R0 = l
            #R1 = l+1

            #if l == 0:
                #R0,R1 = 1,1

            #print o.basis.cell_size[r0]
            w0 = o.basis.cell_size[r0]**2 / np.sum(o.basis.cell_size[r0]**2)
            w1 = o.basis.cell_size[r1]**2 / np.sum(o.basis.cell_size[r1]**2)

            row = new_row(o)
            row[pix_start+r0] =  w0 #* R0**minsteep
            row[pix_start+r1] = -w1 #* R1**minsteep
            #print row[pix_start+r0]
            #print row[pix_start+r1]
            #print 

            if minsteep == maxsteep:
                eq(row)
            else:
                geq(row)

                if l > 0 and maxsteep is not None:
                    row = new_row(o)
                    row[pix_start+r0] =  w0 * R0**maxsteep
                    row[pix_start+r1] = -w1 * R1**maxsteep
                    leq(row)
                    c += 1
            c += 1

    Log( 2*indent + "# eqs = %i" % c )

@default_prior
@object_prior
def profile_steepnessXXX(o, leq, eq, geq):
    steep = o.prior_options.get('steepness', None)

    if steep is None: 
        Log( "[DISABLED] Profile Steepness" )
        return

    Log( indent + "Profile Steepness %s" % steep )

    minsteep, maxsteep = steep

    assert maxsteep is None or maxsteep >= minsteep

    pix_start, pix_end = 1+o.basis.offs_pix

    nrings = len(o.basis.rings)

    c = 0

    for l,[r0,r1,r2] in enumerate(izip(o.basis.rings[:-1], o.basis.rings[1:], o.basis.rings[2:])):

        if not o.basis.hires_levels \
        or (o.basis.hires_levels and l < (o.basis.hiresR*o.basis.hires_levels - o.basis.hires_levels//2)):

            w0 = o.basis.cell_size[r0]**2 / sum(o.basis.cell_size[r0]**2)
            w1 = o.basis.cell_size[r1]**2 / sum(o.basis.cell_size[r1]**2)
            w2 = o.basis.cell_size[r2]**2 / sum(o.basis.cell_size[r2]**2)

            row = new_row(o)
            row[pix_start+r0] =  w0
            row[pix_start+r1] = -2*w1
            row[pix_start+r2] =  w2

            if minsteep == maxsteep:
                eq(row)
            else:
                geq(row)

                if l > 0 and maxsteep is not None:
                    row = new_row(o)
                    row[pix_start+r0] =  w0
                    row[pix_start+r1] = -w1
                    leq(row)
                    c += 1
            c += 1

    Log( 2*indent + "# eqs = %i" % c )

@object_prior
def PLprofile_steepness(o, leq, eq, geq):
    steep = o.prior_options.get('steepness', None)

    if steep is None: 
        Log( "[DISABLED] Profile Steepness" )
        return

    Log( indent + "Profile Steepness %s" % steep )

    minsteep, maxsteep = steep
    assert maxsteep is None or maxsteep >= minsteep

    pix_start, pix_end = 1+o.basis.offs_pix

    nrings = len(o.basis.rings)
    row = new_row(o)

    #---------------------------------------------------------------------------
    # First handle the central pixel
    #---------------------------------------------------------------------------
    r0,r1 = o.basis.rings[0:2]
    lc = (0.0) ** minsteep
    lpc = (1) ** minsteep
    row[pix_start+r0] = lc / len(r0)
    row[pix_start+r1] = -lpc / len(r1)
    #print r0,r1
    #print row
    c=1
    geq(row)

    c = 0

    #---------------------------------------------------------------------------
    # Now the rest of the rings.
    #---------------------------------------------------------------------------
    for l in xrange(1,nrings-1):
        r0 = o.basis.rings[l]
        r1 = o.basis.rings[l+1]

        row = new_row(o)
        lc  = (l) ** minsteep
        lpc = (l+1) ** minsteep
        row[pix_start+r0] =  lc  / len(r0)
        row[pix_start+r1] = -lpc / len(r1)

        if minsteep == maxsteep:
            eq(row)
        else:
            geq(row)

    if maxsteep is not None and maxsteep > minsteep:
        row = zeros(1+o.basis.nvar)
        r0 = o.basis.rings[1]
        r1 = o.basis.rings[-1]

        lc  = 1
        lpc = (nrings) ** maxsteep
        row[pix_start+r0] = -lc  / len(r0)
        row[pix_start+r1] =  lpc / len(r1)
        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )
        
#@default_prior
@object_prior
def gradient(o, leq, eq, geq):
    cen_ang = o.prior_options.get('gradient', pi/4)

    Log( "Gradient %s" % cen_ang )
    pix_start, pix_end = 1+o.basis.offs_pix

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

    Log( 2*indent + "# eqs = %i" % c )
    Log( 2*indent + "sn=%g" % sn )

#@default_prior
@object_prior
def Pgradient(o, leq, eq, geq):
    cen_ang = o.prior_options.get('gradient', pi/4)

    Log( "PGradient %s" % cen_ang )

    pix_start, pix_end = 1+o.basis.offs_pix

    cs = cos(cen_ang)
    sn = sin(cen_ang)
    c = 0
    for ri,r,nbrs in o.basis.nbrs2:
        if ri == o.basis.central_pixel: continue

        #print sort(nbrs1)

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

    Log( 2*indent + "gradient eqs = %i" % c )
    Log( 2*indent + "sn=%g" % sn )

#@default_prior
@object_prior
def J2gradient(o, leq, eq, geq):

    opts = o.prior_options.get('J2Gradient', None)

    if not opts:
        opts = {}
        #Log( 'J2Gradient NOT ACTIVE' )
        #return

    #Lmin = np.sqrt(2)*o.basis.top_level_cell_size
    #Lmin = 1.1*o.basis.top_level_cell_size

    theta = opts.get('theta', 45)
    size  = opts.get('size',  None)

    #XXX assert (L >= Lmin), 'size=%f < %f is too small' % (L, Lmin)

    Log( indent + "J2Gradient (theta=%.2f  size=%s)" % (theta, size) )
    #Log( indent + "J2Gradient (theta=%.2f  size=%.2f)" % (theta, size) )

    pix_start, pix_end = 1+o.basis.offs_pix

    theta = radians(90-theta)
    cs = cos(theta)
    sn = sin(theta)
    c = 0
    for i,[ri,r] in enumerate(izip(o.basis.int_ploc, o.basis.ploc)):
        #if i in o.basis.rings[-2]: break
        if i == o.basis.central_pixel: continue

        if size is None:
            L = 1.1*o.basis.cell_size[i]
        else:
            L = size

        #nbrs = neighbors(r,L, o.basis.ploc)
        nbrs = o.basis.nbrs2[i][2]

        #if o.basis.int_cell_size[i] < 1 and len(nbrs) < 4: continue
        #if o.basis.int_cell_size[i] < 1 and np.any(o.basis.int_cell_size[nbrs] > o.basis.int_cell_size[i]): continue
        #if np.any(o.basis.int_cell_size[nbrs] != o.basis.int_cell_size[i]): continue

        #if len(nbrs) < 4: continue
        #if np.any(o.basis.cell_size[nbrs] != o.basis.cell_size[i]): continue
        #if np.any(o.basis.int_cell_size[nbrs] != o.basis.cell_size[i]): continue

        px = r.real
        py = r.imag
        x = cs*px - sn*py
        y = sn*px + cs*py

        #dr = ri - o.basis.int_ploc[nbrs]
        dr = r - o.basis.ploc[nbrs]
        dir = dr / abs(dr)

        dW = abs(dr)*(L + o.basis.cell_size[i] - abs(dr))
        #dW = abs(dr)*(L + o.basis.top_level_cell_size - abs(dr))
        #dW = abs(dr)*(o.basis.grad_rmax + 1 - abs(dr))

        wght = (o.basis.cell_size[nbrs] / amax(o.basis.cell_size[nbrs]))**2
        #wght = amax([wght, o.basis.cell_size[i]])

        wght = o.basis.cell_size[nbrs]**2 / sum(o.basis.cell_size[nbrs]**2)
        #wght = 1 #(o.basis.cell_size[nbrs]/o.basis.cell_size[i])**2
        row = new_row(o)
        row[pix_start + nbrs] = dW * (dir * complex(x,-y)).real * wght
        geq(row)
        c += 1

        if sn != 0:
            x =  cs*px + sn*py
            y = -sn*px + cs*py
            row = new_row(o)
            row[pix_start + nbrs] = dW * (dir * complex(x,-y)).real * wght
            geq(row)
            c += 1

#   i = o.basis.rings[-1]
#   for j,k in zip(i[:-1], i[1:]):
#       row = new_row(o)
#       row[ [0, pix_start+j, pix_start+k] ] = -1e-2, 1, -1
#       leq(row)
#   row = new_row(o)
#   row[ [0, pix_start+i[0], pix_start+i[-1]] ] = -1e-2, 1, -1
#   leq(row)
        

    Log( 2*indent + "gradient eqs = %i" % c )
    Log( 2*indent + "sn=%g" % sn )

#@default_prior
@object_prior
def J3gradient(o, leq, eq, geq):

    opts = o.prior_options.get('J3gradient', None)

    if not opts:
        opts = {}
        #Log( 'J2Gradient NOT ACTIVE' )
        #return

    #Lmin = np.sqrt(2)*o.basis.top_level_cell_size
    #Lmin = 1.1*o.basis.top_level_cell_size

    theta = opts.get('theta', 45)
    size  = opts.get('size',  None)

    #XXX assert (L >= Lmin), 'size=%f < %f is too small' % (L, Lmin)

    Log( indent + "J3Gradient (theta=%.2f  size=%s)" % (theta, size) )

    pix_start, pix_end = 1+o.basis.offs_pix

    phi = radians(90-theta)
    cs,sn = cos(phi), sin(phi)
    c = 0

    rs = [ abs(img.pos) for src in o.sources for img in src.images if img.parity_name != 'max']
    rmin, rmax = amin(rs), amax(rs)

    for i,[ri,r] in enumerate(izip(o.basis.int_ploc, o.basis.ploc)):
        if i == o.basis.central_pixel: continue

#       for j,n in enumerate(o.basis.image_nbrs):
#           #for nn in n[2]:
#           if i == n:
#               print 'F', j,i, n
#               #break
#       else:
        if 1:
            n,e,s,w = o.basis.nbrs3[i][2]

            px = r.real
            py = r.imag

            def wght(x):
                _wght = lambda x: o.basis.cell_size[x]**2 / sum(o.basis.cell_size[x]**2) if len(x) else 0
                #_wght = lambda x: 1.0 / len(x) if len(x) else 0
                return _wght(x)

            for i in ([1,-1] if sn != 0 else [1]):

                x =   cs*px - i*sn*py
                y = i*sn*px +   cs*py

                row = new_row(o)

                row[pix_start + w] =  wght(w) * x
                row[pix_start + e] = -wght(e) * x
                row[pix_start + s] =  wght(s) * y
                row[pix_start + n] = -wght(n) * y
                geq(row)
                c += 1

    Log( 2*indent + "gradient eqs = %i" % c )
    Log( 2*indent + "sn=%g" % sn )

#@default_prior
@object_prior
def central_pixel_as_maximum(o, leq, eq, geq):
    cp = o.basis.central_pixel + 1+o.basis.offs_pix[0]
    Log( "Central pixel as maximum %i" % cp )

    pix_start, pix_end = 1 + o.basis.offs_pix
    for i in xrange(pix_start, pix_end):
        if i == cp: continue
        row = new_row(o)
        row[ [cp,i] ] = 1, -1
        geq(row)

@object_prior
def central_pixel_max(o, leq, eq, geq):

    cp = o.basis.central_pixel + 1+o.basis.offs_pix[0]
    Log( "Central pixel maximum %i" % cp )

#   g = o.prior_options.get('central_pixel_maximum')
#   M     = g['M']
#   H0inv = g['H0inv']
#   nu    = g['nu']

#   max_nu = env().nu[-1] # should this be min?

#   M = convert('Msun/kpc^2 to kappa',  M, o.dL, max_nu)

    row = new_row(o)
    row[ [0,cp] ] = -7, 1
    leq(row)

#@default_prior
@object_prior
def central_pixel_min(o, leq, eq, geq):

    cp = o.basis.central_pixel + 1+o.basis.offs_pix[0]
    Log( "Central pixel minimum %i" % cp )

    #g = o.prior_options.get('central_pixel_minimum')
    #M     = g['M']
    #H0inv = g['H0inv']
    #nu    = g['nu']
#
    #max_nu = env().nu[-1] # should this be min?
##
    #M = convert('Msun/kpc^2 to kappa',  M, o.dL, max_nu)

    row = new_row(o)
    row[ [0,cp] ] = -40, 1
    #row[ [0,cp] ] = -M, 1
    geq(row)

#@default_prior
@object_prior
def ring_smoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of its ring neighbours."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "[DISABLED] Ring smoothness" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "Ring smoothness (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    if include_central_pixel:
        row = new_row(o)
        ring = o.basis.rings[1]

        row[pix_start + ring] = 1
        row[pix_start + o.basis.central_pixel] = -len(ring) / smoothness_factor
        c += 1

    for ring in o.basis.rings[1:]:
        for i,cntr in enumerate(ring):

            row = new_row(o)
            left = ring[(i-1) % len(ring)]
            rght = ring[(i+1) % len(ring)]

            row[pix_start + left] = 1
            row[pix_start + rght] = 1
            row[pix_start + cntr] = -2 / smoothness_factor

            geq(row)
            c += 1

    Log( 2*indent + "# eqs = %i" % c )

#@default_prior
@object_prior
def PLsmoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "[DISABLED] Smoothness" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "Smoothness (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    for i,r,nbrs in o.basis.nbrs:
        if not include_central_pixel and i == o.basis.central_pixel: continue

        row = new_row(o)
        #w0 = o.basis.cell_size[nbrs]**2 / sum(o.basis.cell_size[nbrs]**2)

        row[pix_start + nbrs] = 1
        row[pix_start + i]    = -len(nbrs) / smoothness_factor

        #print 'PLs', row[row != 0]
        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )

#@default_prior
@object_prior
def PLsmoothness2(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "[DISABLED] PLSmoothness2" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "PLSmoothness2 (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    wght = lambda x: 1.0 / len(x) if len(x) else 0
    for i,r,nbrs in o.basis.nbrs:
        if not include_central_pixel and i == o.basis.central_pixel: continue

        row = new_row(o)
        for d in o.basis.nbrs3[i][2]:
            row[pix_start + d] = wght(d)
        row[pix_start + i] = -1 / smoothness_factor

        #print 'PLs', row[row != 0]
        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )

#@default_prior
@object_prior
def PLsmoothness3(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "[DISABLED] PLSmoothness3" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "PLSmoothness3 (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    #wght = lambda x: 1.0 / len(x) if len(x) else 0
    wght = lambda x: o.basis.cell_size[x]**2 / sum(o.basis.cell_size[x]**2)
    nbr_list = o.basis.nbrs

    #print 'In smooth!'

    for i,r,nbrs in nbr_list:
        if not include_central_pixel and i == o.basis.central_pixel: continue

        row = new_row(o)
        #N = sum( (sum(wght(d)) for d in o.basis.nbrs3[i][2]) )
        l = len(nbr_list[i][2])
#       for d in nbr_list[i][2]:
#           if d == o.basis.central_pixel:
#               l -= 1

        W = 0
        for d in nbr_list[i][2]:
            #if d == o.basis.central_pixel: continue
            #print len(o.basis.nbrs3[i][2])
            #print i, o.basis.nbrs3[i][2]
            #row[pix_start + d] = 1. / 8
            w = 1
            W += 1 #np.sum(w)
            row[pix_start + d] = w * wght(d)

        row[pix_start + i] = -W / smoothness_factor

        #print 'PLs', row[row != 0]
        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )

#@default_prior
@object_prior
def PLsmoothnessExp(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "[DISABLED] PLSmoothness3" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "PLSmoothness3 (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    #wght = lambda x: 1.0 / len(x) if len(x) else 0
    wght = lambda x: o.basis.cell_size[x]**2 / sum(o.basis.cell_size[x]**2)
    nbr_list = o.basis.nbrs

    #print 'In smooth!'

    for i,r,nbrs in nbr_list:
        if not include_central_pixel and i == o.basis.central_pixel: continue

        row = new_row(o)
        #N = sum( (sum(wght(d)) for d in o.basis.nbrs3[i][2]) )
        l = len(nbr_list[i][2])
#       for d in nbr_list[i][2]:
#           if d == o.basis.central_pixel:
#               l -= 1

        W = 0
        for d in nbr_list[i][2]:
            #if d == o.basis.central_pixel: continue
            #print len(o.basis.nbrs3[i][2])
            #print i, o.basis.nbrs3[i][2]
            #row[pix_start + d] = 1. / 8
            w = np.exp(-(abs(o.basis.ploc[d] - r)/o.basis.top_level_cell_size)/2.)
            W += 1 #np.sum(w)
            row[pix_start + d] = w * wght(d)

        row[pix_start + i] = -W / smoothness_factor

        #print 'PLs', row[row != 0]
        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )

@object_prior
def pixel_smoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "[DISABLED] pixel_smoothness" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "PLSmoothness2 (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    wght = lambda x: 1.0 / len(x) if len(x) else 0
    for i,r,nbrs in o.basis.nbrs:
        if i != o.basis.central_pixel: continue
        #if not include_central_pixel and i == o.basis.central_pixel: continue

        for d in o.basis.nbrs3[i][2]:
            row = new_row(o)
            row[pix_start + d] = wght(d)
            row[pix_start + i] = -1 / smoothness_factor

        #print 'PLs', row[row != 0]
        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )

@object_prior
def JCsmoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {'factor': 2, 'include_central_pixel': True})
    if not smth:
        Log( "JCSmoothness DISABLED" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
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

    Log( 2*indent + "# eqs = %i" % c )

#@default_prior
@object_prior
def smoothness(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbouring pixels."""

    smth = o.prior_options.get('smoothness', {})

    Lmin = 1.5*o.basis.top_level_cell_size

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    L                     = smth.get('L', Lmin)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "Smoothness (factor=%.1f L=%.1f include_central_pixel=%s)" % (smoothness_factor, L, include_central_pixel) )
    Log( indent + "Smoothness factor decreases with radius" )

    c=0
    for i,[ri,r] in enumerate(izip(o.basis.int_ploc, o.basis.ploc)):
        if not include_central_pixel and i == o.basis.central_pixel: continue

        nbrs = neighbors(r,L, o.basis.ploc)

        row = new_row(o)
        row[pix_start + nbrs] = 1
        #row[pix_start + i]    = -len(nbrs) / ((smoothness_factor-1) * (1 - abs(ri) / o.basis.pixrad) + 1)
        row[pix_start + i]    = -len(nbrs) / smoothness_factor

        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )

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
        Log( "[DISABLED] Shared h" )
    else:
        Log( indent + "Shared h" )


@object_prior
def max_kappa(o, leq, eq, geq):
    Log( "Maximum Kappa" )
    pix_start, pix_end    = 1+o.basis.offs_pix
    for i in xrange(pix_start, pix_end):
        row = new_row(o)
        row[ [0,i] ] = -6, 1
        leq(row)

    
@object_prior
def min_kappa(o, leq, eq, geq):
    g = o.prior_options.get('min_kappa')

    if not g: 
        Log( "[DISABLED] Minimum Kappa" )
        return

    Log( indent + "Minimum Kappa" )

    v = g['kappa']

    pix_start, pix_end    = 1+o.basis.offs_pix
    for i in xrange(pix_start, pix_end):
        row = new_row(o)
        row[ [0,i] ] = -v, 1
        geq(row)

@object_prior
def min_annular_density(o, leq, eq, geq):
    g = o.prior_options.get('min_annular_density', None)
    if g is None:
        Log( "[DISABLED] Minimum annular density" )
        return

    theta = g['v']

    Log( indent + "Minimum annular density %s" % theta )

    pix_start, pix_end = 1+o.basis.offs_pix

    if theta is not None and theta > 0:
        for r in o.basis.rings:
            row = new_row(o)
            row[pix_start + r] = 1
            row[0] = -theta * len(r)
            geq(row)
        on = True

#@object_prior
def min_kappa_particles(o, leq, eq, geq):

    g = o.prior_options.get('min_kappa_particles')

    if not g: 
        Log( "[DISABLED] Minimum Kappa (Particles)" )
        return

    Log( indent + "Minimum Kappa (Particles)" )

    X,Y,M = g['particles']
    H0inv = g['H0inv']
    nu    = g['nu']

    phys_cell_size = convert('arcsec to kpc', o.basis.cell_size, o.dL, nu)
    phys_ploc = convert('arcsec to kpc', o.basis.ploc, o.dL, nu)
    h = irrhistogram2d(-Y, X, phys_ploc, phys_cell_size, weights=M) / phys_cell_size**2
    h *= convert('Msun/kpc^2 to kappa',  1., o.dL, nu)
    if 0:
        import pylab as pl
        pl.matshow(o.basis._to_grid(h,o.basis.subdivision)) #, vmin=0, vmax=4)
        pl.colorbar()
        #Rmap = o.basis.mapextent
        #rx = convert('arcsec to kpc', Rmap, o.dL, nu)
        #ry = convert('arcsec to kpc', Rmap, o.dL, nu)
        #g = pl.histogram2d(-Y, X, bins=21, weights=M, range=[[-ry,ry], [-rx,rx]])[0]
        #pl.matshow(g) #, vmin=0, vmax=4)
        #pl.colorbar()
        pl.show()
        assert 0

    pix_start, pix_end = 1+o.basis.offs_pix
    for i,j in enumerate(xrange(pix_start, pix_end)):
        row = new_row(o)
        row[ [0,j] ] = -h[i], 1
        geq(row)

#@object_prior
def Xmin_kappa_grid(o, leq, eq, geq):

    g = o.prior_options.get('minkappa')

    if not g: 
        Log( "[DISABLED] Minimum Kappa Grid" )
        return

    Log( indent + "Minimum Kappa Grid" )

    X,Y,M = g['grid']
    H0inv = g['H0inv']

    g = o.basis.grid_mass(X,Y,M, H0inv) * H0inv

    pix_start, pix_end = 1+o.basis.offs_pix
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

    pix_start, pix_end = 1+o.basis.offs_pix

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

    pix_start, pix_end    = 1+o.basis.offs_pix
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

    Log( 2*indent + "# eqs = %i" % c )

#@default_prior
#@object_prior
def symmetry(o, leq, eq, geq):

    if not o.symm:
        Log( "[DISABLED] Symmetry" )
        return

    Log( indent + "Symmetry" )

    pix_start, pix_end = 1+o.basis.offs_pix

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

    Log( 2*indent + "# eqs = %i" % c )


#@default_prior
@object_prior
def max_kappa(o, leq, eq, geq):

    Log( "Maximum Kappa" )

    pix_start, pix_end = 1+o.basis.offs_pix

    for i,j in enumerate(xrange(pix_start, pix_end)):
        row = new_row(o)
        row[ [0,j] ] = 100, -1
        leq(row)

#@default_prior
@object_prior
def smooth_symmetry(o, leq, eq, geq):
    """A pixel cannot be more that twice the average of the neighbours of the opposing pixel."""

    smth = o.prior_options.get('smoothness', {'factor': 3})
    if not smth:
        Log( "[DISABLED] Smooth Symmetry" )
        return

    pix_start, pix_end    = 1+o.basis.offs_pix
    smoothness_factor     = smth.get('factor', 2)
    include_central_pixel = smth.get('include_central_pixel', True)

    Log( indent + "Smooth Symmetry (factor=%.1f include_central_pixel=%s)" % (smoothness_factor, include_central_pixel) )

    c=0
    for i,j in enumerate(o.basis.oppose):
        if i == o.basis.central_pixel: continue

        _,r,nbrs = o.basis.nbrs[j]


        row = new_row(o)
        row[pix_start + nbrs] = 1
        row[pix_start + i]    = -len(nbrs) / smoothness_factor

        geq(row)
        c += 1

    Log( 2*indent + "# eqs = %i" % c )

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

def load_leier_grid(o, fname, grid_size, units, H0inv, scale=1.0):

    if units != 'arcsec':
        grid_size = convert('%s to arcsec' % units, grid_size, o.dL, max_nu)

    data = loadtxt(fname, unpack=True)
    r,c = amax(data[0])+1, amax(data[1])+1
    assert r==c
    grid = data[5] * 1e10 * scale
    grid = grid.reshape((r,c)).T        # Data is stored in column-major order!
    grid = flipud(grid)

#   print data[11] / data[12]
#   from np import cos,sin,arctan2, arctan, angle
#   xs = cos(angle(data[1] + 1j*data[0] - 15-15j)) * data[12]
#   ys = sin(angle(data[1] + 1j*data[0] - 15-15j)) * data[12]

    if 0:
        import pylab as pl

        G = grid.copy()
        G *= convert('Msun/arcsec^2 to kappa',  1., o.dL, convert('H0^-1 in Gyr to nu', H0inv, o.dL))
        R = grid_size / 2
        pl.matshow(pl.log10(G), extent=[-R,R,-R,R], origin='upper')
        pl.colorbar()


    Log( 'Projecting mass onto grid.' )
    grid *= (grid_size / grid.shape[0])**2
    a = o.basis.project_grid(grid, grid_size, H0inv)
    a /= o.basis.cell_size**2
    a *= convert('Msun/arcsec^2 to kappa',  1., o.dL, convert('H0^-1 in Gyr to nu', H0inv, o.dL))

    #print 'stellar', a

    #a = o.basis._to_grid(a)

    if 0:
        G = o.basis._to_grid(a)
        R = o.basis.maprad
        pl.matshow(pl.log10(G), extent=[-R,R,-R,R], origin='upper')
        pl.contour(pl.log10(G), extent=[-R,R,-R,R], origin='upper')
        pl.colorbar()
        pl.show()

    #a = o.basis.from_grid(a)
    #a[0] = np.mean(a[1:5])
    o.stellar_mass = a


    #a[0] = 0
    #ps = o.basis.solution_from_array(a, [[0,0]], H0inv=13.7)
    #m  = package_solution(ps, [o])

#   from funcs import default_post_process
#   default_post_process(m['obj,data'][0])
#   kappa_plot(m,0) #{'obj,data': [[o,m]]},0)
#   pl.figure(); glplot([m], 'kappa(R)', ['R', 'arcsec'], mark_images=True)
#   pl.show()
    #pl.show()
    #assert 0


#@object_prior
def min_kappa_leier_grid(o, leq, eq, geq):

    #assert 0

    g = o.prior_options.get('minkappa Leier grid')

    if not g: 
        Log( "[DISABLED] Minimum Kappa Leier Grid" )
        return

    Log( indent + "Minimum Kappa Leier Grid" )

    fname = g['filename']
    grid_size = g['grid size']
    units = g['grid size units']
    scale = g['scale']
    max_nu = env().nu[-1]
    H0inv = convert('nu to H0^-1 in Gyr', env().nu[-1], o.dL)

    load_leier_grid(o, fname, grid_size, units, H0inv, scale)

    pix_start, pix_end = 1+o.basis.offs_pix
    for i,j in enumerate(xrange(pix_start, pix_end)):
        row = new_row(o)
        row[ [0,j] ] = -o.stellar_mass[i], 1
        geq(row)

#@default_prior
@object_prior
def extended_source_size(o, leq,eq,geq):
    d = o.prior_options['extended source size']
    b = o.basis
    srcpos_start, srcpos_end = 1+b.srcpos_start, 1+b.srcpos_end

    for i1 in xrange(len(o.sources[:-1])):
        i2 = i1 + 1
        x1 = srcpos_start + 2*i1 + 0
        x2 = srcpos_start + 2*i2 + 0
        y1 = srcpos_start + 2*i1 + 1
        y2 = srcpos_start + 2*i2 + 1

        row = new_row(o,4)
        row[0, [0,x1,x2]] = -d, 1, -1; leq(row[0])
        row[1, [0,x1,x2]] = +d, 1, -1; geq(row[1])
        row[2, [0,y1,y2]] = -d, 1, -1; leq(row[2])
        row[3, [0,y1,y2]] = +d, 1, -1; geq(row[3])

@default_prior
@object_prior
def source_position(o, leq,eq,geq):
    b = o.basis
    srcpos_start, srcpos_end = 1+b.offs_srcpos

    for i,src in enumerate(o.sources):
        if src.pos is None: continue

        x = srcpos_start + 2*i + 0
        y = srcpos_start + 2*i + 1

        sx = src.pos.real
        sy = src.pos.imag
        tol = src.pos_tol

        print x,y

        print tol
        if tol != 0:
            lb = src.zcap * (b.map_shift + sx - tol)
            ub = src.zcap * (b.map_shift + sx + tol)
            row = new_row(o,2)
            row[0, [0,x]] = lb, -1; leq(row[0])
            row[1, [0,x]] = ub, -1; geq(row[1])

            print lb, ub

            lb = src.zcap * (b.map_shift + sy - tol)
            ub = src.zcap * (b.map_shift + sy + tol)
            row = new_row(o,2)
            row[0, [0,y]] = lb, -1; leq(row[0])
            row[1, [0,y]] = ub, -1; geq(row[1])

            print lb, ub
        else:
            row = new_row(o,2)
            row[0, [0,x]] = sx, -1; eq(row[0])
            row[1, [0,y]] = sy, -1; eq(row[1])

@default_prior
@object_prior
def link_sources(o, leq,eq,geq):
    b = o.basis
    srcpos_start, srcpos_end = 1+b.offs_srcpos

    w = o.prior_options['link sources']

    for sources, tol in w.values():
        xs, ys = [], []
        for src in sources:
            xs.append(srcpos_start + 2*src.index + 0)
            ys.append(srcpos_start + 2*src.index + 1)

        for src in sources:

            i = src.index
            x = srcpos_start + 2*i + 0
            y = srcpos_start + 2*i + 1

            for X, Xs in [[x,xs], [y,ys]]:
                row = new_row(o)

                row[0]   = -tol * src.zcap
                row[X]  -= 1
                row[Xs] += 1 / len(sources)
                leq(row)

                row = new_row(o)
                row[0]   = tol * src.zcap
                row[X]  -= 1
                row[Xs] += 1 / len(sources)
                geq(row)

@object_prior_check(link_sources)
def check_link_sources(o, sol):
    b = o.basis
    srcpos_start, srcpos_end = 1+b.offs_srcpos

    w = o.prior_options['link sources']

    for sources, tol in w.values():
        xs, ys = [], []
        for src in sources:
            xs.append(srcpos_start + 2*src.index + 0)
            ys.append(srcpos_start + 2*src.index + 1)

        for src in sources:

            i = src.index
            x = srcpos_start + 2*i + 0
            y = srcpos_start + 2*i + 1

            for X, Xs in [[x,xs], [y,ys]]:

                r0  = -tol * src.zcap
                r0 -= sol[X]
                r0 += np.sum(sol[Xs]) / len(sources)
                #assert r0 <= 0, r0

                r1  = tol * src.zcap
                r1 -= sol[X]
                r1 += np.sum(sol[Xs]) / len(sources)
                #assert r1 >= 0, r1
                #print r0, r1

if __name__ == '__main__':
    import pylab as pl
    import matplotlib as mpl
    fig=pl.figure()
    ax = fig.add_subplot(111)
    def rect(R, fc='none'):
        t,b,l,r = R
        return mpl.patches.Rectangle( (l,b), width=r-l, height=t-b, fc=fc)

    B = [10,0,0,10]
    A = [1,0.5,0.5,1.5]
    ax.add_artist(rect(A))
    ax.add_artist(rect(B))
    ax.add_artist(rect(intersect(A,B), fc=None))

    B = [8,5,4,8]
    A = [7,6,6,9]
    ax.add_artist(rect(A))
    ax.add_artist(rect(B))
    ax.add_artist(rect(intersect(A,B), fc='r'))

    B = [4,3,3,4]
    A = [3.5,2,3.5,6]
    ax.add_artist(rect(A))
    ax.add_artist(rect(B))
    ax.add_artist(rect(intersect(A,B), fc='r'))

    A = [3.5,2,1,3.2]
    ax.add_artist(rect(A))
    ax.add_artist(rect(intersect(A,B), fc='r'))

    A = [4.2,3.8,1,3.2]
    ax.add_artist(rect(A))
    ax.add_artist(rect(intersect(A,B), fc='r'))

    A = [4.2,3.8,3.5,6]
    ax.add_artist(rect(A))
    ax.add_artist(rect(intersect(A,B), fc='r'))

    pl.xlim(-1,11)
    pl.ylim(-1,11)
    pl.show()




