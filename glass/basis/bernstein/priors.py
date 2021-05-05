
if __name__ != '__main__':
    from glass.environment import env
    from glass.command import command
    from glass.shear import Shear
    from glass.exmass import ExternalMass
    
    from glass.log import log as Log, dlog as DLog
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

    s = ('minkappa Leier grid' in o.prior_options) \
      + ('min_kappa_particles' in o.prior_options)

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

    bcoeff_start, bcoeff_end = 1+b.offs_bcoeff
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
            p_dx      = poten_dx(positions, b.cell_size, b.maprad)
            p_dy      = poten_dy(positions, b.cell_size, b.maprad)

            for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                rows[0,ri] = -np.sum(p_dx * _g) #n[_n] * b.gm[_m])
                rows[1,ri] = -np.sum(p_dy * _g) #n[_n] * b.gm[_m])

            #rows[0,pix_start:pix_end] = -poten_dx(positions, b.cell_size, b.maprad)
            #rows[1,pix_start:pix_end] = -poten_dy(positions, b.cell_size, b.maprad)

            srcpos = srcpos_start + 2*i
            rows[0,srcpos:srcpos+2] = -1,  0
            rows[1,srcpos:srcpos+2] =  0, -1

#           sm_x = np.sum(poten_dx(positions, b.cell_size, b.maprad) * stellar_mass)
#           sm_y = np.sum(poten_dy(positions, b.cell_size, b.maprad) * stellar_mass)

#           if o.stellar_mass_error != 0:
#               rows[0,sm_err] = -sm_x
#               rows[1,sm_err] = -sm_y
#           else:
#               rows[0,0] -= sm_x
#               rows[1,0] -= sm_y

#           for e,[start,end] in zip(o.extra_potentials, b.extra_potentials_array_offsets):
#               start += 1
#               end += 1

#               coeff = e.poten_dx(img.pos)
#               rows[0,start:end] -= coeff
#               rows[0,0] += e.shift * sum(coeff)

#               coeff = e.poten_dy(img.pos)
#               rows[1,start:end] -= coeff
#               rows[1,0] += e.shift * sum(coeff)

            eq(rows[0])
            eq(rows[1])

@object_prior_check(lens_eq)
def check_lens_eq(o, sol):
    b    = o.basis
    offs = 0 #b.array_offset

    #pix_start,    pix_end    = offs+b.offs_pix
    bcoeff_start, bcoeff_end = offs+b.offs_bcoeff
    srcpos_start, srcpos_end = offs+b.offs_srcpos
    sm_err = offs+b.offs_sm_err
    #shear_start,  shear_end  = offs+b.shear_start,  offs+b.shear_end
    #ptmass_start, ptmass_end = offs+b.ptmass_start, offs+b.ptmass_end

    assert np.all(sol[bcoeff_start:bcoeff_end] >= 0)

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

#           sm = o.stellar_mass

#           if o.stellar_mass_error != 0:
#               sm = sm * sol[sm_err]

            #print(i,r0,r1)
            p_dx = poten_dx(positions, b.cell_size, o.basis.maprad)
            p_dy = poten_dy(positions, b.cell_size, o.basis.maprad)

            for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                r0 -= sol[ri] * np.sum(_g * p_dx) #n[_n] * b.gm[_m] * p_dx)
                r1 -= sol[ri] * np.sum(_g * p_dy) #n[_n] * b.gm[_m] * p_dy)
                
            #print('   ',i,r0,r1)

#           r0 -= sum((sm + sol[pix_start:pix_end]) * poten_dx(positions, b.cell_size, o.basis.maprad))
#           r1 -= sum((sm + sol[pix_start:pix_end]) * poten_dy(positions, b.cell_size, o.basis.maprad))

#           for e,[start,end] in zip(o.extra_potentials, b.extra_potentials_array_offsets):
#               r0 -= np.sum((sol[start:end] - e.shift) * e.poten_dx(img.pos))
#               r1 -= np.sum((sol[start:end] - e.shift) * e.poten_dy(img.pos))

#           #print img.pos, r0,r1, sol[srcpos:srcpos+2]

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

    #pix_start,    pix_end    = 1+b.offs_pix
    bcoeff_start, bcoeff_end = 1+b.offs_bcoeff
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
            #row[pix_start:pix_end] -= poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad)
            #row[pix_start:pix_end] += poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad)

            p0 = poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad)
            p1 = poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad)

            for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                row[ri] = np.sum((p0-p1) * _g)
                #row[ri] = np.sum((p0-p1) * b.gn[_n] * b.gm[_m])

#           sm = stellar_mass

#           if o.stellar_mass_error != 0:
#               sm = sm * sol[sm_err] # don't modified the original array with *=

#           row[0] -= sum(sm * poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad))
#           row[0] += sum(sm * poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad))

#           for e,[start,end] in zip(o.extra_potentials, b.extra_potentials_array_offsets):
#               start += 1
#               end += 1

#               p0 = e.poten(img0.pos)
#               p1 = e.poten(img1.pos)
#               coeff  = p1-p0
#               row[start:end] -= coeff
#               row[0] += e.shift * sum(coeff)

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

    #pix_start,    pix_end    = b.offs_pix
    bcoeff_start, bcoeff_end = b.offs_bcoeff
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

#           sm = o.stellar_mass

#           if o.stellar_mass_error != 0:
#               sm *= sol[sm_err]

#           S -= sum((sm + sol[pix_start:pix_end]) * poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad))
#           S += sum((sm + sol[pix_start:pix_end]) * poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad))

            p0 = poten(img0.pos - o.basis.ploc, b.cell_size, o.basis.maprad)
            p1 = poten(img1.pos - o.basis.ploc, b.cell_size, o.basis.maprad)

            for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                S += sol[ri] * np.sum((p0-p1) * _g)
                #S += sol[ri] * np.sum((p0-p1) * b.gn[_n] * b.gm[_m])


#           for e,[start,end] in zip(o.extra_potentials, b.extra_potentials_array_offsets):
#               p0 = e.poten(img0.pos)
#               p1 = e.poten(img1.pos)
#               coeff  = p1-p0
#               S -= sum((sol[start:end] - e.shift) * coeff)

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
            l1 = log10(abs(S2)) if S2 is not None and S2 > 0 else -15

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
        print(msg)

@default_prior
@object_prior
def hubble_constant(o, leq, eq, geq):
    """This requires a particular hubble constant for the object."""

    if env().nu is None:
        Log( "[DISABLED] Hubble Constraint")
        return

    Log( 1*indent + "Hubble Constraint " + str(env().hubble_args) )
    Log( 2*indent + "nu=" + str(env().nu) )

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


@object_prior
def magnification(o, leq, eq, geq):

    Log( indent + "Magnification" )

    MINIMUM, SADDLE, MAXIMUM = 0,1,2

    b = o.basis
    #pix_start,     pix_end = 1+o.basis.offs_pix
    bcoeff_start, bcoeff_end = 1+b.offs_bcoeff

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
                #rows[0, pix_start:pix_end] = -k1*xx + yy
                #rows[1, pix_start:pix_end] = -k2*yy + xx
                #rows[2, pix_start:pix_end] =     xy + xx*eps
                #rows[3, pix_start:pix_end] =    -xy + xx*eps
                #rows[4, pix_start:pix_end] =     xy + yy*eps
                #rows[5, pix_start:pix_end] =    -xy + yy*eps

                for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                    rows[0,ri] = np.sum((-k1*xx + yy         ) * _g) # n[_n] * b.gm[_m])
                    rows[1,ri] = np.sum((-k2*yy + xx         ) * _g) # n[_n] * b.gm[_m])
                    rows[2,ri] = np.sum((         xy + xx*eps) * _g) # n[_n] * b.gm[_m])
                    rows[3,ri] = np.sum((        -xy + xx*eps) * _g) # n[_n] * b.gm[_m])
                    rows[4,ri] = np.sum((         xy + yy*eps) * _g) # n[_n] * b.gm[_m])
                    rows[5,ri] = np.sum((        -xy + yy*eps) * _g) # n[_n] * b.gm[_m])

            if parity == SADDLE:
                #rows[0, pix_start:pix_end] = -k1*xx - yy
                #rows[1, pix_start:pix_end] =  k2*yy + xx
                #rows[2, pix_start:pix_end] =     xy + xx*eps
                #rows[3, pix_start:pix_end] =    -xy + xx*eps
                #rows[4, pix_start:pix_end] =     xy - yy*eps
                #rows[5, pix_start:pix_end] =    -xy - yy*eps

                for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                    rows[0,ri] = np.sum((-k1*xx - yy         ) * _g) # n[_n] * b.gm[_m])
                    rows[1,ri] = np.sum(( k2*yy + xx         ) * _g) # n[_n] * b.gm[_m])
                    rows[2,ri] = np.sum((         xy + xx*eps) * _g) # n[_n] * b.gm[_m])
                    rows[3,ri] = np.sum((        -xy + xx*eps) * _g) # n[_n] * b.gm[_m])
                    rows[4,ri] = np.sum((         xy - yy*eps) * _g) # n[_n] * b.gm[_m])
                    rows[5,ri] = np.sum((        -xy - yy*eps) * _g) # n[_n] * b.gm[_m])

            if parity == MAXIMUM:
                #rows[0, pix_start:pix_end] =  k1*xx - yy
                #rows[1, pix_start:pix_end] =  k2*yy - xx
                #rows[2, pix_start:pix_end] =     xy - xx*eps
                #rows[3, pix_start:pix_end] =    -xy - xx*eps
                #rows[4, pix_start:pix_end] =     xy - yy*eps
                #rows[5, pix_start:pix_end] =    -xy - yy*eps

                for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                    rows[0,ri] = np.sum(( k1*xx - yy         ) * _g) # n[_n] * b.gm[_m])
                    rows[1,ri] = np.sum(( k2*yy - xx         ) * _g) # n[_n] * b.gm[_m])
                    rows[2,ri] = np.sum((         xy - xx*eps) * _g) # n[_n] * b.gm[_m])
                    rows[3,ri] = np.sum((        -xy - xx*eps) * _g) # n[_n] * b.gm[_m])
                    rows[4,ri] = np.sum((         xy - yy*eps) * _g) # n[_n] * b.gm[_m])
                    rows[5,ri] = np.sum((        -xy - yy*eps) * _g) # n[_n] * b.gm[_m])

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
        for r in range(o.basis.inner_image_ring, o.basis.outer_image_ring):
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

        for e,[start, end] in zip(o.extra_potentials, o.basis.extra_potentials_array_offsets):
            if not isinstance(e, Shear): continue
            for s in range(1+start, 1+end):
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
        Log( "%10s External Shear" % on)
    else:
        on = ''
        Log( "%10s External Shear (%.4g)" % (on,v))


@default_prior
@object_prior
def external_mass(o, leq, eq, geq):
    on = None
    v = o.prior_options.get('external mass', None)
    if v is None:
        return

    #assert v < shear.shift

    for e,[start, end] in zip(o.extra_potentials, o.basis.extra_potentials_array_offsets):
        if not (hasattr(e.__class__, '__bases__') and ExternalMass in e.__class__.__bases__): continue
        min,max = o.prior_options['external mass'][e]
        #print(indent + 'External mass %s  %g-%g' % (e.name, min, max))
        print(indent + 'External mass %s' % (e.name))
        for s in range(1+start, 1+end):
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

    b = o.basis

    bcoeff_start, bcoeff_end = 1+b.offs_bcoeff
    #pix_start, pix_end = 1+o.basis.offs_pix

    nrings = len(o.basis.rings)

    c = 0

    for l,[r0,r1] in enumerate(zip(o.basis.rings[:-1], o.basis.rings[1:])):

        if 1:
        #if not o.basis.hires_levels \
        #or (o.basis.hires_levels and l < (o.basis.hiresR*o.basis.hires_levels - o.basis.hires_levels//2)):

            R0 = (o.basis.rs[l]   + o.basis.radial_cell_size[l]/2)
            R1 = (o.basis.rs[l+1] + o.basis.radial_cell_size[l+1]/2)

            #print R0,R1, o.basis.radial_cell_size[l], o.basis.radial_cell_size[l+1]

            w0 = o.basis.cell_size[r0]**2 / np.sum(o.basis.cell_size[r0]**2)
            w1 = o.basis.cell_size[r1]**2 / np.sum(o.basis.cell_size[r1]**2)

            row = new_row(o)
            #row[pix_start+r0] =  w0 #* R0**minsteep
            #row[pix_start+r1] = -w1 #* R1**minsteep

            #row[pix_start+r0] =  w0 * R1**(-minsteep)
            #row[pix_start+r1] = -w1 * R0**(-minsteep)

            for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                row[ri]  = np.sum((w0 * R0**(minsteep)) * _g[r0]) #n[_n][r0] * b.gm[_m][r0])
                row[ri] -= np.sum((w1 * R1**(minsteep)) * _g[r1]) #n[_n][r1] * b.gm[_m][r1])

            if minsteep == maxsteep:
                eq(row)
            else:
                geq(row)

                if l > 0 and maxsteep is not None:
                    row = new_row(o)
                    #row[pix_start+r0] =  w0 * R1**(-maxsteep)
                    #row[pix_start+r1] = -w1 * R0**(-maxsteep)
                    for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                        row[ri]  = np.sum((w1 * R1**(maxsteep)) * _g[r1]) #n[_n][r1] * b.gm[_m][r1])
                        row[ri] -= np.sum((w0 * R0**(maxsteep)) * _g[r0]) #n[_n][r0] * b.gm[_m][r0])
                    geq(row)
                    c += 1
            c += 1

    DLog( 1, 2*indent + "# eqs = %i" % c )

@default_prior
@object_prior
def profile_steepnessX(o, leq, eq, geq):
    steep = o.prior_options.get('steepness', None)

    if steep is None:
        Log( "[DISABLED] Profile Steepness XX" )
        return

    Log( indent + "Profile Steepness  XX %s" % steep )

    minsteep, maxsteep = steep

    assert maxsteep is None or maxsteep >= minsteep

    b = o.basis

    bcoeff_start, bcoeff_end = 1+b.offs_bcoeff
    #pix_start, pix_end = 1+o.basis.offs_pix

    nrings = len(o.basis.rings)

    c = 0

    l=0
    R0 = (o.basis.rs[l]   + o.basis.radial_cell_size[l]/2)
    r0 = o.basis.rings[0]
    w0 = o.basis.cell_size[r0]**2 / np.sum(o.basis.cell_size[r0]**2)

    for l,[_,r1] in enumerate(zip(o.basis.rings[:-1], o.basis.rings[1:])):

        if 1:
        #if not o.basis.hires_levels \
        #or (o.basis.hires_levels and l < (o.basis.hiresR*o.basis.hires_levels - o.basis.hires_levels//2)):

            R1 = (o.basis.rs[l+1] + o.basis.radial_cell_size[l+1]/2)

            #print R0,R1, o.basis.radial_cell_size[l], o.basis.radial_cell_size[l+1]

            w1 = o.basis.cell_size[r1]**2 / np.sum(o.basis.cell_size[r1]**2)

            row = new_row(o)
            #row[pix_start+r0] =  w0 #* R0**minsteep
            #row[pix_start+r1] = -w1 #* R1**minsteep

            #row[pix_start+r0] =  w0 * R1**(-minsteep)
            #row[pix_start+r1] = -w1 * R0**(-minsteep)

            for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                #row[ri]  = np.sum((w0 * minsteep) * _g[r0]) #n[_n][r0] * b.gm[_m][r0])
                #row[ri] -= np.sum((w1 *        1) * _g[r1]) #n[_n][r1] * b.gm[_m][r1])

                row[ri]  = np.sum(w0 * _g[r0]) #n[_n][r0] * b.gm[_m][r0])
                row[ri] -= np.sum((w1 * R1**(minsteep)) * _g[r1]) #n[_n][r1] * b.gm[_m][r1])

            if minsteep == maxsteep:
                eq(row)
            else:
                geq(row)

                if l > 0 and maxsteep is not None:
                    row = new_row(o)
                    #row[pix_start+r0] =  w0 * R1**(-maxsteep)
                    #row[pix_start+r1] = -w1 * R0**(-maxsteep)
                    for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                        row[ri]  = np.sum((w0 * R0**(-maxsteep)) * _g[r0]) #n[_n][r0] * b.gm[_m][r0])
                        row[ri] -= np.sum((w1 * R1**(-maxsteep)) * _g[r1]) #n[_n][r1] * b.gm[_m][r1])
                    geq(row)
                    c += 1
            c += 1

    DLog( 1, 2*indent + "# eqs = %i" % c )

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
    for l in range(1,nrings-1):
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

    b = o.basis

    bcoeff_start, bcoeff_end = 1+b.offs_bcoeff
    #pix_start, pix_end = 1+o.basis.offs_pix

    phi = radians(90-theta)
    cs,sn = cos(phi), sin(phi)
    c = 0

    rs = [ abs(img.pos) for src in o.sources for img in src.images if img.parity_name != 'max']
    rmin, rmax = amin(rs), amax(rs)

    for i,[ri,r] in enumerate(zip(o.basis.int_ploc, o.basis.ploc)):
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

                #row[pix_start + w] =  wght(w) * x
                #row[pix_start + e] = -wght(e) * x
                #row[pix_start + s] =  wght(s) * y
                #row[pix_start + n] = -wght(n) * y

                for ri, [_i,_n,_m,_g] in enumerate(b.bern_nm, bcoeff_start):
                    row[ri]  = np.sum(wght(w) * x * _g[w]) #n[_n][w] * b.gm[_m][w])
                    row[ri] -= np.sum(wght(e) * x * _g[e]) #n[_n][e] * b.gm[_m][e])
                    row[ri] += np.sum(wght(s) * y * _g[s]) #n[_n][s] * b.gm[_m][s])
                    row[ri] -= np.sum(wght(n) * y * _g[n]) #n[_n][n] * b.gm[_m][n])

                geq(row)
                c += 1

    DLog( 1, 2*indent + "gradient eqs = %i" % c )
    DLog( 1, 2*indent + "sn=%g" % sn )


#@default_prior
@object_prior
def BernGradient(o, leq, eq, geq):

    opts = o.prior_options.get('BernGradient', None)

    if not opts:
        opts = {}
        #Log( 'J2Gradient NOT ACTIVE' )
        #return

    #Lmin = np.sqrt(2)*o.basis.top_level_cell_size
    #Lmin = 1.1*o.basis.top_level_cell_size

    theta = opts.get('theta', 45)
    size  = opts.get('size',  None)

    #XXX assert (L >= Lmin), 'size=%f < %f is too small' % (L, Lmin)

    Log( indent + "BernGradient (theta=%.2f  size=%s)" % (theta, size) )

    b = o.basis

    bcoeff_start, bcoeff_end = 1+b.offs_bcoeff
    #pix_start, pix_end = 1+o.basis.offs_pix

    phi = radians(90-theta)
    cs,sn = cos(phi), sin(phi)
    c = 0

    rs = [ abs(img.pos) for src in o.sources for img in src.images if img.parity_name != 'max']
    rmin, rmax = amin(rs), amax(rs)

    for i,[ri,r] in enumerate(zip(o.basis.int_ploc, o.basis.ploc)):
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

                #row[pix_start + w] =  wght(w) * x
                #row[pix_start + e] = -wght(e) * x
                #row[pix_start + s] =  wght(s) * y
                #row[pix_start + n] = -wght(n) * y

                ri = bcoeff_start
                for _n in range(b.poly_order+1):
                    for _m in range(b.poly_order+1):

                        row[ri]  = np.sum(( wght(w) * x) * b.gs[_i][w]) #n[_n][w] * b.gm[_m][w])
                        row[ri] += np.sum((-wght(e) * x) * b.gs[_i][e]) #n[_n][e] * b.gm[_m][e])
                        row[ri] += np.sum(( wght(s) * y) * b.gs[_i][s]) #n[_n][s] * b.gm[_m][s])
                        row[ri] += np.sum((-wght(n) * y) * b.gs[_i][n]) #n[_n][n] * b.gm[_m][n])

                        ri += 1
                geq(row)
                c += 1

    DLog( 1, 2*indent + "gradient eqs = %i" % c )
    DLog( 1, 2*indent + "sn=%g" % sn )

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
    for i,[ri,r] in enumerate(zip(o.basis.int_ploc, o.basis.ploc)):
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
    for o1,o2 in zip(objs[:-1], objs[1:]):
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

#@default_prior
@object_prior
def source_position_quadrant(o, leq,eq,geq):
    b = o.basis
    srcpos_start, srcpos_end = 1+b.offs_srcpos
    wedge = o.prior_options.get('source_position_quadrant_angle', 90.)

    for i,src in enumerate(o.sources):
        if src.pos is not None: continue

        x = srcpos_start + 2*i + 0
        y = srcpos_start + 2*i + 1

        img = src.images[0]

        tx = (360+90 - (img.angle+wedge/2.)) * (np.pi/180.)
        ty = (360    - (img.angle-wedge/2.)) * (np.pi/180.)

        cx,sx = np.cos(tx), np.sin(tx)
        cy,sy = np.cos(ty), np.sin(ty)
        Kx = -b.map_shift*(cx-sx) * src.zcap
        Ky = -b.map_shift*(sy+cy) * src.zcap

        row = new_row(o,2)
        row[0, [0,x,y]] = Kx, cx, -sx; geq(row[0])
        row[1, [0,x,y]] = Ky, sy,  cy; geq(row[1])

@object_prior_check(source_position_quadrant)
def check_source_position_quadrant(o, sol):
    b    = o.basis
    offs = 0 #b.array_offset
    wedge = o.prior_options.get('source_position_quadrant_angle', 90)

    srcpos_start, srcpos_end = offs+b.offs_srcpos

    for i,src in enumerate(o.sources):
        if src.pos is not None: continue

        srcpos = srcpos_start + 2*i

        img = src.images[0]

        x = sol[srcpos+0]/src.zcap - b.map_shift
        y = sol[srcpos+1]/src.zcap - b.map_shift

        tx = (360+90 - (img.angle+wedge/2.)) * (np.pi/180.)
        ty = (360    - (img.angle-wedge/2.)) * (np.pi/180.)
        cx,sx = np.cos(tx), np.sin(tx)
        cy,sy = np.cos(ty), np.sin(ty)

        nx = cx*x - sx*y
        ny = sy*x + cy*y

        if nx < 0 or ny < 0:
            Log( "Source position contrained by quadrant is outside quadrant! Source is at (%f,%f). In rotated frame this is (%f,%f), which is outside of upper left quadrant." % (x,y,nx,ny) )

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

        print(x,y)

        print(tol)
        if tol != 0:
            lb = src.zcap * (b.map_shift + sx - tol)
            ub = src.zcap * (b.map_shift + sx + tol)
            row = new_row(o,2)
            row[0, [0,x]] = lb, -1; leq(row[0])
            row[1, [0,x]] = ub, -1; geq(row[1])

            print(lb, ub)

            lb = src.zcap * (b.map_shift + sy - tol)
            ub = src.zcap * (b.map_shift + sy + tol)
            row = new_row(o,2)
            row[0, [0,y]] = lb, -1; leq(row[0])
            row[1, [0,y]] = ub, -1; geq(row[1])

            print(lb, ub)
        else:
            row = new_row(o,2)
            row[0, [0,x]] = sx, -1; eq(row[0])
            row[1, [0,y]] = sy, -1; eq(row[1])


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




