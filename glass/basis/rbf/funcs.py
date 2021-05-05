
from glass.scales import convert
import numpy as np
from numpy import cumsum, mean, array, where, pi, dot, abs
from glass.environment import DArray
from . potential import poten, poten_dx, poten_dy
import glass.shear as shear


def estimated_Rlens(obj, ps, src_index):

    #---------------------------------------------------------------------
    # Estimate an Einstein radius. 
    # Take the inertia tensor of the pixels above kappa_crit and use the
    # eigenvalues to scale the most distance pixel position to the major
    # and minor axes. Re is then defined here as the mean of the two.
    #
    # TODO: On convergence. Since the centers of each pixel are used, as
    # the resolution increases, Re will tend to move outward. A better
    # solution would be to use the maximum extent of each pixel.
    #---------------------------------------------------------------------

    #avgkappa = ps['kappa(<R)'] / cumsum(map(len,obj.basis.rings))
    #avgkappa = ps['kappa(<R)'] # / cumsum(map(len,obj.basis.rings))

    #print map(len,obj.basis.rings)

    #print '^' * 10
    #print kappa
    #print '^' * 10
    w = ps['kappa(<R)'] >= 1

    #print ps['kappa(<R)']
    #print ps['R']['kpc']
    #print w
    #assert 0

    if not w.any(): return 0,0,0,0

#   print '@'*80
#   print ps['R']['arcsec'][w]
#   print '@'*80

    w = where(w)[0][-1]

    #print w

    r = ps['R']['arcsec'][w]
    #r = mean(abs(obj.basis.ploc[obj.basis.rings[w]]))

    #print r

    Vl = abs(r)
    Vs = abs(r)
    if Vl < Vs: 
        Vl,Vs = Vs,Vl
        D1,D2 = D1,D2

    return mean([Vl,Vs]), Vl, Vs, 0

def arrival_time(m):

    obj,ps = m

    at = []
    for s,src in zip(obj.sources, ps['src']):
        taus = []
        for img in s.images:
            theta = img.pos

            tau  = abs(theta-src)**2 / 2
            tau *= s.zcap
            tau -= dot(ps['kappa'], poten(theta - obj.basis.ploc, obj.basis.cell_size, obj.basis.maprad))
            tau -= np.sum( [ ps[e.name] * e.poten(theta).T for e in obj.extra_potentials ] )

            taus.append(tau)
        at.append(taus)

    return at

#def estimated_profile_slope(m, vdisp_true, beta):
#
#    a = r_half * (2**(1./(3-beta)) - 1)
#
#    def f(gamma):
#        return (a/2)**(2-gamma) * Gamma(gamma)**2 * Gamma(5 - gamma - beta) / Gamma(3 - beta)


def default_post_process(m):
    return 

    obj,ps = m
    b = obj.basis

    ps['H0']     = convert('nu to H0 in km/s/Mpc', ps['nu'])
    ps['1/H0']   = convert('nu to H0^-1 in Gyr',   ps['nu'])

    rscale = convert('arcsec to kpc', 1, obj.dL, ps['nu'])

    #print ps['nu'], convert('nu to H0^-1 in Gyr', ps['nu'])

    dscale1 = convert('kappa to Msun/arcsec^2', 1, obj.dL, ps['nu'])
    dscale2 = convert('kappa to Msun/kpc^2',    1, obj.dL, ps['nu'])

    #ps['R']     = b.rs + b.radial_cell_size / 2
    #ps['R_kpc'] = ps['R'] * rscale

    ps['R'] = DArray(b.rs + b.radial_cell_size / 2,
                     r'$R$', {'arcsec': [1, r'$\mathrm{arcsec}$'],
                              'kpc':    [rscale, r'$\mathrm{kpc}$']})

    def mean_kappa(x):
        return sum(ps['kappa'][x] * b.cell_size[x]**2) /  sum(b.cell_size[x]**2)

    def mean_kappa2(x, a):
        return sum(a[x] * b.cell_size[x]**2) /  sum(b.cell_size[x]**2)

    for e in obj.extra_potentials:
        if hasattr(e, 'kappa') and hasattr(obj, e.name):
            ps['kappa %s' % e.name] = getattr(obj, e.name) * ps[e.name]
            ps['kappa'] += ps['kappa %s' % e.name]
            ps['kappa(R) %s' % e.name] = DArray([mean_kappa2(r, ps['kappa %s' % e.name]) for r in b.rings],
                                    r'$\langle\kappa(R)\rangle$', {'$\kappa$': [1, None]})

    
    
    #print ps['R']['arcsec']
    #print ps['R']['kpc']
    #assert 0
    

    ps['M(<R)'] = DArray(cumsum([sum(ps['kappa'][r]*b.cell_size[r]**2)*dscale1 for r in b.rings]),
                         r'$M(<R)$', {'Msun': [1, r'$M_\odot$']})

    ps['Sigma(R)'] = DArray([mean_kappa(r)*dscale2 for r in b.rings],
                            r'$\Sigma$', {'Msun/kpc^2': [1, r'$(M_\odot/\mathrm{kpc}^2)$']})

    ps['kappa(R)'] = DArray([mean_kappa(r) for r in b.rings],
                            r'$\langle\kappa(R)\rangle$', {'$\kappa$': [1, None]})

    ps['kappa(<R)'] = DArray(cumsum([sum(ps['kappa'][r]*b.cell_size[r]**2) for r in b.rings]) / cumsum([sum(b.cell_size[r]**2) for r in b.rings]),
                             r'$\kappa(<R)$', {'kappa': [1, None]})


    ps['Rlens'] = DArray([estimated_Rlens(obj, ps,i)[0] for i,src in enumerate(obj.sources)],
                          r'$R_e$', {'arcsec': [1, r'$\mathrm{arcsec}$'],
                                     'kpc':    [rscale, '$\mathrm{kpc}$']})

    ps['Ktot'] = sum(ps['kappa'])
    #ps['R(1/2 K)'] = {}
    #ps['R(1/2 K)']['arcsec'] = ps['R']['arcsec'][(ps['kappa(<R)'] - 0.5*ps['Ktot']) >= 0.0][0]
    #ps['R(1/2 K)']['kpc']    = ps['R(1/2 K)']['arcsec'] * rscale

    ps['arrival times'] = arrival_time(m)

    ps['time delays'] = []
    for src_index,src in enumerate(obj.sources):
        d = []
        if ps['arrival times'][src_index]:
            t0 = ps['arrival times'][src_index][0]
            for i,t in enumerate(ps['arrival times'][src_index][1:]):
                d.append( float('%0.6f'%convert('arcsec^2 to days', t-t0, obj.dL, obj.z, ps['nu'])) )
                t0 = t
        ps['time delays'].append(d)


    # convert to DArray

#   ps['R']['arcsec'] = DArray(ps['R']['arcsec'], ul=['arcsec', r'$R$ $(\mathrm{arcsec})$'])
#   ps['R']['kpc']    = DArray(ps['R']['kpc'],    ul=['kpc',    r'$R$ $(\mathrm{kpc})$'])

#   ps['kappa(<R)'] = DArray(ps['kappa(<R)'], ul=['kappa',      r'$\kappa(<R)$'])
#   ps['M(<R)']     = DArray(ps['M(<R)'],     ul=['Msun',       r'$M(<R)$ $(M_\odot)$'])
#   ps['Sigma(R)']  = DArray(ps['Sigma(R)'],  ul=['Msun/kpc^2', r'$\Sigma$ $(M_\odot/\mathrm{kpc}^2)$'])
#   ps['kappa(R)']  = DArray(ps['kappa(R)'],  ul=['kappa',      r'$\langle\kappa(R)\rangle$'])

#   ps['Rlens']['arcsec'] = [ DArray(v, ul=['arcsec', r'$R_e$ $(\mathrm{arcsec})$']) for v in ps['Rlens']['arcsec'] ]
#   ps['Rlens']['kpc']    = [ DArray(v, ul=['kpc',    r'$R_e$ $(\mathrm{kpc})$'   ]) for v in ps['Rlens']['kpc']    ]

#   ps['R(1/2 K)']['arcsec'] = [ DArray(v, ul=['arcsec', r'$R_e$ $(\mathrm{arcsec})$']) for v in ps['Rlens']['arcsec'] ]
#   ps['R(1/2 K)']['kpc']    = [ DArray(v, ul=['kpc',    r'$R_e$ $(\mathrm{kpc})$'   ]) for v in ps['Rlens']['kpc']    ]


#   ps['Rlens'] = [ DArray(v, r'$R_e$', ul=[{'arcsec': [1, r'$\mathrm{arcsec}$']}]) for v in ps['Rlens'] ]



