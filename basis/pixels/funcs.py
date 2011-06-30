from __future__ import division
from scales import convert
from numpy import cumsum, mean, average, array, where, pi
from environment import DArray

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

    #print w

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

#def estimated_profile_slope(m, vdisp_true, beta):
#
#    a = r_half * (2**(1./(3-beta)) - 1)
#
#    def f(gamma):
#        return (a/2)**(2-gamma) * Gamma(gamma)**2 * Gamma(5 - gamma - beta) / Gamma(3 - beta)

def default_post_process(m):
    obj,ps = m
    b = obj.basis

    rscale = convert('arcsec to kpc', 1, obj.dL, ps['nu'])

    dscale1 = convert('kappa to Msun/arcsec^2', 1, obj.dL, ps['nu'])
    dscale2 = convert('kappa to Msun/kpc^2',    1, obj.dL, ps['nu'])

    #ps['R']     = b.rs + b.radial_cell_size / 2
    #ps['R_kpc'] = ps['R'] * rscale

    ps['R'] = {}
    ps['R']['arcsec'] = b.rs + b.radial_cell_size / 2
    ps['R']['kpc']    = ps['R']['arcsec'] * rscale

    ps['M(<R)']     = cumsum([    sum(ps['kappa'][r]*b.cell_size[r]**2)*dscale1 for r in b.rings])
    ps['Sigma(R)']  =  array([average(ps['kappa'][r]                  )*dscale2 for r in b.rings])
    ps['kappa(R)']  =  array([average(ps['kappa'][r]                  )         for r in b.rings])
    #ps['kappa(<R)'] = (lambda a: cumsum(a[:,0]) / cumsum(a[:,1]))([ [sum(ps['kappa'][r]), len(r)] for r in b.rings ])
    ps['kappa(<R)'] = cumsum([sum(ps['kappa'][r]) for r in b.rings]) / cumsum([len(r) for r in b.rings])

    #ps['kappa(<R)'] = cumsum([sum(ps['kappa'][r]) for r in b.rings]) / cumsum([len(r) for r in b.rings])

    ps['Rlens'] = {}
    ps['Rlens']['arcsec'] = [ estimated_Rlens(obj, ps,i)[0] for i,src in enumerate(obj.sources) ]
    ps['Rlens']['kpc']    = [ r * rscale              for r in ps['Rlens']['arcsec'] ]

    ps['Ktot'] = sum(ps['kappa'])
    ps['R(1/2 K)'] = {}
    #ps['R(1/2 K)']['arcsec'] = ps['R']['arcsec'][(ps['kappa(<R)'] - 0.5*ps['Ktot']) >= 0.0][0]
    #ps['R(1/2 K)']['kpc']    = ps['R(1/2 K)']['arcsec'] * rscale


    # convert to DArray

    ps['R']['arcsec'] = DArray(ps['R']['arcsec'], ul=['arcsec', r'$R$ $(\mathrm{arcsec})$'])
    ps['R']['kpc']    = DArray(ps['R']['kpc'],    ul=['kpc',    r'$R$ $(\mathrm{kpc})$'])

    ps['kappa(<R)'] = DArray(ps['kappa(<R)'], ul=['kappa',      r'$\kappa(<R)$'])
    ps['M(<R)']     = DArray(ps['M(<R)'],     ul=['Msun',       r'$M(<R)$ $(M_\odot)$'])
    ps['Sigma(R)']  = DArray(ps['Sigma(R)'],  ul=['Msun/kpc^2', r'$\Sigma$ $(M_\odot/\mathrm{kpc}^2)$'])
    ps['kappa(R)']  = DArray(ps['kappa(R)'],  ul=['kappa',      r'$\langle\kappa(R)\rangle$'])

    ps['Rlens']['arcsec'] = [ DArray(v, ul=['arcsec', r'$R_e$ $(\mathrm{arcsec})$']) for v in ps['Rlens']['arcsec'] ]
    ps['Rlens']['kpc']    = [ DArray(v, ul=['kpc',    r'$R_e$ $(\mathrm{kpc})$'   ]) for v in ps['Rlens']['kpc']    ]

    ps['R(1/2 K)']['arcsec'] = [ DArray(v, ul=['arcsec', r'$R_e$ $(\mathrm{arcsec})$']) for v in ps['Rlens']['arcsec'] ]
    ps['R(1/2 K)']['kpc']    = [ DArray(v, ul=['kpc',    r'$R_e$ $(\mathrm{kpc})$'   ]) for v in ps['Rlens']['kpc']    ]

