from __future__ import division
import sys, getopt
from numpy import array, average
from numpy import loadtxt

from environment import env
from plots import mass_plot, potential_plot, sigma_plot, H0_plot, encmass_plot, arrival_plot, src_plot, img_plot
import pylab
from pylab import figure, show, clf, ioff, draw, ion, subplot, cla, matshow


def help():
    print >>sys.stderr, "Usage: glass.py <input>"
    sys.exit(2)

# Although this is technically a command, we need it here so that it
# can see 'init_model_generator' which will be defined by the executed
# input file.
def model(nmodels):
    for o in env.objects:
        print o.name
        o.init()

    init_model_generator()

    if 0:
        model = empty(2002)
        f = open('sols', 'r')
        for i in xrange(2002):
            model[i] = float(f.readline())

    #print arriv_grid(model, env.objects[0])
    #print mass_grid(model, env.objects[0])

    env.models = []
    H0s = []
    mass = None

    for i,m in enumerate(generate_models(env.objects, nmodels)): 
        if i == 0: continue
        env.models.append(m)

    env.accepted_models = env.models

def plot():

    if env.accepted_models:

        ensemble_avg = average([ m['sol'] for m in env.accepted_models ], axis=0)

        print ensemble_avg

        obj0_ensemble_ps = packaged_solution(env.objects[0], ensemble_avg)

        #mass_plot([env.objects[0], {'mass': mass/nmodels}])
        matshow(env.objects[0].basis._lnr())
        figure()
        subplot(331)
        mass_plot([env.objects[0], obj0_ensemble_ps])
        src_plot([env.objects[0], obj0_ensemble_ps])
        img_plot([env.objects[0], obj0_ensemble_ps])
        #figure()
        subplot(335)
        potential_plot([env.objects[0], obj0_ensemble_ps], 0)
        src_plot([env.objects[0], obj0_ensemble_ps])
        img_plot([env.objects[0], obj0_ensemble_ps])
        #figure()
        for i,sys in enumerate(env.objects[0].systems):
            subplot(3,3,7+i, aspect='equal')
            arrival_plot([env.objects[0], obj0_ensemble_ps], i)
            src_plot([env.objects[0], obj0_ensemble_ps])
            img_plot([env.objects[0], obj0_ensemble_ps])

    figure()
    sigma_plot(env.models)

    files = []

    if not files:
        dir  = '/smaug/data/theorie/justin/Backup/Mylaptop/Scratch/Lensing/Cuspcore/CMerger1'
        files.append(dir + '/cmerger_1_sdenx.txt')

    for f in files:
        data = loadtxt(f,
                       dtype = {'names': ('R', 'sigp', 'err'),
                                'formats': ('f8', 'f8', 'f8')})

    pylab.plot(data['R'], data['sigp'], 'g-')


    figure()
    encmass_plot(env.models)
    figure()
    H0_plot(env.models)
    show()
    #potential_plot([env.objects[0], {'mass':None}])
    #mass_plot(m[0])

    #arrival_plot(m[0][0], m[0][1])


    if 0:

        # This is set in apply_priors
        mg = env.model_gen

        mg.start()
        f = open('sols', 'w')
        for sol in mg.next(500):
            pass
            #for v in sol: print >>f, v
            #print >>f
        f.close()

if __name__ == "__main__":

    if len(sys.argv) < 2: help()

    optlist, list = getopt.getopt(sys.argv[1:], 't:h')
    print optlist, list
    for opt in optlist:
        if   opt[0] == '-h':
            help()
        elif opt[0] == '-t':
            ncpus = int(opt[1])
            assert ncpus > 0
            env.ncpus = ncpus

    from glcmds import *
    from filters import *

    execfile(list[0])

    #plot()

