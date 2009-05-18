from __future__ import division
import sys
from environment import env
from plots import mass_plot, potential_plot, encmass_plot

from pylab import figure, show

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
    for i,m in enumerate(generate_models(env.objects, nmodels)): 
        encmass_plot(m[1][0])
        #mass_plot(m[1][0])
        #mass_plot([env.objects[0], {'mass':None}])
        show()
        env.models.append(m)
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
    from commands import *
    execfile(sys.argv[1])

