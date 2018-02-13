#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript that loads a glass state file and dumps all relevant information
into a pure python dictorionary with numpy files and saves
it as pickle file.

Created on Wed Jul  8 12:57:52 2015
@author: psaha, rafik
"""

import sys
import os.path
import pickle


try: # read first parameter from command line (remember: 0 is the script file name)
    mname = sys.argv[1]

except IndexError:
    print "Usage:\n    python %s glassfile.gls" % sys.argv[0]
    sys.exit("No glass file specified")
    

#check if the file really exists
if not os.path.isfile(mname):
    print "no valid file! (%s)" % mname
    sys.exit('no valid file')


state = loadstate(mname)

grids = []
for m in state.models:
    obj,data = m['obj,data'][0]
    g = obj.basis._to_grid(data['kappa DM'])
    grids.append(g)
    
imgradii = [abs(img.pos) for img in state.objects[0].sources[0].images]

# create dict to be pickled afterwards
d = {
    'grids': grids,
    'maprad': obj.basis.maprad,
    'pixrad': obj.basis.pixrad,
    'maximgpos': max(imgradii),
}
    
# clean up filename
pklfn = '.'.join(mname.split('.')[:-1]) + '.pkl'

with open(pklfn,'w') as fil:
    pickle.dump(d,fil)








