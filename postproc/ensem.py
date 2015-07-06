mname = 'ASW0007k4r/012771'
mname = 'ASW0000h2m/007022'
mname = 'ASW0000h2m/gribbles'

state = loadstate(mname+'.state')

grids = []
for m in state.models:
    obj,data = m['obj,data'][0]
    g = obj.basis._to_grid(data['kappa DM'])
    grids.append(g)

import pickle
fil = open(mname+'.pkl','w')
pickle.dump(grids,fil)



