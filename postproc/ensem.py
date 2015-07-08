mname = 'ASW0007k4r/012771'
mname = 'ASW0000h2m/007022'
mname = 'ASW0000h2m/gribbles'
mname = 'WM4H5RZXQZ'
#mname = 'IHRULOMX6D'

state = loadstate(mname+'.state')

grids = []
for m in state.models:
    obj,data = m['obj,data'][0]
    g = obj.basis._to_grid(data['kappa DM'])
    grids.append(g)
    
imgradii = [abs(img.pos) for img in state.objects[0].sources[0].images]


d = {
    'grids': grids,
    'maprad': obj.basis.maprad,
    'pixrad': obj.basis.pixrad,
    'maximgpos': max(imgradii),
}
    
import pickle

with open(mname+'.pkl','w') as fil:
    pickle.dump(d,fil)

