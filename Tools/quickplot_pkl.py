import pickle
import numpy as np
import matplotlib.pyplot as pl

mname = 'ASW0007k4r/012771'
mname = 'ASW0000h2m/007022'
mname = 'ASW0000h2m/gribbles'

fil = open(mname+'.pkl')
ensem = pickle.load(fil)

N = ensem[0].shape[0]
R = (N-1)/2
x = np.linspace(-R,R,N)
X,Y = np.meshgrid(x,x)
F = ensem[0]

print F

lev = np.linspace(0,10,21)
pl.contour(X,Y,F, levels=[0,1,2,3,4,5,6])
pl.axes().set_aspect('equal')
pl.show()
