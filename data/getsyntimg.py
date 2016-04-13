# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:39:04 2014

@author: rafik
"""

from __future__ import division

import numpy as np
#import scipy as sp
#from scipy import misc
import matplotlib.pyplot as plt

import imgproc as ip


#from pprint import pprint


glass_basis('glass.basis.pixels', solver=None)
exclude_all_priors()

state = loadstate('012402.state')

state.make_ensemble_average()



model = state.ensemble_average

obj_index = 0
obj, ps = model['obj,data'][obj_index]

#
#pprint(model)

#env().make_ensemble_average()
#obj,ps = env()#.models


src_index = 0
src = ps['src'][src_index]
zcap = obj.sources[src_index].zcap


def delta_beta(theta):
    return src - theta + obj.basis.deflect(theta, ps) / zcap




pixel = []

for x, xrow in enumerate(ip.image):
    for y, rgb in enumerate(xrow):
        if sum(rgb) != 0:           #TODO make this depend on a binary mask
            pixel.append((x,y))

#pix_arr = np.array(pixel)

theta = map(ip.px2arcs, pixel)
d_source = np.array(map(delta_beta, theta))
#print pix_arr

#if True:
#    plt.scatter(pix_arr[:,1], -1*pix_arr[:,0])
#    plt.scatter(np.array(theta).real, np.array(theta).imag)
#    plt.scatter(np.array(d_source).real, np.array(d_source).imag)
#    plt.show()


# range of grid on sourceplane
r = max(np.max(np.abs(d_source.real)), np.max(np.abs(d_source.imag)))

# number of gridpoints on sourceplane (2M+1)
M = 20

X = np.floor(M*(1+d_source.real/r)+.5)
Y = np.floor(M*(1+d_source.imag/r)+.5)

#plt.scatter(X,Y)
#plt.show()

pxllist = np.zeros((2*M+1,2*M+1), dtype=list)


for (x,y), value in np.ndenumerate(pxllist):
    pxllist[x,y] = []
    
for i, pnt in enumerate(d_source):
    x,y = (X[i], Y[i])
#    if not pxllist[x,y]:
#        pxllist[x,y] = []        
    pxllist[x,y].append(i)



#for (x,y), value in np.ndenumerate(pxllist):
#    print (x,y), value


#nimage = np.clip(ip.image+255,0,255)
nimage = 0*ip.image

for (x,y), lst in np.ndenumerate(pxllist):
    n = len(lst)    
    if n>0:
        summ = np.array([0,0,0], dtype=np.int)
        for i in lst:
            ix,iy = pixel[i]
            summ += ip.image[ix,iy]
        summ = np.uint8(summ / n)
        #print summ.dtype
        
        for i in lst:
            ix,iy = pixel[i]
            #print summ
#            nimage[ix,iy] = ip.image[ix,iy]
            nimage[ix,iy] = summ

#            if np.sum(summ)==0:
#                print summ, x, y, ix, iy

#            if x==M and y==M:
#                nimage[ix,iy] = np.array([255,0,0], dtype=np.uint8)

diff = np.array(ip.image, dtype=np.int16)
diff = np.array(np.clip(np.abs(diff - nimage)*2,0,255), dtype=np.uint8)

f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)

ax0.imshow(ip.img, interpolation='none')
ax1.imshow(ip.image, interpolation='none')
ax2.imshow(nimage, interpolation='none')
ax3.imshow(diff, interpolation='none')
plt.show()

