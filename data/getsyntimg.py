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


# load and cut the image to roi

img_name = '7k4r.png'

# start with 3 hardcoded squares
shapes = [
    ('sq', (48,73,146,161)),
    ('sq', (18,28,176,191)),
    ('sq', (60,71,184,196)),
]

# this only works for squares, it's a shortcut for the paper plot below.. don't acutally use this for production
# shapes2 = [ (t,c, np.s_[c[0]:c[1],c[2]:c[3]]) for t,c in shapes ]

# select region / cut
# (x1,x2,y1,y2)
# x is vertical
cut = (10,80,140,200)
cut = (0,100,120,220)

cutslice = np.s_[cut[0]:cut[1],cut[2]:cut[3]] # can directly be used on images

# setup the center / lens pos.. (in reference to the NEW / cutted image)
centrx = 177.95
centry = -53.54
# set offsets
offsx = cut[2]
offsy = cut[0]

center = (centrx-offsx, centry-offsy) 


image = ip.get_image(img_name, shapes, cut)
image2 = np.copy(image)



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

for x, xrow in enumerate(image):
    for y, rgb in enumerate(xrow):
        if sum(rgb) != 0:           #TODO make this depend on a binary mask
            pixel.append((x,y))

#pix_arr = np.array(pixel)

theta = [ip.px2arcs(p, center) for p in pixel]
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

image = np.copy(image2)
nimage = 0 * image

for (x,y), lst in np.ndenumerate(pxllist):
    n = len(lst)    
    if n>0:
        summ = np.array([0,0,0], dtype=np.int)
        for i in lst:
            ix,iy = pixel[i]
            summ += image[ix,iy]
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


if False:
    diff = np.array(image, dtype=np.int16)
    diff = np.array(np.clip(np.abs(diff - nimage),0,255), dtype=np.int16)
    diff = np.array((diff[:,:,0]+diff[:,:,1]+diff[:,:,2]) / 3. , dtype=np.uint8)
    
    f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
    
    ax0.imshow(ip.img, interpolation='none')
    ax1.imshow(image, interpolation='none')
    ax2.imshow(nimage, interpolation='none')
    ax3.imshow(diff, interpolation='none', cmap="gray")
    plt.show()


# nice plot for paper
f, axs = plt.subplots(1, 2)
(ax0, ax1) = axs

inpimg = np.copy(ip.img)

drawcol = np.array([0,255,0], dtype=np.uint8)

for t,c in shapes:
    x1,x2,y1,y2 = c
    for x in range(x1-1,x2+2):
        inpimg[x,y1-1,:] = drawcol
        inpimg[x,y2+1,:] = drawcol
    for y in range(y1-1,y2+2):
        inpimg[x1-1,y,:] = drawcol
        inpimg[x2+1,y,:] = drawcol

inpimg = inpimg[cutslice]

ax0.imshow(inpimg, interpolation='none')
ax1.imshow(nimage, interpolation='none')

for ax in axs:
    ax.axis("off")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#f.set_size_inches(2, 1)
plt.savefig('new_synth_img.png', bbox_inches='tight', pad_inches = 0, dpi=300)
#plt.show()

