# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:39:04 2014

@author: rafik
"""

from __future__ import division

import numpy as np
#import scipy as sp
#from scipy import misc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import imgproc as ip


# load and cut the image to roi

img_name = '7k4r.png'

# start with 3 hardcoded squares
shapes = [
    ('sq', (48,73,146,161)),
    ('sq', (18,28,176,191)),
    ('sq', (60,71,184,196)), # (60,71,184,196))
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
mask = ip.get_mask(shapes, cutslice)

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

X = np.int32(np.floor(M*(1+d_source.real/r)+.5))
Y = np.int32(np.floor(M*(1+d_source.imag/r)+.5))

#plt.scatter(X,Y)
#plt.show()

pxllist = np.zeros((2*M+1,2*M+1), dtype=list)
srcimg  = np.zeros((2*M+1,2*M+1,3), dtype=np.uint8)
srcimg_gray = np.zeros((2*M+1,2*M+1), dtype=np.uint8)
cntimg  = np.zeros((2*M+1,2*M+1), dtype=np.uint8)


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
        summ = np.array([0,0,0], dtype=np.int32)
        for i in lst:
            ix,iy = pixel[i]
            summ += image[ix,iy]
        pxlave = np.uint8(np.clip((summ / n), 0, 255))

        srcimg[x,y] = pxlave
        srcimg_gray[x,y] = np.average(pxlave)

        cntimg[x,y] = n
        
        for i in lst:
            ix,iy = pixel[i]
            #print summ
#            nimage[ix,iy] = ip.image[ix,iy]
            nimage[ix,iy] = pxlave

#            if np.sum(summ)==0:
#                print summ, x, y, ix, iy

#            if x==M and y==M:
#                nimage[ix,iy] = np.array([255,0,0], dtype=np.uint8)


# nimage: the actual output!!

# prepare input image
inpimg = np.copy(ip.img)

drawcol = np.array([0,255,0], dtype=np.uint8) # color for drwaing the shapes

for t,c in shapes:
    x1,x2,y1,y2 = c
    for x in range(x1-1,x2+2):
        inpimg[x,y1-1,:] = drawcol
        inpimg[x,y2+1,:] = drawcol
    for y in range(y1-1,y2+2):
        inpimg[x1-1,y,:] = drawcol
        inpimg[x2+1,y,:] = drawcol

# finally cut the image
inpimg = inpimg[cutslice]

# composite new image:
new_img_comp = ip.img[cutslice]
gray = np.average(new_img_comp, axis=2, weights=[0.2989, 0.5870, 0.1140])
new_img_comp[:,:,0] = gray
new_img_comp[:,:,1] = gray
new_img_comp[:,:,2] = gray
new_img_comp[mask] = np.array([255,0,0])
new_img_comp[mask] = nimage[mask]


# difference image
diff = image.astype(np.int32)
diff = np.abs(diff - nimage)
diff = np.clip(np.mean(diff, axis=2, dtype=np.uint32), 0, 255).astype(np.uint8)


if False:
    
    f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
    
    ax0.imshow(ip.img, interpolation='none')
    ax1.imshow(image, interpolation='none')
    ax2.imshow(nimage, interpolation='none')
    ax3.imshow(diff, interpolation='none', cmap="gray")
    plt.show()

if True:
    
    gs0  = gridspec.GridSpec(2, 1, height_ratios=[2,1])
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[1])
    ax00 = plt.subplot(gs00[0])
    ax01 = plt.subplot(gs00[1])
    ax10 = plt.subplot(gs01[0])
    ax11 = plt.subplot(gs01[1])
    ax12 = plt.subplot(gs01[2])
    ax13 = plt.subplot(gs01[3])
    axs = (ax00, ax01, ax10, ax11, ax12, ax13)
    
    colors = [ 'black', 'red', 'orange', 'yellow', 'green', 'blue', 'magenta', 'white']
    bounds = [0,      2,     4,        8,       16,       32,      64,      128,   1024]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
#    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax00.imshow(inpimg, interpolation='none')
    ax01.imshow(new_img_comp, interpolation='none')
    ax10.imshow(srcimg, interpolation='none')
    ax11.imshow(srcimg_gray/255., interpolation='none', cmap="magma")
    ax12.imshow(cntimg, interpolation='none', cmap=cmap, norm=norm, vmin=0, vmax=100)
    ax13.imshow(diff/255., interpolation='none', cmap="magma")
    
#    ax0.imshow(srcimg, interpolation='none')
#    ax1.imshow(srcimg_gray/255., interpolation='none', cmap="magma")
#    ax2.imshow(cntimg, interpolation='none', cmap=cmap, norm=norm, vmin=0, vmax=100)
    #cbar = fig.colorbar(ax2)
    for ax in axs:
        ax.axis("off")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
    

if False:
    # nice plot for paper
    f, axs = plt.subplots(1, 2)
    (ax0, ax1) = axs
    
    
    ax0.imshow(inpimg, interpolation='none')
    ax1.imshow(nimage, interpolation='none')
    
    for ax in axs:
        ax.axis("off")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    #f.set_size_inches(2, 1)
    plt.savefig('new_synth_img.png', bbox_inches='tight', pad_inches = 0, dpi=300)
    #plt.show()
    plt.close()
    
