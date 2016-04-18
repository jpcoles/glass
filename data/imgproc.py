# coding: utf-8

import numpy as np
import scipy as sp
from scipy import misc
import matplotlib.pyplot as plt



def px2arcs(coords, center, pxsize = 0.187):

    x = coords[1]
    y = -coords[0]

    cx, cy = center
#    return x+1j*y
    
#    x = ( x-(centrx-offsx) ) * pxsize
#    y = ( y-(centry-offsy) ) * pxsize

    x = ( x-cx ) * pxsize
    y = ( y-cy ) * pxsize
    
    return x+1j*y
    

img = None
image = None

def get_image(iname, shapes, cut):

    global img
    global image
    
    img = misc.imread(iname)
    nimg = np.zeros(img.shape, dtype=np.uint8)
    
#    # start with 3 hardcoded squares
#    shapes = [
#        ('sq', (48,73,146,161)),
#        ('sq', (18,28,176,191)),
#        ('sq', (60,71,184,196)),
#    ]
#    
#    # select region / cut
#    # (x1,x2,y1,y2)
#    # x is vertical
#    cut = (0,100,100,200)
#    
    
    for tjpe, parameters in shapes:
        if tjpe=="sq": #square:
            x1,x2,y1,y2 = parameters
            nimg[x1:x2,y1:y2,:] = img[x1:x2,y1:y2,:]
    
    #nimg[48:73,146:161,:] = img[48:73,146:161,:]
    #nimg[18:28,176:191,:] = img[18:28,176:191,:]
    #nimg[60:71,184:196,:] = img[60:71,184:196,:]
    
    image = nimg[cut[0]:cut[1], cut[2]:cut[3], : ]
    
    return image


def get_mask(shapes, cutslice):

    global img
    mask = np.zeros(img.shape[:-1], dtype=np.bool) # fill with False
    
    for tjpe, parameters in shapes:
        if tjpe=="sq": #square:
            x1,x2,y1,y2 = parameters
            mask[x1:x2,y1:y2] = True
            
    return mask[cutslice]

#bimg = np.zeros(img.shape[0:1], dtype=np.bool)
#
#bimg[48:73,146:161] = np.ones()
#bimg[18:28,176:191] = True
#bimg[60:71,184:196] = True
#bimg2 = bimg[0:100,100:200,:]


#plt.imshow(nimg2)
#plt.show()
