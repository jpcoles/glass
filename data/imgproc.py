# coding: utf-8

import numpy as np
import scipy as sp
from scipy import misc
import matplotlib.pyplot as plt



def px2arcs(coords):

    x = coords[1]
    y = -coords[0]

#    return x+1j*y

    centrx = 177.95
    offsx = 100.0
    centry = -53.54
    offsy = 0.0

    pxsize = 0.187
    
    x = ( x-(centrx-offsx) ) * pxsize
    y = ( y-(centry-offsy) ) * pxsize
    
    return x+1j*y
    



img = misc.imread('7k4r.png')
nimg = np.zeros(img.shape, dtype=np.uint8)

nimg[48:73,146:161,:] = img[48:73,146:161,:]
nimg[18:28,176:191,:] = img[18:28,176:191,:]
nimg[60:71,184:196,:] = img[60:71,184:196,:]

image = nimg[0:100,100:200,:]


#bimg = np.zeros(img.shape[0:1], dtype=np.bool)
#
#bimg[48:73,146:161] = np.ones()
#bimg[18:28,176:191] = True
#bimg[60:71,184:196] = True
#bimg2 = bimg[0:100,100:200,:]


#plt.imshow(nimg2)
#plt.show()
