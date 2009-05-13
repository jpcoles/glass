from numpy import zeros
from math import hypot, atan2

import priors, constraints

def radius(x, y):
    return int(hypot(x,y)+0.5)

class PixelBasis: 
    symm = False
    L = 0
    maprad = 0
    shear = None
    minsteep = 0
    maxsteep = 0
    cen_ang = 0
    h_spec = 0

    def init(self, obj):
        param = obj.basis
        L = param.L

        # Create pixel map
        pmap = zeros((2*L+1, 2*L+1), 'double')
        npix = 0
        for i in xrange(-L,L+1):
            for j in xrange(-L,L+1):
                if radius(i,j) < L:
                    if pmap[L+i,L+j] == 0:
                        npix += 1
                        pmap[L+i,L+j] = npix
                        if param.symm: pmap[L-i,L-j] = npix

        # Create inverse pixel map
        imap = zeros((npix+1,4), 'int')
        for i in xrange(-L,L+1):
            for j in xrange(-L,L+1):
                p = pmap[L+i,L+j]
                if p != 0: imap[p] = radius(i,j), i, j, atan2(j,i)

        for i in xrange(1, npix+1):
            lr  = imap[i][0]
            lth = imap[i][3]
            l = i
            while l > 1:
                pr = imap[l-1][0]
                th = imap[l-1][3]
                if pr < lr: break
                if pr == lr and th < lth: break
                imap[l] = imap[l-1]
                l -= 1
            imap[l] = imap[i]

        for i in xrange(1, npix+1):
            i,j = imap[i][1:3]
            pmap[L+i,L+j] = i
            if param.symm: pmap[L-i,L-j] = i

        # Create rings
        rings = zeros((L+1,2), 'int')
        for i in xrange(L+1): rings[i] = npix, 0
        for i in xrange(1,npix+1):
            pr, i, j = imap[i][0:3]
            if rings[pr][0] > i: rings[pr][0] = i
            if rings[pr][1] < i: rings[pr][1] = i


        # Calculate map radius. Provide a few extra pixels outside of the 
        # outermost image.
        if param.maprad > 0: 
            param.maprad /= L
        else:
            for s in obj.systems:
                rmin = 1e30
                rmax = -1e30
                rad  = 1e30
                for d in s[1:]:
                    x,y = d[1:3]
                    r = hypot(x,y)
                    rmin = min(r, rmin)
                    rmax = max(r, rmax)

                if rmin+rmax < rad: rad = rmin+rmax
                if 2*rmax - rmin < rad: rad = 2*rmax - rmin
                if param.maprad < rad: param.maprad = rad
            param.maprad = (L+1) * param.maprad/(L**2)
            print "Map radius = %f" % (param.maprad*L)
