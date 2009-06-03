from __future__ import division
from numpy import arctan, log, vectorize
from math import pi, sin, cos

@vectorize
def poten2d(x,y,a):
    return poten(complex(x,y),a)

@vectorize
def poten_indef(r):
    x,y = r.real, r.imag
    x2,y2 = x**2, y**2
    v  = (x!=0) and x2 * arctan(y/x)
    v += (y!=0) and y2 * arctan(x/y)
    v += (x!=0 and y!=0) and x*y*(log(x2+y2) - 3) 
    return v / (2*pi)

@vectorize
def poten2d_indef(x,y):
    return poten_indef(complex(x,y))

@vectorize
def poten(r, a):
    x,y = r.real, r.imag
    xm,xp = x - a/2, x + a/2
    ym,yp = y - a/2, y + a/2

    xm2, xp2 = xm**2, xp**2
    ym2, yp2 = ym**2, yp**2

    # FIXME: What happens when we divide by 0 or do log 0?
    v = ( -3 * a**2
      + ( xm2*arctan(ym/xm) + ym2*arctan(xm/ym) + xm*ym*log(xm2 + ym2) )
      + ( xp2*arctan(yp/xp) + yp2*arctan(xp/yp) + xp*yp*log(xp2 + yp2) )
      - ( xm2*arctan(yp/xm) + yp2*arctan(xm/yp) + xp*ym*log(xp2 + ym2) )
      - ( xp2*arctan(ym/xp) + ym2*arctan(xp/ym) + xm*yp*log(xm2 + yp2) ))
    return v / (2*pi)

@vectorize
def poten_dx(r, a):
    x,y = r.real, r.imag
    xm = x - a/2
    xp = x + a/2
    ym = y - a/2
    yp = y + a/2

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2

    v = (xm*arctan(ym/xm)    + xp*arctan(yp/xp)
       - xm*arctan(yp/xm)    - xp*arctan(ym/xp)
       + ym*log(xm2 + ym2)/2 + yp*log(xp2 + yp2)/2
       - ym*log(xp2 + ym2)/2 - yp*log(xm2 + yp2)/2)
    return v / pi

@vectorize
def poten_dy(r, a):
    x,y = r.real, r.imag
    xm = x - a/2
    xp = x + a/2
    ym = y - a/2
    yp = y + a/2

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2

    v = (ym*arctan(xm/ym)    + yp*arctan(xp/yp)
       - ym*arctan(xp/ym)    - yp*arctan(xm/yp)
       + xm*log(xm2 + ym2)/2 + xp*log(xp2 + yp2)/2
       - xm*log(xm2 + yp2)/2 - xp*log(xp2 + ym2)/2)
    return v / pi

@vectorize
def poten_dxdy(r, a):
    x,y = r.real, r.imag
    xm = x - a/2
    xp = x + a/2
    ym = y - a/2
    yp = y + a/2

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2
    v = log(xp2+yp2) + log(xm2+ym2)  \
      - log(xp2+ym2) - log(xm2+yp2)
    return v/(2*pi)

@vectorize
def poten_dxdx(r, a):
    x,y = r.real, r.imag
    xm = x - a/2
    xp = x + a/2
    ym = y - a/2
    yp = y + a/2

    v = arctan(yp/xp) + arctan(ym/xm) \
      - arctan(yp/xm) - arctan(ym/xp)
    return v/pi

@vectorize
def poten_dydy(r, a):
    x,y = r.real, r.imag
    xm = x - a/2
    xp = x + a/2
    ym = y - a/2
    yp = y + a/2
    v = arctan(xp/yp) + arctan(xm/ym) \
      - arctan(xp/ym) - arctan(xm/yp)
    return v/pi

def maginv(r, theta, a):
    xx    = poten_dxdx(r,a)
    yy    = poten_dydy(r,a)
    delta = poten_dxdy(r,a)

    theta *= pi/90
    cs = cos(theta)
    sn = sin(theta)

    kappa = (xx+yy)/2
    gamma = (xx-yy)/2
    return [-sn*gamma + cs*delta,
            kappa + cs*gamma + sn*delta,
            kappa - cs*gamma - sn*delta]
    
