from numpy import arctan, log, vectorize
from math import pi, sin, cos

@vectorize
def poten2d(x,y,a):
    return poten(complex(x,y),a)

@vectorize
def poten(r, a):
    x,y = r.real, r.imag
    xm = x - a/2.
    xp = x + a/2.
    ym = y - a/2.
    yp = y + a/2.

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2

    v  = -3 * a**2
    v += xm2*arctan(ym/xm) + ym2*arctan(xm/ym)
    v += xp2*arctan(yp/xp) + yp2*arctan(xp/yp)
    v -= xm2*arctan(yp/xm) - yp2*arctan(xm/yp)
    v -= xp2*arctan(ym/xp) - ym2*arctan(xp/ym)
    v += xm*ym*log(xm2 + ym2)
    v += xp*yp*log(xp2 + yp2)
    v -= xp*ym*log(xp2 + ym2)
    v -= xm*yp*log(xm2 + yp2)
    return v / (2*pi)

@vectorize
def poten_x(r, a):
    x,y = r.real, r.imag
    xm = x - a/2.
    xp = x + a/2.
    ym = y - a/2.
    yp = y + a/2.

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2

    v  = xm*arctan(ym/xm)
    v += xp*arctan(yp/xp)
    v -= xm*arctan(yp/xm)
    v -= xp*arctan(ym/xp)
    v += ym*log(xm2 + ym2)/2
    v += yp*log(xp2 + yp2)/2
    v -= ym*log(xp2 + ym2)/2
    v -= yp*log(xm2 + yp2)/2
    return v / pi

@vectorize
def poten_y(r, a):
    x,y = r.real, r.imag
    xm = x - a/2.
    xp = x + a/2.
    ym = y - a/2.
    yp = y + a/2.

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2

    v  = ym*arctan(xm/ym)
    v += yp*arctan(xp/yp)
    v -= ym*arctan(xp/ym)
    v -= yp*arctan(xm/yp)
    v += xm*log(xm2 + ym2)/2 
    v += xp*log(xp2 + yp2)/2
    v -= xm*log(xm2 + yp2)/2 
    v -= xp*log(xp2 + ym2)/2
    return v / pi

@vectorize
def poten_xy(r, a):
    x,y = r.real, r.imag
    xm = x - a/2.
    xp = x + a/2.
    ym = y - a/2.
    yp = y + a/2.

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2
    v = log(xp2+yp2) + log(xm2+ym2)  \
      - log(xp2+ym2) - log(xm2+yp2)
    return v/(2*pi)

@vectorize
def poten_xx(r, a):
    x,y = r.real, r.imag
    xm = x - a/2.
    xp = x + a/2.
    ym = y - a/2.
    yp = y + a/2.

    v = arctan(yp/xp) + arctan(ym/xm) \
      - arctan(yp/xm) - arctan(ym/xp)
    return v/pi

@vectorize
def poten_yy(r, a):
    x,y = r.real, r.imag
    xm = x - a/2.
    xp = x + a/2.
    ym = y - a/2.
    yp = y + a/2.
    v = arctan(xp/yp) + arctan(xm/ym) \
      - arctan(xp/ym) - arctan(xm/yp)
    return v/pi

def maginv(r, theta, a):
    xx = poten_xx(r,a)
    yy = poten_yy(r,a)
    delta = poten_xy(r,a);

    theta *= pi/90
    cs = cos(theta)
    sn = sin(theta)

    kappa = (xx+yy)/2
    gamma = (xx-yy)/2
    return [-sn*gamma + cs*delta,
            kappa + cs*gamma + sn*delta,
            kappa - cs*gamma - sn*delta]
    
