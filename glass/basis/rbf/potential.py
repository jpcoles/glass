
import numpy as np
from numpy import arctan, log, vectorize, array, trunc
from math import pi, sin, cos
from glass.environment import Environment

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

#@vectorize
def poten(r, a, R):

    x,y = r.real, r.imag

    xm,xp = x - a/2, x + a/2
    ym,yp = y - a/2, y + a/2

    xm2, xp2 = xm**2, xp**2
    ym2, yp2 = ym**2, yp**2

    # What happens when we divide by 0 or do log 0?
    # xm and ym are guaranteed never to be zero from the definition. x and y may be zero.
    v = ( -3 * a**2
      + ( xm2*arctan(ym/xm) + ym2*arctan(xm/ym) + xm*ym*log(xm2 + ym2) )
      + ( xp2*arctan(yp/xp) + yp2*arctan(xp/yp) + xp*yp*log(xp2 + yp2) )
      - ( xm2*arctan(yp/xm) + yp2*arctan(xm/yp) + xp*ym*log(xp2 + ym2) )
      - ( xp2*arctan(ym/xp) + ym2*arctan(xp/ym) + xm*yp*log(xm2 + yp2) ))

    #v -= v * np.sin(np.angle(r)*2) / 2 * (abs(r) >= R-a)

    return v / (2*pi)

#@vectorize
def poten_dx(r, a, R):
    x,y = r.real, r.imag
    xm = x - a/2
    xp = x + a/2
    ym = y - a/2
    yp = y + a/2

    xm2 = xm**2
    xp2 = xp**2
    ym2 = ym**2
    yp2 = yp**2

    v = ( (xm*arctan(ym/xm)  + xp*arctan(yp/xp)) + (ym*log(xm2 + ym2) + yp*log(xp2 + yp2)) / 2
      -   (xm*arctan(yp/xm)  + xp*arctan(ym/xp)) - (ym*log(xp2 + ym2) + yp*log(xm2 + yp2)) / 2 )
    #v -= v * np.sin(np.angle(r)*2) / 2 * (abs(r) >= R-a)
    return v / pi

def XXpoten_dx(r, a):
    from scipy import weave
    from numpy import empty_like

    if type(r) == type(complex(0,0)):
        return XXpoten_dx

    v = empty_like(r.real)
    code="""
        int i;
        double vi, xi,yi;
        std::complex<double> ri;
        for (i=0; i < l; i++)
        {
            //ri = py_to_complex(PyList_GetItem(r.ptr(),i),"r");
            xi = std::real(r[i]);
            yi = std::imag(r[i]);

            double xm = xi - a[i]/2;
            double xp = xi + a[i]/2;
            double ym = yi - a[i]/2;
            double yp = yi + a[i]/2;

            double xm2 = xm*xm;
            double xp2 = xp*xp;
            double ym2 = ym*ym;
            double yp2 = yp*yp;
            vi = ( (xm*atan(ym/xm)  + xp*atan(yp/xp)) + (ym*log(xm2 + ym2) + yp*log(xp2 + yp2)) / 2
              -   (xm*atan(yp/xm)  + xp*atan(ym/xp)) - (ym*log(xp2 + ym2) + yp*log(xm2 + yp2)) / 2 );
            vi /= pi;
            v[i] = vi;
        }
    """

    l = len(r)
    weave.inline(code, ['l', 'r', 'v', 'pi', 'a'])
    return v

#@vectorize
def poten_dy(r, a, R):
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
    #v -= v * np.sin(np.angle(r)*2)/2 * (abs(r) >= R-a)
    return v / pi

def XXpoten_dy(r, a):
    from scipy import weave
    from numpy import empty_like

    if type(r) == type(complex(0,0)):
        return XXpoten_dx

    v = empty_like(r.real)
    code="""
        int i;
        double vi, xi,yi;
        std::complex<double> ri;
        for (i=0; i < l; i++)
        {
            //ri = py_to_complex(PyList_GetItem(r.ptr(),i),"r");
            xi = std::real(r[i]);
            yi = std::imag(r[i]);

            double xm = xi - a[i]/2;
            double xp = xi + a[i]/2;
            double ym = yi - a[i]/2;
            double yp = yi + a[i]/2;

            double xm2 = xm*xm;
            double xp2 = xp*xp;
            double ym2 = ym*ym;
            double yp2 = yp*yp;
            //vi = ( (xm*atan(ym/xm)  + xp*atan(yp/xp)) + (ym*log(xm2 + ym2) + yp*log(xp2 + yp2)) / 2
              //-   (xm*atan(yp/xm)  + xp*atan(ym/xp)) - (ym*log(xp2 + ym2) + yp*log(xm2 + yp2)) / 2 );
            vi = (ym*atan(xm/ym)    + yp*atan(xp/yp)
               - ym*atan(xp/ym)    - yp*atan(xm/yp)
               + xm*log(xm2 + ym2)/2 + xp*log(xp2 + yp2)/2
               - xm*log(xm2 + yp2)/2 - xp*log(xp2 + ym2)/2);
            vi /= pi;
            v[i] = vi;
        }
    """

    l = len(r)
    weave.inline(code, ['l', 'r', 'v', 'pi', 'a'])
    return v

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
def poten_dydx(r, a):
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
    #print 'maginv', r, theta, a
    xx    = poten_dxdx(r,a)
    yy    = poten_dydy(r,a)
    delta = poten_dxdy(r,a)

    theta *= pi/180
    cs = cos(2*theta)
    sn = sin(2*theta)

    kappa = (xx+yy)/2
    gamma = (xx-yy)/2
    return array([    0 - sn*gamma + cs*delta,
                  kappa + cs*gamma + sn*delta,
                  kappa - cs*gamma - sn*delta])

def maginv_new(r, theta, a):
    #print 'maginv', r, theta, a
    xx = poten_dxdx(r,a)
    yy = poten_dydy(r,a)
    xy = poten_dxdy(r,a)

    theta *= pi/180
    c_2 = cos(2*theta)
    s_2 = sin(2*theta)

    c2 = cos(theta) ** 2
    s2 = sin(theta) ** 2

    alpha = xx*c2  + xy*s_2 + yy*s2
    beta  = xy*c_2 + (xx-yy)/2 * s_2
    delta = yy*c2  + xy*s_2 + xx*s2

    return [beta, alpha, delta]
    

def maginv_new3(r, theta, a, zcap):
    #print 'maginv', r, theta, a
    xx = zcap - poten_dxdx(r,a)
    yy = zcap - poten_dydy(r,a)
    xy = zcap - poten_dxdy(r,a)

    theta *= pi/180
    c = cos(theta)
    s = sin(theta)

    P = mat([[c,s], [-s,c]])
    A = mat([[xx, xy], [xy, yy]])

    D = P*A*P.I
    return [D[0,1], D[0,0], D[1,1]]

def maginv_new4(r, theta, a):
    #print 'maginv', r, theta, a
    xx = poten_dxdx(r,a)
    yy = poten_dydy(r,a)
    xy = poten_dxdy(r,a)

    theta *= pi/180
    c_2 = cos(2*theta)
    s_2 = sin(2*theta)
    c2  = cos(theta) ** 2
    s2  = sin(theta) ** 2

    alpha =  s_2*xy - c2*xx - s2*yy
    beta  = -s_2*xy - c2*yy - s2*xx
    delta =  s_2*(yy-xx)/2  - c_2*xy

    return [delta, alpha, beta]



def maginv_new5(r, a, c, s):
    #print 'maginv', r, theta, a
    xx = poten_dxdx(r,a)
    yy = poten_dydy(r,a)
    xy = poten_dxdy(r,a)

    c2  = c ** 2
    s2  = s ** 2

    alpha = -c2*xx - s2*yy + 2*s*c*xy
    beta  = -s2*xx - c2*yy - 2*s*c*xy
    delta =  s*c*(yy-xx) - xy*(c2-s2)

    return [delta, alpha, beta]


def grad(W,r0,r,a):
    from scipy import weave

    if type(r) == type(complex(0,0)):
        assert 0, "Shouldn't use potential.grad() for none array-based positions."

    code="""
        int i;
        std::complex<double> v(0,0);
        //Py_BEGIN_ALLOW_THREADS
        double xx=0,yy=0;
        #ifdef WITH_OMP
        omp_set_num_threads(threads);
        #endif
        #pragma omp parallel for reduction(+:xx) reduction(+:yy)
        for (i=0; i < l; i++)
        {
            double vx,vy;
            std::complex<double> dr = r0-r[i];
            const double xi = std::real(dr);
            const double yi = std::imag(dr);

            const double xm = xi - a[i]/2;
            const double xp = xi + a[i]/2;
            const double ym = yi - a[i]/2;
            const double yp = yi + a[i]/2;

            const double xm2 = xm*xm;
            const double xp2 = xp*xp;
            const double ym2 = ym*ym;
            const double yp2 = yp*yp;

            const double log_xm2_ym2 = log(xm2 + ym2);
            const double log_xp2_yp2 = log(xp2 + yp2);
            const double log_xp2_ym2 = log(xp2 + ym2);
            const double log_xm2_yp2 = log(xm2 + yp2);

            vx = (xm*atan(ym/xm) + xp*atan(yp/xp)) + (ym*log_xm2_ym2 + yp*log_xp2_yp2) / 2
               - (xm*atan(yp/xm) + xp*atan(ym/xp)) - (ym*log_xp2_ym2 + yp*log_xm2_yp2) / 2;
            vx /= pi;
            vx *= W[i];

            vy = (ym*atan(xm/ym) + yp*atan(xp/yp)) + (xm*log_xm2_ym2 + xp*log_xp2_yp2) / 2
               - (ym*atan(xp/ym) + yp*atan(xm/yp)) - (xm*log_xm2_yp2 + xp*log_xp2_ym2) / 2;
            vy /= pi;
            vy *= W[i];
            //v += std::complex<double>(vx,vy);
            xx += vx;
            yy += vy;
        }

        //Py_END_ALLOW_THREADS
        return_val = std::complex<double>(xx,yy);
    """

    l = len(r)
    threads = Environment.global_opts['ncpus']
    kw = Environment.global_opts['omp_opts']
    v = weave.inline(code, ['l', 'W','r0','r', 'pi', 'a', 'threads'], **kw)
    return v

