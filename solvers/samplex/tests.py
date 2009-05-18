import sys
from numpy import array, radians, linspace
from pylab import show, scatter, figure, plot, xlim, ylim
from samplex import Samplex
from math import cos, sin, pi


def NR1(s):
    s.leq(array([-740, 1, 0,  2,  0]))
    s.leq(array([0,   0,  2,  0, -7]))
    s.geq(array([-0.5, 0,  1,- 1,  2]))
    s.eq( array([-9,   1, 1,  1,  1]))

def poly0(s):

    rad = 5.
    x = 5.
    y = 15.
    
    nPnts = 21
    d = 0.
    p = []
    xs = []
    ys = []
    ls = linspace(0, 360, nPnts+1)
    for d in ls:
        print "!!!!!", d
        p.append([rad * cos(d * pi / 180.) + x,
                  rad * sin(d * pi / 180.) + y,
                  0, 0])
        #p.append([rad * cos(radians(d)) + x,
        #          rad * sin(radians(d)) + y,
        #          0, 0])
        xs.append(p[-1][0])
        ys.append(p[-1][1])

    print xs
    print ys 

    plot(xs[:-1], ys[:-1])

    # Apply the constraints.
    for i in range(nPnts):
        rise = p[i+1][1] - p[i][1]
        run  = p[i+1][0] - p[i][0]

        #if i==4: rise /= 2

        print rise, run

        if   (abs(run) < 1e-13): p[i][2] = None
        else:                    p[i][2] = rise / run

        if p[i][2] is None:
            print "HERE"
            s.geq(array([-p[i][0], 1, 0]))
        else:
            p[i][3] = p[i][1] - p[i][2] * p[i][0]


            print "????? %.20e %.20e %.20e %.20e %i %.20e" % (rise, run, p[i+1][1], p[i][1], i, ls[i])
            #print "  ??? %.20

            m = p[i][2]
            b = p[i][3]

            if (b + m*x - y >= 0): s.geq(array([ b, m, -1], 'double'))
            else:                  s.leq(array([ b, m, -1], 'double'))

    return (0, 0, max(xs), max(ys))

def poly1(s):

    s.leq(array([-25000.0, 1, 1, 1]))
    s.leq(array([-1620, 0.06, 0.07, 0.08]))
    s.leq(array([-6000, 0, 1, -1]))


def XXX(s):
    s.eq(array([2, 0, -6, 1, 0]))
    s.eq(array([0, 0,  0, 0, 0]))
    s.eq(array([0, 0,  0, 0, 0]))
    s.eq(array([8, 0, 3, -4, 0]))

def fromFile(s, filename):

    fp = open(filename, 'r')
    for line in fp:
        line = line.split()
        print line
        if   line[0] == "eq":  s.eq(array(map(float, line[2:])))
        elif line[0] == "leq": s.leq(array(map(float, line[2:])))
        elif line[0] == "geq": s.geq(array(map(float, line[2:])))

def run(fs):
    for f in fs:
        figure()

        s = Samplex()
        f(s)
        s.start()
        print s.data

        xs = []
        ys = []
        cs = []
        for v, [sol,c] in s.next(10000):
            pass
            #print v
            if not xs:
                cs.append('g')
            else:
                cs.append('b')
            xs.append(v[1])
            ys.append(v[2])

            xs.append(sol[1])
            ys.append(sol[2])
            cs.append(c)

        scatter(xs, ys, c=cs)

    show()


#run([poly1])
run([lambda s: fromFile(s, sys.argv[1])])

