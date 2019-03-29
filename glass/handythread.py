import sys
import time
import threading
import operator
from itertools import count
from functools import reduce

def foreach(f,l,threads=3,return_=False):
    """
    Apply f to each element of l, in parallel
    """

    if threads>1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            d = {}
            i = zip(count(),l.__iter__())
        else:
            i = l.__iter__()


        def runall():
            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = next(i)
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n,x = v
                        d[n] = f(x)
                    else:
                        f(v)
                except:
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()
        
        threadlist = [threading.Thread(target=runall) for j in range(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            a, b, c = exceptions[0]
            raise a(b).with_traceback(c)
        if return_:
            r = list(d.items())
            r.sort()
            return [v for (n,v) in r]
    else:
        if return_:
            return [f(v) for v in l]
        else:
            for v in l:
                f(v)
            return

def parallel_map(f,l,threads=3,return_=True):
    return foreach(f,l,threads=threads,return_=return_)

def parallel_map2(f,l,threads=3):
    if threads > 1:
        size = max(1, len(l)//threads)
        d = {}

        def runall(id, work):
            d[id] = list(map(f, work))
        
        threadlist = [threading.Thread(target=runall, args=(j, l[j*size:(j+1)*size])) for j in range(threads)]
        for t in threadlist: t.start()
        for t in threadlist: t.join()

        return reduce(operator.add, iter(d.values()))
    else:
        return list(map(f,l))

if __name__=='__main__':
    def f(x):
        print(x)
        time.sleep(0.5)
    foreach(f,list(range(10)))
    def g(x):
        time.sleep(0.5)
        print(x)
        raise ValueError(x)
        time.sleep(0.5)
    foreach(g,list(range(10)))

