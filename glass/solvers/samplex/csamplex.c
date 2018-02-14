#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <time.h>
#include <unistd.h>
#include <sys/times.h>
#include <signal.h>

#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <errno.h>

#include "timing.h"

/*==========================================================================*/
/* Use double or long double as the data type for the simplex tableau.      */
/* double is much faster but long double has more precision.                */
/* If a given problem is small enough that double will suffice then there   */
/* is no benefit from using long double.                                    */
/*==========================================================================*/
#define USE_LONG_DOUBLE 0

/*==========================================================================*/
/* After the table has been allocated we can reorganize it by letting each  */
/* thread reallocate a bit of the table memory. This puts the memory        */
/* that each thread will be working with local to the CPU. This yields a    */
/* 2x overall speed up.                                                     */
/*==========================================================================*/
#define REORGANIZE_TABLE_MEMORY 0
#define SET_THREAD_AFFINITY     0

#define WITH_GOOGLE_PROFILER 0

#if WITH_GOOGLE_PROFILER
#include <google/profiler.h>
#endif

/*==========================================================================*/
/* Timing variables                                                         */
/*==========================================================================*/
CPUDEFS

/*==========================================================================*/
/* Debugging tools                                                          */
/*==========================================================================*/
#define DBG_LEVEL 0
#define DBG(lvl) if (DBG_LEVEL >= lvl) 


#if USE_LONG_DOUBLE
typedef long double dble_t;
#define EPS ((dble_t)1.0e-17L)
#define INF ((dble_t)1.0e12L)
#define SML ((dble_t)1.0e-8L)
#define EPS_EXP (-56)
#define ABS fabsl
#else
//typedef double dble_t __attribute__ ((aligned(8)));
#define dble_t double
#define EPS ((dble_t)1e-14)
#define INF ((dble_t)1e+12) 
#define SML ((dble_t)1e-08)
#define EPS_EXP (-46)
#define ABS fabs
#endif

#if 0
#define ABS(n) \
    (sizeof(n) == sizeof(float) \
     ? fabsf(n) \
     : sizeof(n) == sizeof(double) \
     ? fabs(n) : fabsl(n))
#endif

#define LOG(n) \
    (sizeof(n) == sizeof(float) \
     ? logf(n) \
     : sizeof(n) == sizeof(double) \
     ? log(n) : logl(n))

#define FREXP(n) \
    (sizeof(n) == sizeof(float) \
     ? frexpf(n) \
     : sizeof(n) == sizeof(double) \
     ? frexp(n) : frexpl(n))

#define FMA(x,y,z) \
    (sizeof(x) == sizeof(float) \
     ? fmaf(x,y,z) \
     : sizeof(x) == sizeof(double) \
     ? fma(x,y,z) : fmal(x,y,z))

#define TEST(x) \
    (sizeof(x) == sizeof(float) \
     ? fprintf(stderr, "float\n") \
     : sizeof(x) == sizeof(double) \
     ? fprintf(stderr, "double\n") : fprintf(stderr, "long double\n"))

enum 
{
    INFEASIBLE     = 0,
    FEASIBLE       = 1,
    NOPIVOT        = 2,
    FOUND_PIVOT    = 3,
    UNBOUNDED      = 4,
};

inline int32_t min(int32_t a, int32_t b)
{
    return (a < b) ? a : b;
}

inline dble_t rtz(dble_t v)
{
    //fprintf(stderr, "WARNING: rtz turned off!\n");
    //return v;
    //if (ABS(v) <= EPS) return 0;
    return v;
    //return v * (ABS(v) > EPS);
}

/*==========================================================================*/
/* Memory allocation wrappers.                                              */
/*==========================================================================*/

#if DBG_LEVEL >= 10000
long total_alloc = 0;
#define CALLOC(type, num) \
    (total_alloc += sizeof(type) * (num), \
    fprintf(stderr, "c'allocating %ld bytes [already alloc'd: %ld].\n", sizeof(type) * (num), total_alloc), \
    ((type *)calloc((num), sizeof(type))))
#else
#define CALLOC(type, num) ((type *)calloc((num), sizeof(type)))
#endif

#define MALLOC(type, num) ((type *)malloc((num) * sizeof(type)))

/*==========================================================================*/
/* Data structures for the table and arrays.                                */
/*==========================================================================*/
typedef struct
{
    uint32_t cols;
    uint32_t rows;
    dble_t * restrict orig;
    dble_t * restrict data;
    dble_t * restrict pcol;
} matrix_t __attribute__ ((aligned(8)));

typedef struct pivot_thread_s
{
    pthread_t thr_id;

    int32_t id;
    int32_t start, end;

    matrix_t * restrict tabl;
    dble_t * restrict pcol;

    long L;
    dble_t piv, inc, coef;
    int32_t lpiv, rpiv;
    int32_t *left, *right;
    int32_t res;

    void (*action)(struct pivot_thread_s *thr);

} pivot_thread_t;

typedef struct 
{
    int32_t total_threads;          // Total number of threads
    int32_t nthreads;               // Number of usable threads 
    int32_t active_threads;         // Number of threads currently executing doPivot
    int32_t threads_initialized;
    pivot_thread_t *thr;
    pthread_mutex_t lock;
    pthread_cond_t activate_all;
    pthread_cond_t thread_finished;
} thread_pool_t;
#define EMPTY_POOL {0,0,0,0,NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER}

typedef struct
{
    int32_t step;
    int32_t nthreads;
    double obj_val;
} report_t;
report_t report = {0,0,0.0};


/*==========================================================================*/
/* The worker threads can be asked to do different things.                  */
/* They can either do the normal pivot operation, or they can capture       */
/* a portion of the table memory by allocating memory local to their CPU.   */
/*==========================================================================*/
#define PT_ACTION_STEAL_TABLE_COLS  (1)
#define PT_ACTION_DO_PIVOT          (2)

/* The pivot threads */
static thread_pool_t pool = EMPTY_POOL;

/* Do we need to update the columns that each thread is working on? */
static int32_t need_assign_pivot_threads;

PyObject *samplex_pivot(PyObject *self, PyObject *args);
void doPivot(pivot_thread_t *thr);
void doPivot0(matrix_t * restrict tabl,
    dble_t * restrict pcol,
    const long L,
    const dble_t piv, 
    const int32_t lpiv, const int32_t rpiv,
    int32_t start, const int32_t end);
void copymem(pivot_thread_t *thr);

static PyMethodDef csamplex_methods[] = 
{
    {"pivot", samplex_pivot, METH_O, "pivot"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcsamplex()
{
    (void)Py_InitModule("csamplex", csamplex_methods);
}

/*==========================================================================*/
/* PivotThread functions                                                    */
/*==========================================================================*/

void pivot_thread_init(pivot_thread_t *pt, int32_t id)
{
    pt->thr_id = 0;
    pt->id     = id;
    pt->tabl   = NULL;
    pt->pcol   = NULL;
    pt->start  = 
    pt->end    = 0;
    pt->action = doPivot;
}

void pivot_thread_reset(pivot_thread_t *pt, int32_t start, int32_t end)
{ 
    pt->start  = start;
    pt->end    = end;
    pt->action = NULL;
    if (pt->pcol != NULL)
    {
        free(pt->pcol);
        pt->pcol = NULL;
    }
}

void *pivot_thread_run(void *arg)
{
    pivot_thread_t *pt = (pivot_thread_t *)arg;

#if SET_THREAD_AFFINITY
    cpu_set_t mask;

    int32_t ncpus      = sysconf(_SC_NPROCESSORS_CONF);
    int32_t cpu_to_use = pt->id % ncpus;

    /* Try to fix the thread to a single CPU */

    CPU_ZERO(&mask);
    CPU_SET(cpu_to_use, &mask);

    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    {
        int32_t e = errno;
        fprintf(stderr, "Thread %i: %s\n", pt->id, strerror(e));
    }
    else DBG(2)
    {
        pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
        fprintf(stderr, "Thread %i on CPU%d\n", pt->id, cpu_to_use);
    }
#endif

    while (1)
    {
        pthread_mutex_lock(&pool.lock);
        pool.active_threads--;
        assert(pool.active_threads >= 0);
        DBG(1) fprintf(stderr, "THREAD %i done at=%i\n", pt->id, pool.active_threads);

        pthread_cond_signal(&pool.thread_finished);
        pthread_cond_wait(&pool.activate_all, &pool.lock);

        DBG(1) fprintf(stderr, "THREAD %i starting\n", pt->id);
        pthread_mutex_unlock(&pool.lock);

        if (pt->action)
            pt->action(pt);
    }

    return NULL;
}

static inline void startup_threads()
{
    pthread_mutex_lock(&pool.lock);
    pool.active_threads = pool.total_threads-1;
    pthread_cond_broadcast(&pool.activate_all);
    pthread_mutex_unlock(&pool.lock);
}

static inline void wait_for_threads()
{
    pthread_mutex_lock(&pool.lock);
    while (pool.active_threads != 0)
        pthread_cond_wait(&pool.thread_finished, &pool.lock);
    pthread_mutex_unlock(&pool.lock);
}

/*==========================================================================*/
/* initPivotThreads                                                         */
/*==========================================================================*/
void init_threads(int32_t n) 
{
    int32_t i;

    if (pool.threads_initialized) return;

    DBG(1) fprintf(stderr, "> initPivotThreads() nthreads=%i\n", n);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);

    pool.nthreads       = n;
    pool.total_threads  = n;
    pool.active_threads = 0;

#if SET_THREAD_AFFINITY
    if (n > 0)
    {
        /* The main thread runs on the first CPU */
        cpu_set_t mask;
        sched_getaffinity(0, sizeof(mask), &mask);
        CPU_ZERO(&mask);
        CPU_SET(0, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask) < 0)
            fprintf(stderr, "%s\n", strerror(errno));
        sched_getaffinity(0, sizeof(mask), &mask);
    }
#endif

    //--------------------------------------------------------------------------
    // Create the worker threads
    //--------------------------------------------------------------------------
    pool.thr = CALLOC(pivot_thread_t, pool.total_threads); assert(pool.thr != NULL);

    //--------------------------------------------------------------------------
    // Notice we start from 1 not 0, because we handle the first thread
    // specially later on. 
    //--------------------------------------------------------------------------
    pool.active_threads = pool.total_threads-1;
    for (i=1; i < pool.total_threads; i++)
    {
        pivot_thread_init(&pool.thr[i], i);
        pthread_create(&pool.thr[i].thr_id, &attr, pivot_thread_run, (void *)&pool.thr[i]);
    }

    wait_for_threads();
    assert(pool.active_threads == 0);
    pool.threads_initialized = 1;

    need_assign_pivot_threads = 1;

    DBG(1) fprintf(stderr, "< initPivotThreads()\n");
}

//@ Now the heart of the algorithm.  In each step of the
//iteration we choose a pivot element (|lpiv,rpiv|, whose actual indices
//are |rpivq|), and then pivot the table.

//To choose a pivot, we start by scanning the top row.  For any top-row
//entry $>$ that for the `best' candidate pivot we have (or $>0$), we
//find the maximum allowed increase, and accept the new pivot if it's
//better.  In case of degeneracies, we prefer to make artificial
//variables leave, otherwise make higher-index variables leave.

//If no pivot is found, we set |conv=true|.  But if a column we try
//gives no pivot, the function is unbounded.

/*==========================================================================*/
/* choosePivot                                                              */
/*==========================================================================*/

int32_t choose_pivot0(matrix_t *tabl, int32_t *left, int32_t *right, long L,
                      int32_t *lpiv0, int32_t *rpiv0, dble_t *piv0, dble_t *inc0, dble_t *coef0, int32_t *res0,
                      int start, int end)
{ 
    int32_t k, r;
    dble_t * restrict bcol = &tabl->data[0];

    int32_t res = NOPIVOT; 
    dble_t coef = 0,
           inc  = 0;

    int32_t rpivq = 0,
            lpiv  = 0,
            rpiv  = 0;

    dble_t piv = 0;

    DBG(3) fprintf(stderr, "> choosePivot()\n");

    for (r = start; r < end; r++)
    {
        const dble_t * restrict col = &tabl->data[r * tabl->rows + 0];

        DBG(2) fprintf(stderr, "r=%i\n", r);

        //fprintf(stderr, "rq=%i, r=%i, ivo=%f, coef=%f  %i\n", rq, r, ivo, coef,ivo>coef);

        if (col[0] <= coef) continue;
        //if (col[0] <= 0) continue;

        //----------------------------------------------------------------------
        // Assume we will find a pivot element.
        //----------------------------------------------------------------------
        res = FOUND_PIVOT;

        //----------------------------------------------------------------------
        // We now look for the column entry that causes the objective function
        // to increase the most.  Set |l,cleft,cpiv,cinc| for candidate pivot
        //----------------------------------------------------------------------
        int32_t l     = 0,
                cleft = 0; 
        dble_t cpiv  = 0,
               cinc  = 0;
        for (k=1; k <= L; k++) 
        { 
            //fprintf(stderr, "0\n");
            //if (col[k] == 0) continue;

            //===========================================================
            //===========================================================


            if (col[k] >= -SML) continue; /* only interested in negative values */

            const dble_t tinc = -bcol[k] * col[0]/col[k];
            //const dble_t tinc = -bcol[k] * col[0]/col[k];
#if 0
            DBG(1) fprintf(stderr, "col[0]=%f\n", col[0]);
            DBG(1) fprintf(stderr, "bcol[k=%i]=%f\n", k, bcol[k]);
            DBG(1) fprintf(stderr, "col[k=%i]=%f\n", k, col[k]);
            DBG(1) fprintf(stderr, "*** tinc=%f\n", tinc);
            DBG(1) fprintf(stderr, "*** cinc=%f\n", cinc);
            //fprintf(stderr, "*** tinc-cinc=%f\n", ABS(tinc-cinc));
            //fprintf(stderr, "*** tinc-cinc=%f\n", ABS(tinc-cinc));
            //fprintf(stderr, "*** tinc-cinc=%i\n", ABS(tinc-cinc)<EPS);
#endif

            DBG(1) assert(!isinf(tinc));

            //------------------------------------------------------------------
            // Accept this pivot element if we haven't found anything yet.
            //------------------------------------------------------------------
            int32_t accept = 0;
            if (l==0) 
            {
                accept = 1;
            }
#if 0
            else if (ABS(tinc-cinc) < EPS) 
            { 
                if (left[k] > 0 && cleft > 0)  
                    accept = (left[k] > cleft) + 2;
                else 
                    accept = (left[k] < cleft) + 4;
            }
#endif
            else 
            {
                accept = (tinc < cinc) + 8;
            }

            if ((accept&1)==0) continue; // && tinc > 0)
            //assert(tinc > 0);

            DBG(2) fprintf(stderr, "ACCEPTING tinc=%.15e %i\n", tinc, accept);

            //------------------------------------------------------------------
            // Remeber this pivot element for later.
            //------------------------------------------------------------------
            l = k; 
            cleft = left[k]; 
            cpiv  = col[k]; 
            cinc  = tinc;
        }

        //----------------------------------------------------------------------
        // Maybe update |lpiv,rpiv,rpivq,piv,inc,coef|
        //----------------------------------------------------------------------
        if (l == 0) 
        {
            lpiv = -1;
            rpiv = -1;
            piv  = -1;
            res = UNBOUNDED;
            break;
        }

        //fprintf(stderr, "* r=%i lpiv=%i cinc=%f inc=%f right[r]=%i rpivq=%i\n", r,lpiv,cinc, inc, right[r], rpivq);
        int32_t accept = 0;
        if (lpiv==0)
            accept = 1;
        else if (ABS(cinc-inc) < EPS)
            accept = (right[r] < rpivq);
        else  
            accept = (cinc > inc);

        if (!accept) continue;

        lpiv  = l;    
        rpiv  = r; 

        rpivq = right[r];

        piv = cpiv; 
        inc = cinc; 
        
        coef = col[0];

        //break; // Bland's Rule: Take the first one you find.
    }

    DBG(2) fprintf(stderr, "< choosePivot() %i %i %e %e\n", lpiv, rpiv, piv, inc);

    *lpiv0 = lpiv;
    *rpiv0 = rpiv;
    *piv0  =  piv;
    *inc0  =  inc;
    *coef0 = coef;
    *res0  = res;
    return res;
}

void choose_pivot(pivot_thread_t *thr)
{
    if (thr->start == thr->end) return;

    choose_pivot0(thr->tabl,
                  thr->left,
                  thr->right,
                  thr->L,
                 &thr->lpiv,
                 &thr->rpiv,
                 &thr->piv,
                 &thr->inc,
                 &thr->coef,
                 &thr->res,
                  thr->start,
                  thr->end);
}

int32_t select_pivot(matrix_t *tabl, int32_t *left, int32_t *right, long L, long R,
                     int32_t *lpiv0, int32_t *rpiv0, dble_t *piv0)
{
#if 1

    int32_t start = pool.thr[0].start,
            end   = pool.thr[0].end;

    pool.thr[0].start = 1;
    pool.thr[0].end   = R+1;
    pool.thr[0].tabl   = tabl;
    pool.thr[0].left   = left;
    pool.thr[0].right  = right;
    pool.thr[0].L      = L;
    pool.thr[0].piv    = 0;
    pool.thr[0].lpiv   = 0;
    pool.thr[0].rpiv   = 0;
    pool.thr[0].inc    = 0;
    pool.thr[0].coef   = 0;
    pool.thr[0].res    = NOPIVOT;
    choose_pivot(&pool.thr[0]);

    *lpiv0 = pool.thr[0].lpiv;    
    *rpiv0 = pool.thr[0].rpiv; 
    *piv0  = pool.thr[0].piv; 

    pool.thr[0].start = start;
    pool.thr[0].end   = end;

    return pool.thr[0].res;
#else

    int i;
    for (i=0; i < pool.nthreads; i++)
    {
        pool.thr[i].tabl   = tabl;
        pool.thr[i].left   = left;
        pool.thr[i].right  = right;
        pool.thr[i].L      = L;
        pool.thr[i].piv    = 0;
        pool.thr[i].lpiv   = 0;
        pool.thr[i].rpiv   = 0;
        pool.thr[i].inc    = 0;
        pool.thr[i].coef   = 0;
        pool.thr[i].res    = NOPIVOT;
        pool.thr[i].action = choose_pivot;
    }

    startup_threads();
    pool.thr[0].start += 1;
    choose_pivot(&pool.thr[0]);
    pool.thr[0].start -= 1;
    wait_for_threads();

    int32_t rpivq = 0,
            lpiv  = 0,
            rpiv  = 0;

    dble_t piv = 0, inc=0, coef=0;
    int32_t res = NOPIVOT;

    for (i=0; i < pool.nthreads; i++)
    {
        //fprintf(stderr, "thr %i start=%i end=%i coef=%f\n", pool.thr[i].id, pool.thr[i].start, pool.thr[i].end, pool.thr[i].coef);

        if (pool.thr[i].lpiv == 0) continue;
        if (pool.thr[i].lpiv == -1) return UNBOUNDED;

        if (pool.thr[i].coef <= coef) continue;

        res = FOUND_PIVOT;

        int32_t accept = 0;

        if (lpiv == 0)
            accept = 1;
        else if (ABS(pool.thr[i].inc-inc) < EPS)
            accept = (pool.thr[i].right[pool.thr[i].rpiv] < rpivq);
        else  
            accept = (pool.thr[i].inc > inc);

        if (!accept) continue;

        lpiv  = pool.thr[i].lpiv;    
        rpiv  = pool.thr[i].rpiv; 

        rpivq = pool.thr[i].right[pool.thr[i].rpiv];

        piv  = pool.thr[i].piv; 
        inc  = pool.thr[i].inc; 
        coef = pool.thr[i].coef;
    }

    //fprintf(stderr, "* %f\n", coef);

    *lpiv0 = lpiv;
    *rpiv0 = rpiv;
    *piv0  =  piv;

    return res;
#endif

}

void assign_threads(int32_t lo, int32_t hi)
{
    int32_t i,n;
    const int32_t ncols = (int32_t)ceil((double)(hi+1 - lo) / pool.nthreads);

    for (i=0; i < pool.total_threads; i++) 
        pivot_thread_reset(pool.thr+i, lo,lo);

    for (i=lo,n=0; i < hi+1; i += ncols, n++)
    {
        pivot_thread_reset(pool.thr+n, 
                           i, 
                           min(i + ncols, hi+1));
        DBG(1) fprintf(stderr, ">> assigned %i to (%i,%i) lo=%i hi=%i\n", n, pool.thr[n].start, pool.thr[n].end, lo, hi);
    }

    DBG(1) fprintf(stderr, "> assign_threads() %i %i\n", ncols, pool.nthreads);

    need_assign_pivot_threads = 0;
}

char buf[100];
void periodic_report(int sig)
{
    assert(sig == SIGALRM);
    sprintf(buf, "\riter %8i  %24.15e [%i]", report.step, report.obj_val, report.nthreads);
    write(STDERR_FILENO, buf, sizeof(buf));
    fsync(STDERR_FILENO);
    alarm(1);
}


/*==========================================================================*/
/* pivot                                                                    */
/*==========================================================================*/
PyObject *samplex_pivot(PyObject *self, PyObject *args)
{
    int32_t i,j;
    matrix_t tabl;

    PyObject *o = args;
    DBG(3) fprintf(stderr, "5> pivot()\n");

    long L = PyInt_AsLong(PyObject_GetAttrString(o, "nLeft")); /* # of constraints (rows)    */
    long R = PyInt_AsLong(PyObject_GetAttrString(o, "nRight")); /* # of variables   (columns) */
    long Z = PyInt_AsLong(PyObject_GetAttrString(o, "nTemp"));

    long T = PyInt_AsLong(PyObject_GetAttrString(o, "nthreads"));

#if 0
    fprintf(stderr, "%ld %ld // %ld %ld\n", PyArray_DIM(data,0), PyArray_DIM(data,1),
                                            PyArray_DIM(orig,0), PyArray_DIM(orig,1));
#endif

    PyObject *lhv = PyObject_GetAttrString(o, "lhv");
    PyObject *rhv = PyObject_GetAttrString(o, "rhv");

    int32_t *left  = (int32_t *)PyArray_DATA(lhv), 
            *right = (int32_t *)PyArray_DATA(rhv);


#if 0
    fprintf(stderr, "tabl dims = %i : %ix%i\n"
                    "L=%i R=%i Z=%i\n"
                    "len(left)=%i, len(right)=%i\n"
                    "nthreads=%i\n", 
                    PyArray_NDIM(data), tabl.cols, tabl.rows,
                    L, R, Z,
                    PyArray_DIM(lhv,0),
                    PyArray_DIM(rhv,0),
                    T
                    );
#endif

    int32_t lpiv,    /* Pivot row              */
            rpiv;    /* Pivot column           */
    dble_t piv;     /* Value of pivot element */

#if 0
        col = &(tabl.data[0 * tabl.rows + 0]);
        for (i=0; i <= L; i++)
            fprintf(stderr, "%.4f ", col[i]);
        fprintf(stderr, "\n");
#endif

    int32_t ret, n;

    long Zorig = Z;


    init_threads(T); 
    need_assign_pivot_threads = 1;

    /* Remember, this is in FORTRAN order */
#if REORGANIZE_TABLE_MEMORY
    import_array();
    PyObject *orig = PyObject_GetAttrString(o, "data");
    assert(PyArray_CHKFLAGS(orig, NPY_F_CONTIGUOUS));
    assert(PyArray_CHKFLAGS(orig, NPY_FORTRAN));
    //PyObject *data = PyArray_EMPTY(2, PyArray_DIMS(orig), NPY_DOUBLE, 1);

    tabl.orig = (dble_t *)PyArray_DATA(orig);
    tabl.rows = PyArray_DIM(orig,0);
    tabl.cols = PyArray_DIM(orig,1);

    tabl.data = malloc(tabl.rows*tabl.cols*sizeof(*tabl.data));

    PyObject *data = PyArray_NewFromDescr(&PyArray_Type, 
                                          PyArray_DESCR(orig), 
                                          2,
                                          PyArray_DIMS(orig), 
                                          PyArray_STRIDES(orig), 
                                          tabl.data,
                                          PyArray_FLAGS(orig),
                                          NULL);

    assign_threads(0,R); 
    for (i=0; i < pool.nthreads; i++)
    {
        pool.thr[i].tabl      = &tabl;
        pool.thr[i].L         = L;
        pool.thr[i].piv       = 0;
        pool.thr[i].lpiv      = 0;
        pool.thr[i].rpiv      = 0;
        pool.thr[i].action    = copymem;
    }

    startup_threads();
    copymem(&pool.thr[0]);
    wait_for_threads();
#else
    PyObject *data = PyObject_GetAttrString(o, "data");
    assert(PyArray_CHKFLAGS(data, NPY_F_CONTIGUOUS));
    assert(PyArray_CHKFLAGS(data, NPY_FORTRAN));
    tabl.data = (dble_t *)PyArray_DATA(data);
    tabl.rows = PyArray_DIM(data,0);
    tabl.cols = PyArray_DIM(data,1);
#endif

    tabl.pcol = malloc(tabl.rows * sizeof(*(tabl.data)));

#if WITH_GOOGLE_PROFILER
    ProfilerStart("googperf.out");
#endif

    double stime = CPUTIME;
    double now = stime;
    double bestperftime = 0;
    int bestnthreads = 0;

    struct timespec t;

    int searchdir = -1;

    signal(SIGALRM, periodic_report);
    alarm(1);

    Py_BEGIN_ALLOW_THREADS

    for (n=0;; n++)
    {
        report.step     = n;
        report.obj_val  = tabl.data[0];
        report.nthreads = pool.nthreads;

#if 0
        if ((n&((1<<5)-1)) == 0) 
        {
            fprintf(stderr, "\riter %8i  %24.15e [%i]", n, tabl.data[0], pool.nthreads);

#if 0
            double then = now;
            now = CPUTIME;
            double perftime = now - then;

            if (!bestperftime) bestperftime = perftime;
            double r = perftime / bestperftime;

            //------------------------------------------------------------------
            // If performance was 20% poorer, reverse direction.
            //------------------------------------------------------------------
            if (r > 1.2) searchdir *= -1;

            //------------------------------------------------------------------
            // Increment or decrement the number of threads according to the
            // search direction.
            //------------------------------------------------------------------
            if (searchdir == -1)
            {
                if (pool.nthreads < pool.total_threads) 
                {
                    bestperftime = perftime;

                    pool.nthreads++;
                    need_assign_pivot_threads = 1;
                }
            }
            else
            {
                if (pool.nthreads > 1) 
                {
                    pool.nthreads--;
                    need_assign_pivot_threads = 1;
                }
            }


            //------------------------------------------------------------------
            // 15% of the time we will forget what we know and begin searching
            // again. Helps getting stuck on a "best time" that applied to
            // a different system load.
            //------------------------------------------------------------------
            if (drand48() < .15)
            {
                bestperftime = 0;
                searchdir = 1;
            }
#endif
        }
#endif

        //if (n == 5) exit(0);

        if (need_assign_pivot_threads) assign_threads(0,R); 

        ret = select_pivot(&tabl, left, right, L, R, &lpiv, &rpiv, &piv);

        if (ret != FOUND_PIVOT) break;

        //------------------------------------------------------------------
        // Actual pivot
        //------------------------------------------------------------------
        //doPivot(&tabl, L, piv, lpiv, rpiv, 0, R+1);
        //==================================================================
        // Let the threads know they need to perform the pivot
        //
        // If there's only one thread, then just do the operation in the 
        // current thread instead of using other worker threads.
        //
        // In any event, we do at least one pivot operation in this thread.
        //==================================================================

        memcpy(tabl.pcol, tabl.data + (rpiv*tabl.rows), tabl.rows * sizeof(*(tabl.data)));

        for (i=0; i < pool.nthreads; i++)
        {
            pool.thr[i].tabl   = &tabl;
            pool.thr[i].L      = L;
            pool.thr[i].piv    = piv;
            pool.thr[i].lpiv   = lpiv;
            pool.thr[i].rpiv   = rpiv;
            pool.thr[i].action = doPivot;
        }

        startup_threads();
        doPivot(&pool.thr[0]);
        wait_for_threads();


        //----------------------------------------------------------------------
        // Swap left and right variables.
        //----------------------------------------------------------------------
        int32_t lq = left[lpiv];
        int32_t rq = right[rpiv];

        left[lpiv]  = rq;
        right[rpiv] = lq;

        if (lq < 0)
        { 
            //------------------------------------------------------------------
            // Remove the pivot column
            //------------------------------------------------------------------
            memmove(right+rpiv+0, 
                    right+rpiv+1, 
                    sizeof(*right)*(R-rpiv)); /* (R+1)-(rpiv+1) */

            memmove(tabl.data + (rpiv+0)*tabl.rows, 
                    tabl.data + (rpiv+1)*tabl.rows,
                    sizeof(*tabl.data) * (R-rpiv)*tabl.rows);

            Z--; 
            R--;
            need_assign_pivot_threads = 1;
            DBG(1) fprintf(stderr, "\nREMOVED %i Z=%ld R=%ld\n\n", rpiv, Z,R);
        }


        DBG(2)
        {
            for (i=0; i <= R; i++)
            {
                dble_t *col = &(tabl.data[i * tabl.rows + 0]);
                for (j=0; j <= L; j++)
                    if (ABS(ABS(col[j]) - 22.00601215427127) < 1e-3)
                    {
                        fprintf(stderr, "+++++ i=%i j=%i  col[j]=%f\n", i, j, col[j]);
                        //assert(0);
                    }

                    //fprintf(stderr, "%.4f ", col[j]);
                //fprintf(stderr, "\n");
            }
        }

        if (Z==0) 
        {
            ret = FEASIBLE;
            if (Zorig != 0) break;
        }
    }

    //fprintf(stderr, "\n");

    double etime = CPUTIME;
    //clock_t et = times(NULL);

#if WITH_GOOGLE_PROFILER
    ProfilerStop();
#endif

    //fprintf(stderr, "time: %f\n", (etime-stime));
            //fprintf(stderr, "\riter %8i  %24.15e", n, tabl.data[0]);
    fprintf(stderr, "\rtime: %4.2f CPU seconds. %39c\n", (etime-stime), ' ');
    //fprintf(stderr, "time: %f\n", (etime-stime) / pool.nthreads);

    free(tabl.pcol);

    Py_END_ALLOW_THREADS

    alarm(0);
    signal(SIGALRM, SIG_DFL);

    PyObject_SetAttrString(o, "nRight", PyInt_FromLong(R));
    PyObject_SetAttrString(o, "nTemp", PyInt_FromLong(Z));
#if REORGANIZE_TABLE_MEMORY
    PyObject_SetAttrString(o, "data", data);
    //Py_XDECREF(orig);
#endif

    return PyInt_FromLong(ret);
}

/*==========================================================================*/
/* doPivot                                                                  */
/*                                                                          */
/* Now the code to do the pivoting.  An artificial variable that leaves     */
/* is removed.  If we are removing the last of these, we set |conv=true|.   */
/*==========================================================================*/
void doPivot(pivot_thread_t *thr)
{
    if (thr->start == thr->end) return;


#if REORGANIZE_TABLE_MEMORY
    if (thr->pcol == NULL)
        thr->pcol = malloc(thr->tabl->rows * sizeof(*(thr->tabl->data)));

    memcpy(thr->pcol, thr->tabl->pcol, thr->tabl->rows * sizeof(*(thr->tabl->data)));
    const dble_t * restrict pcol = thr->pcol;
#else
    const dble_t * restrict pcol = thr->tabl->pcol;
#endif


    doPivot0(thr->tabl, 
             pcol,
             thr->L, 
             thr->piv, 
             thr->lpiv, 
             thr->rpiv, 
             thr->start, 
             thr->end);
}

void copymem(pivot_thread_t *thr)
{
    DBG(1) fprintf(stderr, "%i copy segment %i to %i\n", thr->id, thr->start, thr->end);
    memcpy(thr->tabl->data + thr->start * thr->tabl->rows, 
           thr->tabl->orig + thr->start * thr->tabl->rows,
           (thr->end - thr->start) * thr->tabl->rows * sizeof(*(thr->tabl->data)));
}

//------------------------------------------------------------------------------
// The first column must always remain positive (except for the value of the
// objective function. We special-case this column to print error messages if
// we find a negative value.
//------------------------------------------------------------------------------
#if 0
inline void in0(const int32_t r, 
                int32_t kp, 
                const int32_t lpiv0, 
                const dble_t piv, 
                dble_t * restrict col, 
                const dble_t * restrict pcol)
{
    const dble_t col_lpiv = col[lpiv0];

    //fprintf(stderr, "in %i\n", r);

    /* Everything but the objective function value */
    for (; kp >= 1; kp--) 
    {
        dble_t v = col[kp] - (pcol[kp] * col_lpiv) / piv;

        //dble_t v = col[kp] - pcol[kp] * col_lpiv;
        //dble_t v = fdim(col[kp], pcol[kp] * col_lpiv);

        if (kp != lpiv0 && rtz(v) < 0) 
        {
            fprintf(stderr, "**** \n lpiv=%i pcol[%i]=%.15e "
                            "col_lpiv=%.15e col[%i]=%.15e v=%.15e\n", 
                            lpiv0, kp, pcol[kp], col_lpiv, kp, col[kp], v);
            assert(0);
        }

        // TODO: Since we are adding noise to zero values when the constraints
        // are first added to the tableau, perhaps we should also add noise here
        // so that we avoid later degeneracies in choose_pivot.
        col[kp] = rtz(v);
    }

    /* Objective function value */
    col[0] = rtz(col[0] - (pcol[0] * col_lpiv) / piv);
    //col[0] = rtz(col[0] - pcol[0] * col_lpiv);

    col[lpiv0] = rtz(col_lpiv / -piv);
    if (col[lpiv0] < 0)
    {
        fprintf(stderr, "r=%i lpiv0=%i col_lpiv=%.15e piv=%.15f\n", r, lpiv0, col_lpiv, piv);
        assert(0);
    }
}
#endif

#if 0
inline void in(const int32_t r, 
               int32_t kp, 
               const int32_t lpiv0, 
               const dble_t piv, 
               dble_t * restrict col, 
               const dble_t * restrict pcol)
{
    const dble_t col_lpiv = col[lpiv0];
    int32_t i;

    const double xx = col_lpiv / piv;
    //fprintf(stderr, "%e %e %e\n", col_lpiv, piv, xx);

    if (xx != 0)
    {
        if (ABS(xx) >= SML)
        {
            for (i=0; i <= kp; i++) 
                col[i] -= pcol[i] * xx;
        }
        else
        {
            for (i=0; i <= kp; i++) 
                col[i] -= (pcol[i] * col_lpiv) / piv;
        }
    }
    col[lpiv0] = -xx;
}
#endif

#define IN \
do { \
    int32_t i;     \
    dble_t * restrict col = tabl->data + (r * tabl->rows); \
    const dble_t col_lpiv = col[lpiv];     \
    const dble_t xx = col_lpiv / piv;  \
    {  \
        if (ABS(xx) >= SML)    \
        {\
            for (i=0; i <= L; i++) \
                col[i] -= pcol[i] * xx;  \
        }\
        else   \
        {\
            for (i=0; i <= L; i++) \
                col[i] -= (pcol[i] * col_lpiv) / piv;   \
        }\
    }  \
    col[lpiv] = -xx;   \
} while(0);

#define XIN \
do { \
    int32_t i;     \
    dble_t * restrict col = tabl->data + (r * tabl->rows); \
    dble_t * restrict col1 = col; \
    dble_t * restrict pcol1 = pcol; \
    const dble_t col_lpiv = col[lpiv];     \
    const dble_t xx = col_lpiv / piv;  \
    {  \
        if (ABS(xx) >= SML)    \
        {\
            for (i=0; i <= L; i++) \
                *col1++ -= *pcol1++ * xx;  \
        }\
        else   \
        {\
            for (i=0; i <= L; i++) \
                *col1++ -= (*pcol1++ * col_lpiv) / piv;   \
        }\
    }  \
    col[lpiv] = -xx;   \
} while(0);

void doPivot0(
    matrix_t * restrict tabl,
    dble_t * restrict pcol,
    const long L,
    const dble_t piv, 
    const int32_t lpiv, const int32_t rpiv,
    int32_t start, const int32_t end)
{

    //DBG(3) fprintf(stderr, "> doPivot()\n");
    //DBG(3) fprintf(stderr, "doPivot called: %i %i L=%ld R=%ld rpiv=%i lpiv=%i\n", start, end, L, R, rpiv, lpiv);

    int32_t r=start;
    int32_t i;

    if (start <= rpiv && rpiv < end)
    {
        for (r=start; r < rpiv; ++r) IN;

        dble_t * restrict pcol0 = tabl->data + (rpiv * tabl->rows);
        for (i=0; i <= L; i++)
            pcol0[i] /= piv;
        pcol0[lpiv] = 1.0 / piv;

        for (    ++r; r < end;  ++r) IN;
    }
    else
    {
        for (r=start; r < end; ++r)
            do { 
                int32_t i;     
                dble_t * restrict col = tabl->data + (r * tabl->rows); 
                const dble_t col_lpiv = col[lpiv];     
                const dble_t xx = col_lpiv / piv;  
                {  
                    if (ABS(xx) >= SML)    
                    {
                        for (i=0; i <= L; i++) 
                            col[i] -= pcol[i] * xx;  
                    }
                    else   
                    {
                        for (i=0; i <= L; i++) 
                            col[i] -= (pcol[i] * col_lpiv) / piv;   
                    }
                }  
                col[lpiv] = -xx;   
            } while(0);
    }

    DBG(2)
    {
        int32_t i,j;
        for (i=1; i <= L; i++)
        {
            double v0 = tabl->data[0 * tabl->rows + i];
            if (v0 == 0)
            {
                for (j=1; j < end; j++)
                {
                    double v1 = tabl->data[j * tabl->rows + i];
                    if (v1 == 0)
                    {
                        fprintf(stderr, "ACK %i %i\n", i, j);
                        assert(0);
                    }
                }
            }
        }
    }

    //free(pcol);

    //DBG(3) fprintf(stderr, "< doPivot()\n");
}

