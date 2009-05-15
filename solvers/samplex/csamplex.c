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
#define REORGANIZE_TABLE_MEMORY 1
#define SET_THREAD_AFFINITY     0

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
typedef double dble_t;
#define EPS ((dble_t)1e-14)
#define INF ((dble_t)1e+12) 
#define SML ((dble_t)1e-06)
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

inline int min(int a, int b)
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
    int *data;
    size_t size;
    size_t space;
} int_array;

typedef struct
{
    size_t size;
    size_t space;
    size_t H;
    dble_t **data;
} dd_array;

typedef struct
{
    int w,h;
    dble_t *data;
} matrix_t;


typedef struct pivot_thread_s
{
    pthread_t thr_id;

    int id;
    size_t start, end;

    matrix_t *tabl;
    int L;
    dble_t piv;
    int lpiv, rpiv;

    void (*action)(struct pivot_thread_s *thr);

} pivot_thread_t;

typedef struct 
{
    int nthreads;               // Total number of threads
    int active_threads;         // Number of threads currently executing doPivot
    int threads_initialized;
    pivot_thread_t *thr;
    pthread_mutex_t lock;
    pthread_cond_t activate_all;
    pthread_cond_t thread_finished;
} thread_pool_t;
#define EMPTY_POOL {0,0,0,NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER}


/*==========================================================================*/
/* The worker threads can be asked to do different things.                  */
/* They can either do the normal pivot operation, or they can capture       */
/* a portion of the table memory by allocating memory local to their CPU.   */
/*==========================================================================*/
#define PT_ACTION_STEAL_TABLE_COLS  (1)
#define PT_ACTION_DO_PIVOT          (2)

/*==========================================================================*/
/* Non-JNI function prototypes.                                             */
/*==========================================================================*/
#if 0

void throw_errormsg(JNIEnv *env, const char *msg);
static void setObjective(const dble_t *__restrict  obj);
static void ran_set_seed(long seed);
static int ran_next(int bits);
static jdouble ran_next_double();
void assign_threads();
inline void in(const size_t r, 
               ssize_t kp, 
               const size_t lpiv0, 
               dble_t *__restrict col, 
               const dble_t *__restrict pcol) __attribute__ ((always_inline));
void steal_table_memory(size_t start, size_t end);

void doPivot0(const jint start, const jint end);
void startPivot();
int finishPivot();
jint choosePivot();

/*==========================================================================*/
/* Global variables                                                         */
/*==========================================================================*/
/* Number of samples taken from simplex */
static int samp=0;              
static dble_t sum_ln_k=0;

/* Random number generator seed */
static long ran_seed=0;    
#endif

/* The pivot threads */
static thread_pool_t pool = EMPTY_POOL;

/* Do we need to update the columns that each thread is working on? */
static int need_assign_pivot_threads;

#if 0
static double start_time, end_time, avg_time = 0;
#endif

PyObject *samplex_pivot(PyObject *self, PyObject *args);
void doPivot(pivot_thread_t *thr);
void doPivot0(matrix_t *tabl,
    const int L,
    const dble_t piv, 
    const int lpiv, const int rpiv,
    int start, const int end);
//int choosePivot(dble_t **tabl, size_t L, size_t R, size_t *lpiv0, size_t *rpiv0, size_t *piv0);

static PyMethodDef csamplex_methods[] = 
{
    {"pivot", samplex_pivot, METH_O, "pivot"},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC initcsamplex()
{
    (void)Py_InitModule("csamplex", csamplex_methods);
}

#if 0

/*==========================================================================*/
/* Helper function for throwing Java exceptions                             */
/*==========================================================================*/
void throw_errormsg(const char *msg)
{
    //jclass clazz = (*env)->FindClass(env, "_42util/ErrorMsg");
    //if (clazz != NULL) (*env)->ThrowNew(env, clazz, msg);
    //(*env)->DeleteLocalRef(env, clazz);
}


/* reproduce Java random number generator */
static void ran_set_seed(long seed)
{
    ran_seed = (seed ^ 0x5DEECE66DL) & ((1L << 48) - 1);
}

static inline long triple_shift(long n, long s)
{
    return (n >> s) + (2 << ~s);
}

static int ran_next(int bits)
{
    ran_seed = (ran_seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    return (int)triple_shift(ran_seed, 48 - bits);
    //return (long)(ran_seed >> (48 - bits));
}

static jdouble ran_next_double()
{
    return (((long)ran_next(26) << 27) + ran_next(27)) / (jdouble)(1L << 53);
}

inline static size_t next_power_of_2(size_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v += (v == 0);
    v++;
    return v;
}

#endif

/*==========================================================================*/
/* PivotThread functions                                                    */
/*==========================================================================*/

#if 1

void pivot_thread_init(pivot_thread_t *pt, int id)
{
    pt->thr_id   = 0;
    pt->id       = id;
    pt->start    = 
    pt->end      = 0;
    pt->action   = doPivot;
}

void pivot_thread_reset(pivot_thread_t *pt, size_t start, size_t end)
{ 
    pt->start = start;
    pt->end   = end;
}

void *pivot_thread_run(void *arg)
{
    pivot_thread_t *pt = (pivot_thread_t *)arg;

#if SET_THREAD_AFFINITY
    cpu_set_t mask;

    int ncpus      = sysconf(_SC_NPROCESSORS_CONF);
    int cpu_to_use = pt->id % ncpus;

    /* Try to fix the thread to a single CPU */

    CPU_ZERO(&mask);
    CPU_SET(cpu_to_use, &mask);

    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    {
        int e = errno;
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
    pool.active_threads = pool.nthreads-1;
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

#endif

/*==========================================================================*/
/* extendColumns                                                            */
/*==========================================================================*/
#if 0
static void extendColumns(size_t l) 
{
    int n;
    size_t oldH = tabl.H;
    DBG(3) fprintf(stderr, "> extendColumns()\n");
    DBG(3) fprintf(stderr, "l=%ld oldH=%ld\n", l, oldH);

    if (l < tabl.H) return;

    while (tabl.H <= l)
        tabl.H *= 2;

    DBG(3) fprintf(stderr, "l=%ld oldH=%ld new tabl.H=%ld\n", l, oldH, tabl.H);

    for (n=0; n < tabl.size; n++) 
    {
        dble_t *col = tabl.data[n]; assert(col != NULL);
        dble_t *tmp = CALLOC(dble_t, tabl.H); assert(tmp != NULL);
        memmove(tmp, col, oldH * sizeof(dble_t));
        tabl.data[n] = tmp;
        free(col);
    }


    DBG(3) fprintf(stderr, "< extendColumns() tabl.H=%ld\n", tabl.H);
}

void steal_table_memory(size_t start, size_t end)
{
    size_t X = tabl.H; 
    int i, ncols = end- start;
    DBG(3) fprintf(stderr, "> steal_table_memory()\n");

    DBG(3) fprintf(stderr, "N=%ld L=%ld R=%ld Z=%ld S=%ld H=%ld X=%ld\n",
        N, L, R, Z, S, tabl.H, X);

    dble_t *cols = MALLOC(dble_t, X * ncols); assert(cols != NULL);

    dble_t *col = cols;
    for (i=start; i < end; i++)
    {
        memcpy(col, tabl.data[i], X * sizeof(dble_t));
        free(tabl.data[i]);
        tabl.data[i] = col;
        col += X;
    }

    DBG(3) fprintf(stderr, "< steal_table_memory()\n");
}

void reorganize_table_memory()
{
#if REORGANIZE_TABLE_MEMORY
    int i;

    DBG(3) fprintf(stderr, "> reorganize_table_memory()\n");

    if (need_assign_pivot_threads) assign_threads(); 

    /* Steal the table for the main thread */
    steal_table_memory(pt[0].start, pt[0].end);

    /* Now ask the worker threads to steal the table */
    for (i=1; i < nthreads; i++)
        pt[i].action = PT_ACTION_STEAL_TABLE_COLS;

    startup_threads();
    wait_for_threads();
    
    for (i=1; i < nthreads; i++)
        pt[i].action = PT_ACTION_DO_PIVOT;

    DBG(3) fprintf(stderr, "< reorganize_table_memory()\n");
#endif
}
#endif

#if 0

/*==========================================================================*/
/* init0                                                                    */
/*==========================================================================*/
void init0(JNIEnv *env, jobject obj, jint N0) 
{
    DBG(3) fprintf(stderr, "> init0()\n");

    //DBG(1) fprintf(stderr, "Simplex data type is %ld bytes wide.\n", sizeof(dble_t));
    //DBG(1) fprintf(stderr, "double is %ld bytes wide.\n", sizeof(double));
    //DBG(1) fprintf(stderr, "long double is %ld bytes wide.\n", sizeof(long double));
    //DBG(1) fprintf(stderr, "SML is %ld bytes wide.\n", sizeof(SML));
    //TEST(dble_t);
    //assert(0);

    //DBG(2) { int e; frexp(EPS, &e); fprintf(stderr, "EPS=%Lg frexp = %i\n", EPS, e); }
    
    ran_set_seed(0);
    //srand(time(NULL));


#if 0
    int i=0;
    FILE *fp = fopen("bad.ran", "w");
    for (i=0; i < 1000; i++)
        fprintf(fp, "%.16f\n", ran_next_double());
    fclose(fp);
    assert(0);
#endif

    N = N0;  
    curr = prev = NULL; 
    hist = NULL;
    objective = NULL;

    sum_ln_k = 0;
    samp = 0;

    R = L = S = Z = 0; 


    DBG(3) fprintf(stderr, "< init0()\n");
}

#endif

/*==========================================================================*/
/* initPivotThreads                                                         */
/*==========================================================================*/
void init_threads(int n) 
{
    size_t i;

    if (pool.threads_initialized) return;

    DBG(1) fprintf(stderr, "> initPivotThreads() nthreads=%i\n", n);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);

    pool.nthreads       = n;
    pool.active_threads = 0;

#if SET_THREAD_AFFINITY
    if (nthreads > 1)
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
    pool.thr = CALLOC(pivot_thread_t, pool.nthreads); assert(pool.thr != NULL);

    //--------------------------------------------------------------------------
    // Notice we start from 1 not 0, because we handle the first thread
    // specially later on. 
    //--------------------------------------------------------------------------
    pool.active_threads = pool.nthreads-1;
    for (i=1; i < pool.nthreads; i++)
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

#if 0

/*==========================================================================*/
/* finalize                                                                 */
/*==========================================================================*/
void finalize(JNIEnv *env, jobject obj)
{
    int i;

    DBG(3) fprintf(stderr, "> finalize()\n");

#if REORGANIZE_TABLE_MEMORY
    if (Z==0)
    {
        if (tabl.data != NULL) 
        {
            for (i=0; i < nthreads; i++)
                free(tabl.data[pt[i].start]);
            free(tabl.data);
            tabl.data = NULL;
        }
    }
    else
    {
        dd_array_free(&tabl);
    }
#else
    dd_array_free(&tabl);
#endif

    int_array_free(&left);
    int_array_free(&right);

    if (objective != NULL) free(objective);
    if (curr != NULL) free(curr);
    if (prev != NULL) free(prev);
    if (hist != NULL) free(hist);

    for (i=1; i < nthreads; i++)
    {
        pthread_cancel(pt[i].thr_id);
        pthread_join(pt[i].thr_id, NULL);
    }

    if (pt != NULL) free(pt);

    DBG(3) fprintf(stderr, "< finalize()\n");
}

#endif

//@ Now the heart of the algorithm.  In each step of the
//iteration we choose a pivot element (|lpiv,rpiv|, whose actual indices
//are |lpivq,rpivq|), and then pivot the table.

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
int choose_pivot(matrix_t *tabl, int *left, int *right, size_t L, size_t R, 
                       size_t *lpiv0, size_t *rpiv0, dble_t *piv0)
{ 
    size_t k, r;
    dble_t *__restrict  bcol = &tabl->data[0];
    int res = NOPIVOT; 
    dble_t coef=0,inc=0;

    size_t lpivq = 0, 
           rpivq = 0,
           lpiv  = 0,
           rpiv  = 0;

    dble_t piv = 0;

    DBG(3) fprintf(stderr, "> choosePivot()\n");

    for (r=1; r<=R; r++)
    {
        const dble_t * __restrict  col = &tabl->data[r * tabl->w + 0];

        DBG(2) fprintf(stderr, "r=%ld\n", r);

        //fprintf(stderr, "rq=%i, r=%i, R=%i, ivo=%f, coef=%f  %i\n", rq, r, R, ivo, coef,ivo>coef);

        if (col[0] <= coef) continue;

        //----------------------------------------------------------------------
        // Assume we will find a pivot element.
        //----------------------------------------------------------------------
        res = FOUND_PIVOT;

        //----------------------------------------------------------------------
        // We now look for the column entry that causes the objective function
        // to increase the most.  Set |l,cleft,cpiv,cinc| for candidate pivot
        //----------------------------------------------------------------------
        size_t l     = 0,
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
            assert(!isinf(tinc));

            //------------------------------------------------------------------
            // Accept this pivot element if we haven't found anything yet.
            //------------------------------------------------------------------
            int accept = 0;
            if (l==0) 
            {
                accept = 1;
            }
            else if (ABS(tinc-cinc) < EPS) 
            { 
                if (left[k] > 0 && cleft > 0)  
                    accept = (left[k] > cleft) + 2;
                else 
                    accept = (left[k] < cleft) + 4;
            }
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
        // Maybe update |lpiv,rpiv,lpivq,rpivq,piv,inc,coef|
        //----------------------------------------------------------------------
        if (l == 0) 
        {
            fprintf(stderr, "UNBOUNDED!!!!!!!!!\n");
            return UNBOUNDED;
        }

        //fprintf(stderr, "\n************* lpiv=%i\n", lpiv);
        int accept = 0;
        if (lpiv==0)
            accept = 1;
        else if (ABS(cinc-inc) < EPS)
            accept = (right[r] < rpivq);
        else  
            accept = (cinc > inc);

        if (!accept) continue;

        //fprintf(stderr, "\n!!!!!!!!!!!!!!! cpiv=%f\n", cpiv);
        lpiv  = l;    
        rpiv  = r; 

        lpivq = cleft; 
        rpivq = right[r];

        piv = cpiv; 
        inc = cinc; 
        
        coef = col[0];

        //break; // Bland's Rule: Take the first one you find.
    }

    DBG(1) fprintf(stderr, "< choosePivot() %ld %ld %e %e\n", lpiv, rpiv, piv, inc);
    //assert(piv <  1e4);
    //assert(piv > -1e4);
    //DBG(3) fprintf(stderr, "< choosePivot() %ld %ld %23.15Lf\n", lpiv, rpiv, (long double)piv);

    *lpiv0 = lpiv;
    *rpiv0 = rpiv;
    *piv0  =  piv;
    return res;
}

void assign_threads(int R)
{
    int i,n;
    const int ncols = (int)ceil((double)(R+1) / pool.nthreads);

    for (i=1; i < pool.nthreads; i++) 
        pivot_thread_reset(pool.thr+i, 0,0);

    for (i=0,n=0; i < R+1; i += ncols, n++)
    {
        pivot_thread_reset(pool.thr+n, 
                           i, 
                           min(i + ncols, R+1));
        DBG(1) fprintf(stderr, ">> assigned %i to (%i,%i) R=%i\n", n, pool.thr[n].start, pool.thr[n].end, R);
    }

    DBG(1) fprintf(stderr, "> assign_threads() %i %i\n", ncols, pool.nthreads);

    need_assign_pivot_threads = 0;
}


/*==========================================================================*/
/* pivot                                                                    */
/*==========================================================================*/
PyObject *samplex_pivot(PyObject *self, PyObject *args)
{
    int i,j;
    matrix_t tabl;

    PyObject *o = args;
    DBG(3) fprintf(stderr, "5> pivot()\n");

    int L = PyInt_AsLong(PyObject_GetAttrString(o, "nLeft")); /* # of constraints (rows)    */
    int R = PyInt_AsLong(PyObject_GetAttrString(o, "nRight")); /* # of variables   (columns) */
    int Z = PyInt_AsLong(PyObject_GetAttrString(o, "nTemp"));

    int T = PyInt_AsLong(PyObject_GetAttrString(o, "nthreads"));

    /* Remember, this is in FORTRAN order */
    PyObject *data = PyObject_GetAttrString(o, "data");
    tabl.data = (dble_t *)PyArray_DATA(data);
    tabl.w    = PyArray_DIM(data,0);
    tabl.h    = PyArray_DIM(data,1);

    PyObject *lhv = PyObject_GetAttrString(o, "lhv");
    PyObject *rhv = PyObject_GetAttrString(o, "rhv");

    int *left  = (int *)PyArray_DATA(lhv), 
        *right = (int *)PyArray_DATA(rhv);

    assert(PyArray_CHKFLAGS(data, NPY_F_CONTIGUOUS));
    assert(PyArray_CHKFLAGS(data, NPY_FORTRAN));

#if 0
    fprintf(stderr, "tabl dims = %i : %ix%i\n"
                    "L=%i R=%i Z=%i\n"
                    "len(left)=%ld, len(right)=%ld\n"
                    "nthreads=%i\n", 
                    PyArray_NDIM(data), tabl.h, tabl.w,
                    L, R, Z,
                    PyArray_DIM(lhv,0),
                    PyArray_DIM(rhv,0),
                    T
                    );
#endif

    size_t lpiv,    /* Pivot row              */
           rpiv;    /* Pivot column           */
    dble_t piv;     /* Value of pivot element */
    dble_t *pcol;   /* Pivot column           */

#if 0
        col = &(tabl.data[0 * tabl.w + 0]);
        for (i=0; i <= L; i++)
            fprintf(stderr, "%.4f ", col[i]);
        fprintf(stderr, "\n");
#endif

    int ret = choose_pivot(&tabl, left, right, L, R, &lpiv, &rpiv, &piv);
    //fprintf(stderr, "piv=%f\n", piv);
    //assert(ABS(piv) < 1e6);
    if (ret == FOUND_PIVOT) 
    {
        //----------------------------------------------------------------------
        // Actual pivot
        //----------------------------------------------------------------------
        //doPivot(&tabl, L, piv, lpiv, rpiv, 0, R+1);
        //====================================================================
        // Let the threads know they need to perform the pivot
        //
        // If there's only one thread, then just do the operation in the 
        // current thread instead of using other worker threads.
        //
        // In any event, we do at least one pivot operation in this thread.
        //====================================================================

        init_threads(T); 
        if (need_assign_pivot_threads) assign_threads(R); 
        for (i=0; i < pool.nthreads; i++)
        {
            pool.thr[i].tabl = &tabl;
            pool.thr[i].L    = L;
            pool.thr[i].piv  = piv;
            pool.thr[i].lpiv = lpiv;
            pool.thr[i].rpiv = rpiv;
        }

        startup_threads();
        doPivot(&pool.thr[0]);
        wait_for_threads();

        //----------------------------------------------------------------------
        // Start pivot
        //----------------------------------------------------------------------
        DBG(1) fprintf(stderr, "piv=%f\n", piv);
        pcol = &tabl.data[rpiv * tabl.w + 0];
        for (i=0; i <= L; i++)
            pcol[i] = rtz(pcol[i] / piv); 
        pcol[lpiv] = rtz(1.0 / piv);


        //----------------------------------------------------------------------
        // End pivot. Swap left and right variables.
        //----------------------------------------------------------------------
        int lq = left[lpiv];
        int rq = right[rpiv];

        left[lpiv]  = rq;
        right[rpiv] = lq;

        DBG(2)
        {
            for (i=0; i <= R; i++)
            {
                dble_t *col = &(tabl.data[i * tabl.w + 0]);
                for (j=0; j <= L; j++)
                    if (ABS(ABS(col[j]) - 22.00601215427127) < 1e-3)
                    {
                        fprintf(stderr, "++B+++ i=%i j=%i  col[j]=%f\n", i, j, col[j]);
                        //assert(0);
                    }

                    //fprintf(stderr, "%.4f ", col[j]);
                //fprintf(stderr, "\n");
            }
        }

        if (lq < 0) 
        { 
            DBG(1) fprintf(stderr, "\nREMOVING %ld\n\n", rpiv);
            //------------------------------------------------------------------
            // Remove the column at rpiv
            //------------------------------------------------------------------
            memmove(right+rpiv+0, 
                    right+rpiv+1, 
                    sizeof(*right)*(R-rpiv)); /* (R+1)-(rpiv+1) */

            memmove(tabl.data + (rpiv+0)*tabl.w, 
                    tabl.data + (rpiv+1)*tabl.w,
                    sizeof(*tabl.data) * (R-rpiv)*tabl.w);

            Z--; 
            R--;
            need_assign_pivot_threads = 1;
        }


        DBG(1)
        {
            for (i=0; i <= R; i++)
            {
                dble_t *col = &(tabl.data[i * tabl.w + 0]);
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
#if 0
            DBG(2)
            {
                end_time = CPUTIME;
                double t = end_time - start_time;
                fprintf(stderr, "feasible after %f\n", t);
            }
#endif

            ret = FEASIBLE;
        }
    }

    PyObject_SetAttrString(o, "nRight", PyInt_FromLong(R));
    PyObject_SetAttrString(o, "nTemp", PyInt_FromLong(Z));

    return PyInt_FromLong(ret);
}

#if 0

    //clock_t start_time;
    //clock_t end_time;

    //struct tms start_tms;
    //struct tms end_tms;


    //start_time = times(&start_tms);

    int ret = choosePivot();

    if (ret == FOUND_PIVOT)
    {
        ret = INFEASIBLE;

        //====================================================================
        // Transform the pivot column
        //====================================================================
        size_t i;
        dble_t *__restrict  pcol = tabl.data[rpiv];
        assert(piv != 0);
        const dble_t inv_piv = 1.0 / piv;
        const size_t L0 = L;
        for (i=0; i <= L0; i++) pcol[i] *= inv_piv;
        //for (i=0; i <= L0; i++) pcol[i] /= piv;
        pcol[lpiv] = inv_piv;

        //====================================================================
        // Let the threads know they need to perform the pivot
        //
        // If there's only one thread, then just do the operation in the 
        // current thread instead of using other worker threads.
        //
        // In any event, we do at least one pivot operation in this thread.
        //====================================================================

        if (need_assign_pivot_threads) assign_threads(); 

        startup_threads();

        doPivot0(pt[0].start, pt[0].end);

        wait_for_threads();

        //========================================================================
        // Swap left and right variables
        //========================================================================
        int lq = left.data[lpiv];
        int rq = right.data[rpiv];

        DBG(3) fprintf(stderr, "lq=%i rq=%i lpiv=%ld rpiv=%ld\n", lq, rq, lpiv, rpiv);
        
        left.data[lpiv]  = rq;
        right.data[rpiv] = lq;

        if (lq < 0) { 
            int_array_remove(&right, rpiv); 
            dd_array_remove(&tabl, rpiv); 
            Z--; 
            R--;
            need_assign_pivot_threads = 1;

            if (Z==0) 
            {
                DBG(2)
                {
                    end_time = CPUTIME;
                    double t = end_time - start_time;
                    fprintf(stderr, "feasible after %f\n", t);
                }

                ret = FEASIBLE;
                reorganize_table_memory();
            }

            assert(right.size == R+1);
            assert(tabl.size == R+1);
        }
    }

    //end_time = times(&end_tms);

    //DBG(1) end = CPUTIME;

    //DBG(2) fprintf(stderr, "< pivot() %f\n", ((double)(end_tms.tms_utime - start_tms.tms_utime) / sysconf(_SC_CLK_TCK) / nthreads));

    return ret;
}

#endif

/*==========================================================================*/
/* doPivot                                                                  */
/*                                                                          */
/* Now the code to do the pivoting.  An artificial variable that leaves     */
/* is removed.  If we are removing the last of these, we set |conv=true|.   */
/*==========================================================================*/
void doPivot(pivot_thread_t *thr)
{
    if (thr->start == thr->end) return;

    doPivot0(thr->tabl, 
             thr->L, 
             thr->piv, 
             thr->lpiv, 
             thr->rpiv, 
             thr->start, 
             thr->end);
}

//------------------------------------------------------------------------------
// The first column must always remain positive (except for the value of the
// objective function. We special-case this column to print error messages if
// we find a negative value.
//------------------------------------------------------------------------------
inline void in0(const size_t r, 
                ssize_t kp, 
                const size_t lpiv0, 
                const dble_t piv, 
                dble_t *__restrict col, 
                const dble_t *__restrict pcol)
{
    const dble_t col_lpiv = col[lpiv0];

    //fprintf(stderr, "in %i\n", r);

    /* Everything but the objective function value */
    for (; kp >= 1; kp--) 
    {
        dble_t v = col[kp] - (pcol[kp] * col_lpiv) / piv;

        //dble_t v = col[kp] - pcol[kp] * col_lpiv;
        //dble_t v = fdim(col[kp], pcol[kp] * col_lpiv);

        if (kp != 0 && kp != lpiv0 && rtz(v) < 0) 
        {
            fprintf(stderr, "**** \n lpiv=%ld pcol[%ld]=%.15e "
                            "col_lpiv=%.15e col[%ld]=%.15e v=%.15e\n", 
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
        fprintf(stderr, "r=%ld lpiv0=%ld col_lpiv=%.15e piv=%.15f\n", r, lpiv0, col_lpiv, piv);
        assert(0);
    }
}

inline void in(const size_t r, 
               ssize_t kp, 
               const size_t lpiv0, 
               const dble_t piv, 
               dble_t *__restrict col, 
               const dble_t *__restrict pcol)
{
    const dble_t col_lpiv = col[lpiv0];

    //fprintf(stderr, "in %i\n", r);
    for (; kp >= 0; kp--) 
    {
        col[kp] = rtz(col[kp] - (pcol[kp] * col_lpiv) / piv);

        //col[kp] = rtz(col[kp] - (pcol[kp] * col_lpiv) / piv);
        //col[kp] = rtz(col[kp] - pcol[kp] * col_lpiv);
//      if (r == 354 && kp == 917) {
//          fprintf(stderr, "col[%ld]=%f col_lpiv=%f lpiv0=%ld\n", kp, col[kp], col_lpiv, lpiv0);
//      }
    }
    col[lpiv0] = rtz(col_lpiv / -piv);
}

void doPivot0(matrix_t *tabl,
    const int L,
    const dble_t piv, 
    const int lpiv, const int rpiv,
    int start, const int end)
{

    //DBG(3) fprintf(stderr, "> doPivot()\n");
    //DBG(3) fprintf(stderr, "doPivot called: %i %i L=%ld R=%ld rpiv=%ld lpiv=%ld\n", start, end, L, R, rpiv, lpiv);

    size_t r=start;
    const dble_t *__restrict  pcol = &tabl->data[rpiv * tabl->w + 0]; 

#if 0
    if (start== 0)
    {
        in0(r, L, lpiv, piv, &tabl->data[r * tabl->w + 0], pcol);
        start++;
    }
#endif

    if (start <= rpiv && rpiv < end)
    {
        for (r=start; r < rpiv; ++r) in(r, L, lpiv, piv, &tabl->data[r * tabl->w + 0], pcol);
        for (++r; r < end; ++r)      in(r, L, lpiv, piv, &tabl->data[r * tabl->w + 0], pcol);
    }
    else
    {
        for (r=start; r < end; ++r) in(r, L, lpiv, piv, &tabl->data[r * tabl->w + 0], pcol);
    }

    DBG(2)
    {
        int i,j;
        for (i=1; i <= L; i++)
        {
            double v0 = tabl->data[0 * tabl->w + i];
            if (v0 == 0)
            {
                for (j=1; j < end; j++)
                {
                    double v1 = tabl->data[j * tabl->w + i];
                    if (v1 == 0)
                    {
                        fprintf(stderr, "ACK %i %i\n", i, j);
                        assert(0);
                    }
                }
            }
        }
    }

    //DBG(3) fprintf(stderr, "< doPivot()\n");
}

