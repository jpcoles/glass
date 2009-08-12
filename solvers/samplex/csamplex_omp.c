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
#define SET_THREAD_AFFINITY     1

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

typedef struct
{
    int32_t step;
    int32_t nthreads;
    double obj_val;
} report_t;
report_t report = {0,0,0.0};

void doPivot0(
    matrix_t * restrict tabl,
    const long L, const long R,
    const dble_t piv, 
    const int32_t lpiv, const int32_t rpiv);



/*==========================================================================*/
/* The worker threads can be asked to do different things.                  */
/* They can either do the normal pivot operation, or they can capture       */
/* a portion of the table memory by allocating memory local to their CPU.   */
/*==========================================================================*/
#define PT_ACTION_STEAL_TABLE_COLS  (1)
#define PT_ACTION_DO_PIVOT          (2)

/* Do we need to update the columns that each thread is working on? */
static int32_t need_assign_pivot_threads;

PyObject *samplex_pivot(PyObject *self, PyObject *args);

static PyMethodDef csamplex_methods[] = 
{
    {"pivot", samplex_pivot, METH_O, "pivot"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcsamplex()
{
    (void)Py_InitModule("csamplex", csamplex_methods);
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

int32_t choose_pivot0(matrix_t *tabl, int32_t *left, int32_t *right, long L, long R,
                      int32_t *lpiv0, int32_t *rpiv0, dble_t *piv0)
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

    for (r = 1; r <= R; r++)
    {
        const dble_t * restrict col = &tabl->data[r * tabl->rows + 0];

        DBG(2) fprintf(stderr, "r=%i\n", r);

        if (col[0] <= coef) continue;

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
            if (col[k] >= -SML) continue; /* only interested in negative values */

            const dble_t tinc = -bcol[k] * col[0]/col[k];
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
    }

    DBG(2) fprintf(stderr, "< choosePivot() %i %i %e %e\n", lpiv, rpiv, piv, inc);

    *lpiv0 = lpiv;
    *rpiv0 = rpiv;
    *piv0  =  piv;
    return res;
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

    struct timespec t;

    int searchdir = -1;

    signal(SIGALRM, periodic_report);
    alarm(1);

    Py_BEGIN_ALLOW_THREADS

    for (n=0;; n++)
    {
        report.step     = n;
        report.obj_val  = tabl.data[0];
        report.nthreads = 0;

        ret = choose_pivot0(&tabl, left, right, L, R, &lpiv, &rpiv, &piv);

        if (ret != FOUND_PIVOT) break;

        //------------------------------------------------------------------
        // Actual pivot
        //------------------------------------------------------------------
        doPivot0(&tabl, L, R, piv, lpiv, rpiv);

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
    long L, const long R,
    dble_t piv, 
    int32_t lpiv, const int32_t rpiv)
{

    //DBG(3) fprintf(stderr, "> doPivot()\n");
    //DBG(3) fprintf(stderr, "doPivot called: %i %i L=%ld R=%ld rpiv=%i lpiv=%i\n", start, end, L, R, rpiv, lpiv);

    int32_t r;

    int32_t i;     
    dble_t * restrict col;
    dble_t col_lpiv;
    dble_t xx;

    dble_t * restrict pcol = tabl->data + (rpiv * tabl->rows);

    #pragma omp parallel for \
            private(i, col, col_lpiv, xx) \
            shared(pcol, piv, L, tabl, lpiv)
    for (r=0; r <= R; ++r)
    {
        if (r == rpiv) continue;

        col      = tabl->data + (r * tabl->rows); 
        col_lpiv = col[lpiv];     
        xx       = col_lpiv / piv;  

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
        col[lpiv] = -xx;   
    }

    #pragma omp parallel for shared(pcol, piv, L)
    for (i=0; i <= L; i++)
        pcol[i] /= piv;
    pcol[lpiv] = 1.0 / piv;


    //DBG(3) fprintf(stderr, "< doPivot()\n");
}

