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

#include <sched.h>
#include <errno.h>

#include "timing.h"

#define WITH_WELL 1

#if WITH_WELL
#include "WELL44497a.h"
#define U01 WELLRNG44497a
#else
#define U01 drand48
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

/*==========================================================================*/
/* Data structures for the table and arrays.                                */
/*==========================================================================*/

typedef double dble_t __attribute__ ((aligned(8)));

typedef struct
{
    uint32_t cols;
    uint32_t rows;
    dble_t * restrict orig;
    dble_t * restrict data;
    dble_t * restrict pcol;
} matrix_t __attribute__ ((aligned(8)));

PyObject *samplex_rwalk(PyObject *self, PyObject *args);
PyObject *set_rwalk_seed(PyObject *self, PyObject *args);

static PyMethodDef csamplex_methods[] = 
{
    {"rwalk", samplex_rwalk, METH_VARARGS, "rwalk"},
    {"set_rwalk_seed", set_rwalk_seed, METH_O, "set_rwalk_seed"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcsamplex()
{
    (void)Py_InitModule("csamplex", csamplex_methods);
}

double ggl(double *ds)
{
    /* generate u(0,1) distributed random numbers.
    Seed ds must be saved between calls. ggl is
    essentially the same as the IMSL routine RNUM.

    W. Petersen and M. Troyer, 24 Oct. 2002, ETHZ */

    double t,d2=0.2147483647e10;
    t = (double) *ds;
    t = fmod(0.16807e5*t,d2);
    *ds = t;
    return((double)((t-1.0e0)/(d2-1.0e0)));
}

PyObject *set_rwalk_seed(PyObject *self, PyObject *args)
{
    double seed;
    if (args == Py_None)
        seed = time(NULL);
    else
        seed = PyInt_AsLong(args);
 
#if WITH_WELL
    unsigned int init[1391];
    double scale = 5.0e+9; // almost any big scale factor will work
    int i;
    for(i=0;i<1391;i++) init[i] = (unsigned int) scale*ggl(&seed);
    InitWELLRNG44497a(init);
#else
    srand48(seed);
#endif

    fprintf(stderr, "First random number is %f\n", U01());

    return Py_None;
}

/*****************************************************************************/
/*****************************************************************************/

void normal(double stddev, double mean, double *r0, double *r1)
{

    double x1, x2, w, y1, y2;
 
    do 
    {
        x1 = 2.0 * U01() - 1.0;
        x2 = 2.0 * U01() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while (w >= 1.0);

    w = sqrt((-2.0 * log(w)) / w);
    y1 = x1 * w;
    y2 = x2 * w;

    // XXX
    //y1 = 2.0 * drand48() - 1.0;
    //y2 = 2.0 * drand48() - 1.0;
    // XXX
    *r0 = y1 * stddev + mean;
    *r1 = y2 * stddev + mean;
}

PyObject *samplex_rwalk(PyObject *self, PyObject *args)
{
    int32_t i,j;
    matrix_t evec, eqs;

    DBG(3) fprintf(stderr, "5> pivot()\n");

    long accepted;
    long rejected;
    double twiddle;

    PyObject *po_vec;
    PyObject *po_eval;
    PyObject *po_evec;
    PyObject *po_eqs;
    PyObject *po_S;
    PyObject *po_S0;

    if (!PyArg_ParseTuple(args, "OOOOOOOdll", &self, &po_eqs, &po_vec, &po_eval, &po_evec, &po_S, &po_S0, &twiddle, &accepted, &rejected))
        return NULL;

    assert(PyArray_CHKFLAGS(po_evec, NPY_F_CONTIGUOUS));
    //assert(PyArray_CHKFLAGS(po_eval, NPY_C));  // One dimensional, so we don't need to check

          long redo = PyInt_AsLong(PyObject_GetAttrString(self, "redo"));
    const long dim = PyInt_AsLong(PyObject_GetAttrString(self, "dim"));
    const long dof = PyInt_AsLong(PyObject_GetAttrString(self, "dof"));

    const long  eq_count = PyInt_AsLong(PyObject_GetAttrString(self, "eq_count"));
    const long geq_count = PyInt_AsLong(PyObject_GetAttrString(self, "geq_count"));
    const long leq_count = PyInt_AsLong(PyObject_GetAttrString(self, "leq_count"));

    dble_t * restrict vec  = (dble_t * restrict)PyArray_DATA(po_vec), 
           * restrict eval = (dble_t * restrict)PyArray_DATA(po_eval),
           * restrict S    = (dble_t * restrict)PyArray_DATA(po_S),
           * restrict S0   = (dble_t * restrict)PyArray_DATA(po_S0);

    /* Remember, this is in C order */
    evec.data = (dble_t * restrict)PyArray_DATA(po_evec);
    evec.rows = PyArray_DIM(po_evec,0);
    evec.cols = PyArray_DIM(po_evec,1);

    double * data_ptr;
    eqs.data = (double * restrict)PyArray_DATA(po_eqs);
    eqs.rows = PyArray_DIM(po_eqs,0);
    eqs.cols = PyArray_DIM(po_eqs,1);

    const long eq_offs = 0;
    const long leq_offs = eq_offs + eq_count;
    const long geq_offs = leq_offs + leq_count;

    //fprintf(stderr, "eq  %ld %ld\n", eq_offs, eq_count);
    //fprintf(stderr, "leq %ld %ld\n", leq_offs, leq_count);
    //fprintf(stderr, "geq %ld %ld\n", geq_offs, geq_count);
    //fprintf(stderr, "redo %ld\n", redo);
    //fprintf(stderr, "accepted/rejected/twiddle %ld %ld %e\n", accepted,rejected,twiddle);

    Py_BEGIN_ALLOW_THREADS

    for (i=0; i < eqs.rows; i++)
    {
        long offs = i * eqs.cols;

        S[i] = eqs.data[offs++];
        for (j=0; j < dim; j++)
            S[i] += vec[j] * eqs.data[offs++];
    }

    double r,r1;
    double step;
    long walk_step;
    long dir_index;

    double redo_stime = CPUTIME;
    for (walk_step = 0; walk_step < redo; walk_step++)
    {
        /* Choose a random eigen direction */
        do
        {
            dir_index = U01() * dim;
        } while (fabs(eval[dir_index]) < 1e-14);

        if (!(walk_step & 1))
            normal(twiddle/sqrt(dof), 0, &r, &r1);
        else
            r = r1;

        step = r * eval[dir_index];

        /* Check if we are still in the simplex */
        long offs = leq_offs * eqs.cols + dir_index + 1;
        data_ptr = eqs.data + offs;
        int accept = 1;

#if 0
        for (i=eq_offs; accept && i < (eq_offs + eq_count); i++, data_ptr += eqs.cols)
        {
            S0[i] = S[i] + (step * *data_ptr);
            accept = fabs(S0[i]) < 1e-8;
        }
#endif

        // equalities are ignored
        for (i=leq_offs; accept && i < (leq_offs + leq_count); i++, data_ptr += eqs.cols)
        {
            S0[i] = S[i] + (step * *data_ptr);
            accept = S0[i] <= 0;
        }

        //for (i=geq_offs; accept && i < (geq_offs + geq_count); i++, data_ptr += eqs.cols)
        for (i=geq_offs; accept && i < eqs.rows; i++, data_ptr += eqs.cols)
        {
            S0[i] = S[i] + (step * *data_ptr);
            accept = S0[i] >= 0;
        }

        assert(!accept || i==eqs.rows);


        /* Maybe take the new point as the current vector for the next round */
        if (accept)
        {
            vec[dir_index] += step;
            memcpy(S, S0, sizeof(*S) * eqs.rows);
            accepted++;
        }
        else
        {
            rejected++;
        }
    }
    double redo_etime = CPUTIME;

    fprintf(stderr, "TOTAL TOOK %fs\n", redo_etime-redo_stime);

    Py_END_ALLOW_THREADS

    return Py_BuildValue("ll", accepted, rejected);
    return PyInt_FromLong(0);
}
