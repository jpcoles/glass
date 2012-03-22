#ifndef __TIMING_H__
#define __TIMING_H__

#include <sys/time.h>
#include <sys/resource.h>

extern int getrusage();
#define CPUDEFS struct rusage ruse;
#define CPUTIME (getrusage(RUSAGE_SELF,&ruse),\
ruse.ru_utime.tv_sec + ruse.ru_stime.tv_sec + \
1e-6 * (ruse.ru_utime.tv_usec + ruse.ru_stime.tv_usec))

#endif 
