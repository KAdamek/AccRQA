#include "../include/AccRQA_threads.hpp"
#include "../include/AccRQA_printf.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

static int g_accrqa_num_threads = 0;

void accrqa_set_num_threads(int n)
{
#ifdef _OPENMP
	if (n < 1){
		n = 1;
	}
	int maxn = omp_get_num_procs(); //omp_get_max_threads();
	if (n > maxn){
	       	n = maxn;
	}
	omp_set_num_threads(n);
	g_accrqa_num_threads = n;
#else
	(void)n;
	g_accrqa_num_threads = 1;
#endif
}

// get max number of threads
int accrqa_get_max_threads(void)
{
#ifdef _OPENMP
	return omp_get_num_procs();
#else
	return 1;
#endif
}

// get current number of threads
int accrqa_get_num_threads(void)
{
#ifdef _OPENMP
	if (g_accrqa_num_threads > 0)
        	return g_accrqa_num_threads;
	return omp_get_max_threads();
	#else
	return 1;
#endif
}

