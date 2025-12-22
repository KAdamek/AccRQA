#ifndef ACCRQA_THREADS_HPP
#define ACCRQA_THREADS_HPP

#ifdef __cplusplus
extern "C" {
#endif

void accrqa_set_num_threads(int n);
int  accrqa_get_max_threads(void);
int  accrqa_get_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif

