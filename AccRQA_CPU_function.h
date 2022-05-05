#ifndef _ACCRQA_CPU_FUNCTION
#define _ACCRQA_CPU_FUNCTION

#include "debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>


template <typename input_type>
int rqa_R_matrix(int *R_matrix, input_type *time_series, long int corrected_size, input_type threshold, int tau, int emb, int distance_type);


template <typename input_type>
int rqa_R_matrix_diagonal(int *R_matrix_diagonal, input_type *time_series, long int corrected_size, input_type threshold, int tau, int emb, int distance_type);


template <typename input_type>
void rqa_RR_metric(unsigned long long int *recurrent_rate_integers, std::vector<input_type> threshold_list, int tau, int emb, input_type *time_series, unsigned long long int input_size, int distance_type);


template<typename input_type>
void rqa_LAM_metric_CPU(unsigned long long int *metric, unsigned long long int *scan_histogram, unsigned long long int *length_histogram, input_type threshold, int tau, int emb, input_type *time_series, long int input_size, int distance_type);


template<typename input_type>
void rqa_DET_metric_CPU(unsigned long long int *metric, unsigned long long int *scan_histogram, unsigned long long int *length_histogram, input_type threshold, int tau, int emb, input_type *time_series, long int input_size, int distance_type);


//void get_length_histogram(unsigned long long int *LAM_histogram, int *time_series, size_t size);

#include "AccRQA_CPU_function.cpp"

#endif
