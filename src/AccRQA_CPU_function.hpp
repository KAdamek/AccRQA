#ifndef _ACCRQA_CPU_FUNCTION
#define _ACCRQA_CPU_FUNCTION

#include <stdio.h>
#include <stdlib.h>
#include <vector>

template <typename input_type>
void rqa_CPU_RR_metric_ref(
	input_type *output_RR, 
	input_type *time_series, 
	unsigned long long int input_size, 
	input_type threshold, 
	int tau, 
	int emb, 
	int distance_type
);

template <typename input_type>
void rqa_CPU_RR_metric_ref_parallel(
	input_type *output_RR, 
	input_type *time_series, 
	unsigned long long int input_size, 
	input_type threshold, 
	int tau, 
	int emb, 
	int distance_type
);

template <typename input_type>
void rqa_CPU_RR_metric(
	unsigned long long int *recurrent_rate_integers, 
	std::vector<input_type> threshold_list, 
	int tau, 
	int emb, 
	input_type *time_series, 
	unsigned long long int input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_LAM_metric_ref(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_LAM_metric(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_DET_metric_ref(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_DET_metric(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
);

#include "AccRQA_CPU_function.cpp"

#endif
