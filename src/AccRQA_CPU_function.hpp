#ifndef _ACCRQA_CPU_FUNCTION
#define _ACCRQA_CPU_FUNCTION

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdint>

template<typename input_type>
void rqa_process_length_histogram(
	input_type *metric, 
	input_type *scan_histogram, 
	input_type *length_histogram, 
	size_t histogram_size
);

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
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	int64_t input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_LAM_metric(
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	int64_t input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_LAM_metric_parallel(
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	int64_t input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_DET_metric_ref(
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	int64_t input_size, 
	int distance_type
);


template<typename input_type>
void rqa_CPU_DET_metric(
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	int64_t input_size, 
	int distance_type
);

template<typename input_type>
void rqa_CPU_DET_metric_parallel_mk1(
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	int64_t input_size, 
	int distance_type
);

template<typename input_type>
void rqa_CPU_DET_metric_parallel_mk2(
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	int64_t input_size, 
	int distance_type
);
#include "AccRQA_CPU_function.cpp"

#endif
