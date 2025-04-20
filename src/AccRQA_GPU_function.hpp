#ifndef _ACCRQA_GPU_FUNCTION_WRAPPERS
#define _ACCRQA_GPU_FUNCTION_WRAPPERS

#include "../include/AccRQA_utilities_error.hpp"

void GPU_RQA_R_matrix(int *R_matrix, float *h_input, unsigned long long int size, float threshold, int tau, int emb, int distance_type, Accrqa_Error *error);
void GPU_RQA_R_matrix(int *R_matrix, double *h_input, unsigned long long int size, double threshold, int tau, int emb, int distance_type, Accrqa_Error *error);


void GPU_RQA_diagonal_R_matrix(int *h_diagonal_R_matrix, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time, Accrqa_Error *error);
void GPU_RQA_diagonal_R_matrix(int *h_diagonal_R_matrix, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time, Accrqa_Error *error);


void GPU_RQA_RR_metric_integer(unsigned long long int *h_RR_metric_integer, float *h_input, size_t input_size, float *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, Accrqa_Error *error);
void GPU_RQA_RR_metric_integer(unsigned long long int *h_RR_metric_integer, double *h_input, size_t input_size, double *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, Accrqa_Error *error);


int GPU_RQA_length_start_end_test(unsigned long long int *h_length_histogram, int *h_input, long int input_size, int device, int nRuns, double *execution_time);


void GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error);
void GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error);


void GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error);
void GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error);


void GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error);
void GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error);

void GPU_RQA_length_histogram_diagonal_sum(double *h_DET, double *h_L, unsigned long long int *h_Lmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);
void GPU_RQA_length_histogram_diagonal_sum(float *h_DET, float *h_L, unsigned long long int *h_Lmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);

void GPU_RQA_diagonal_boxcar(double *h_DET, double *h_L, unsigned long long int *h_Lmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);
void GPU_RQA_diagonal_boxcar(float *h_DET, float *h_L, unsigned long long int *h_Lmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);

void GPU_RQA_diagonal_boxcar_square(double *h_DET, double *h_L, unsigned long long int *h_Lmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);
void GPU_RQA_diagonal_boxcar_square(float *h_DET, float *h_L, unsigned long long int *h_Lmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);

void GPU_RQA_horizontal_boxcar_square(double *h_LAM, double *h_TT, unsigned long long int *h_TTmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);
void GPU_RQA_horizontal_boxcar_square(float *h_LAM, float *h_TT, unsigned long long int *h_TTmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error);

#endif