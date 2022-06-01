#ifndef _ACCRQA_GPU_FUNCTION_WRAPPERS
#define _ACCRQA_GPU_FUNCTION_WRAPPERS

int GPU_RQA_R_matrix(int *R_matrix, float *h_input, unsigned long long int size, float threshold, int tau, int emb, int distance_type, int device, double *execution_time);
int GPU_RQA_R_matrix(int *R_matrix, double *h_input, unsigned long long int size, double threshold, int tau, int emb, int distance_type, int device, double *execution_time);


int GPU_RQA_diagonal_R_matrix(int *h_diagonal_R_matrix, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);
int GPU_RQA_diagonal_R_matrix(int *h_diagonal_R_matrix, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);


int GPU_RQA_RR_metric(unsigned long long int *h_RR_metric_integer, float *h_input, size_t input_size, float *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, int device, double *execution_time);
int GPU_RQA_RR_metric(unsigned long long int *h_RR_metric_integer, double *h_input, size_t input_size, double *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, int device, double *execution_time);

int GPU_RQA_RR_ER_metric(unsigned long long int *h_RR_metric_integer, float *h_input, size_t input_size, float *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, int device, double *execution_time);
int GPU_RQA_RR_ER_metric(unsigned long long int *h_RR_metric_integer, double *h_input, size_t input_size, double *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, int device, double *execution_time);

int GPU_RQA_length_start_end_test(unsigned long long int *h_length_histogram, int *h_input, long int input_size, int device, int nRuns, double *execution_time);


int GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);
int GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);


int GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);
int GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);


int GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);
int GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, int device, double *execution_time);


#endif