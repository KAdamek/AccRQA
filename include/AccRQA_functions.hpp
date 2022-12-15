#ifndef _ACCRQA_LENGTHHISTOGRAM_HPP
#define _ACCRQA_LENGTHHISTOGRAM_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "AccRQA_config.hpp"

void accrqaDeterminismGPU(
	float *DET, float *L, float *Lmax, float *ENTR, 
	float *input_data, size_t input_size, 
	float threshold, int tau, int emb, int lmin, int distance_type, 
	int device
);
void accrqaDeterminismGPU(
	double *DET, double *L, double *Lmax, double *ENTR, 
	double *input_data, size_t input_size, 
	double threshold, int tau, int emb, int lmin, int distance_type, 
	int device
);

void accrqaLaminarityGPU(
	float *LAM, float *TT, float *TTmax, 
	float *input_data, size_t input_size, 
	float threshold, int tau, int emb, int vmin, int distance_type, 
	int device
);
void accrqaLaminarityGPU(
	double *LAM, double *TT, double *TTmax, 
	double *input_data, size_t input_size, 
	double threshold, int tau, int emb, int vmin, int distance_type, 
	int device
);

void accrqaDeterminismCPU(
	float *DET, float *L, float *Lmax, float *ENTR, float *input_data, size_t input_size, float threshold, int tau, int emb, int lmin, int distance_type
);
void accrqaDeterminismCPU(
	double *DET, double *L, double *Lmax, double *ENTR, double *input_data, size_t input_size, double threshold, int tau, int emb, int lmin, int distance_type
);

void accrqaLaminarityCPU(
	float *LAM, float *TT, float *TTmax, float *input_data, size_t input_size, float threshold, int tau, int emb, int vmin, int distance_type
);
void accrqaLaminarityCPU(
	double *LAM, double *TT, double *TTmax, double *input_data, size_t input_size, double threshold, int tau, int emb, int vmin, int distance_type
);

void accrqaRecurrentRateGPU(
	float *RR, float *thresholds, int nThresholds, float *input, size_t input_size, int tau, int emb, int distance_type, int device
);
void accrqaRecurrentRateGPU(
	double *RR, double *thresholds, int nThresholds, double *input, size_t input_size, int tau, int emb, int distance_type, int device
);

void accrqaRecurrentRateCPU(
	float *RR, float *thresholds, int nThresholds, float *input, size_t input_size, int tau, int emb, int distance_type
);
void accrqaRecurrentRateCPU(
	double *RR, double *thresholds, int nThresholds, double *input, size_t input_size, int tau, int emb, int distance_type
);


#endif
