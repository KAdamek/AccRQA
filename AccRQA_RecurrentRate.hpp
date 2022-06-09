#ifndef _ACCRQA_RECURRENTRATE_HPP
#define _ACCRQA_RECURRENTRATE_HPP

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

void accrqaRecurrentRateGPU(float *RR, float *thresholds, int nThresholds, float *input, size_t input_size, int tau, int emb, int distance_type, int device);
void accrqaRecurrentRateGPU(double *RR, double *thresholds, int nThresholds, double *input, size_t input_size, int tau, int emb, int distance_type, int device);

void accrqaRecurrentRateCPU(float *RR, float *thresholds, int nThresholds, float *input, size_t input_size, int tau, int emb, int distance_type);
void accrqaRecurrentRateCPU(double *RR, double *thresholds, int nThresholds, double *input, size_t input_size, int tau, int emb, int distance_type);

void accrqaRecurrentRateERGPU(float *RR, float threshold, float *input, size_t input_size, int tau, int emb, int distance_type, int device);
void accrqaRecurrentRateERGPU(double *RR, double threshold, double *input, size_t input_size, int tau, int emb, int distance_type, int device);

#endif
