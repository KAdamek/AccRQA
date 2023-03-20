#ifndef _ACCRQA_LENGTHHISTOGRAM_HPP
#define _ACCRQA_LENGTHHISTOGRAM_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "AccRQA_config.hpp"
#include "AccRQA_utilities_error.hpp"

//void accrqa_print_error(Accrqa_Error *error);

//==========================================================
//========================= LAM ============================
//==========================================================
void accrqa_LAM_GPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_LAM_GPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_LAM_CPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_LAM_CPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_LAM(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_LAM(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
int accrqa_LAM_output_size_in_elements(int nTaus, int nEmbs, int nVmins, int nThresholds);


//==========================================================
//========================= DET ============================
//==========================================================
void accrqa_DET_GPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_DET_GPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_DET_CPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_DET_CPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_DET(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
void accrqa_DET(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, Accrqa_Error *error);
int accrqa_DET_output_size_in_elements(int nTaus, int nEmbs, int nLmins, int nThresholds);


//==========================================================
//========================== RR ============================
//==========================================================
void accrqa_RR_GPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, float *threshold_values, int nThresholds, int distance_type, Accrqa_Error *error);
void accrqa_RR_GPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, double *threshold_values, int nThresholds, int distance_type, Accrqa_Error *error);
void accrqa_RR_CPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, float *threshold_values, int nThresholds, int distance_type, Accrqa_Error *error);
void accrqa_RR_CPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, double *threshold_values, int nThresholds, int distance_type, Accrqa_Error *error);
void accrqa_RR(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, float *threshold_values, int nThresholds, int distance_type, Accrqa_Error *error);
void accrqa_RR(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, double *threshold_values, int nThresholds, int distance_type, Accrqa_Error *error);
int accrqa_RR_output_size_in_elements(int nTaus, int nEmbs, int nThresholds);

#endif
