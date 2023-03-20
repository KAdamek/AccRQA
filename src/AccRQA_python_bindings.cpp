#include "../include/AccRQA_definitions.hpp"
#include "../include/AccRQA_functions.hpp"

#include "../include/AccRQA_utilities_mem.hpp"
#include "../include/AccRQA_utilities_error.hpp"


#ifdef __cplusplus
extern "C" {
#endif

void py_accrqa_RR(Mem *mem_output, Mem *mem_input, Mem *mem_tau_values, Mem *mem_emb_values, Mem *mem_threshold_values, int distance_type, Accrqa_Error *error){
	printf("in accrqa_RR\n");
	if(
		mem_location(mem_output) != MEM_CPU 
		|| mem_location(mem_input) != MEM_CPU 
		|| mem_location(mem_tau_values) != MEM_CPU 
		|| mem_location(mem_emb_values) != MEM_CPU 
		|| mem_location(mem_threshold_values) != MEM_CPU
	){
		printf("ERROR! Data are stored in wrong destination.\n");
		*error = ERR_MEM_LOCATION;
		return;
	}
	
	if( 
		mem_type(mem_input) != mem_type(mem_output) 
		|| mem_type(mem_input) != mem_type(mem_threshold_values) 
	) {
		printf("ERROR! Input, output and threshold values must have the same data type.\n");
		*error = ERR_DATA_TYPE;
		return;
	}
	
	if( 
		mem_type(mem_tau_values) != MEM_INT 
		|| mem_type(mem_emb_values) != MEM_INT
	) {
		printf("ERROR! tau and emb must be integers.\n");
		*error = ERR_DATA_TYPE;
		return;
	}
	
	if(
		mem_num_dims(mem_tau_values) != 1 
		|| mem_num_dims(mem_emb_values) != 1 
		|| mem_num_dims(mem_threshold_values) != 1
		|| mem_num_dims(mem_input) != 1
	) {
		printf("input, tau, emb and threshold must be one-dimensional.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	if( mem_num_dims(mem_output) != 3 ){
		printf("mem_output must be four-dimensional.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	if( 
		mem_shape_dim(mem_input, 0) == 0
		|| mem_shape_dim(mem_tau_values, 0) == 0
		|| mem_shape_dim(mem_emb_values, 0) == 0
		|| mem_shape_dim(mem_threshold_values, 0) == 0
	) {
		printf("input, tau, emb and threshold number of elements must be non zero.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	int *tau_values = (int *) mem_data(mem_tau_values);
	int nTaus = mem_shape_dim(mem_tau_values, 0);
	int *emb_values = (int *) mem_data(mem_emb_values);
	int nEmbs = mem_shape_dim(mem_emb_values, 0);
	
	if(mem_type(mem_input) == MEM_FLOAT){
		printf("doing float\n");
		float *threshold_values = (float *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		float *input_data = (float *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		float *output = (float *) mem_data(mem_output);
		
		accrqa_RR(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, ACCRQA_METRIC_MAXIMAL, error);
	}
	else if(mem_type(mem_input) == MEM_DOUBLE){
		printf("doing double\n");
		double *threshold_values = (double *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		double *input_data = (double *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		double *output = (double *) mem_data(mem_output);
		
		accrqa_RR(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, ACCRQA_METRIC_MAXIMAL, error);
	}
	else {
		printf("ERROR! Unsupported data type.\n");
		*error = ERR_DATA_TYPE;
	}
}



void py_accrqa_DET(Mem *mem_output, Mem *mem_input, Mem *mem_tau_values, Mem *mem_emb_values, Mem *mem_lmin_values, Mem *mem_threshold_values, int distance_type, int calc_ENTR, Accrqa_Error *error){
	if(
		mem_location(mem_output) != MEM_CPU 
		|| mem_location(mem_input) != MEM_CPU 
		|| mem_location(mem_tau_values) != MEM_CPU 
		|| mem_location(mem_emb_values) != MEM_CPU 
		|| mem_location(mem_lmin_values) != MEM_CPU 
		|| mem_location(mem_threshold_values) != MEM_CPU
	){
		printf("ERROR! Data are stored in wrong destination.\n");
		*error = ERR_MEM_LOCATION;
		return;
	}
	
	if( 
		mem_type(mem_input) != mem_type(mem_output) 
		|| mem_type(mem_input) != mem_type(mem_threshold_values) 
	) {
		printf("ERROR! Input, output and threshold values must have the same data type.\n");
		*error = ERR_DATA_TYPE;
		return;
	}
	
	if( 
		mem_type(mem_tau_values) != MEM_INT 
		|| mem_type(mem_emb_values) != MEM_INT
		|| mem_type(mem_lmin_values) != MEM_INT
	) {
		printf("ERROR! tau, emb and lmin must be integers.\n");
		*error = ERR_DATA_TYPE;
		return;
	}
	
	if(
		mem_num_dims(mem_tau_values) != 1 
		|| mem_num_dims(mem_emb_values) != 1 
		|| mem_num_dims(mem_lmin_values) != 1 
		|| mem_num_dims(mem_threshold_values) != 1
		|| mem_num_dims(mem_input) != 1
	) {
		printf("input, tau, emb and threshold must be one-dimensional.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	if( mem_num_dims(mem_output) != 5 ){
		printf("mem_output must be four-dimensional.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	if( 
		mem_shape_dim(mem_input, 0) == 0
		|| mem_shape_dim(mem_tau_values, 0) == 0
		|| mem_shape_dim(mem_emb_values, 0) == 0
		|| mem_shape_dim(mem_lmin_values, 0) == 0
		|| mem_shape_dim(mem_threshold_values, 0) == 0
	) {
		printf("input, tau, emb and threshold number of elements must be non zero.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	int *tau_values = (int *) mem_data(mem_tau_values);
	int nTaus = mem_shape_dim(mem_tau_values, 0);
	int *emb_values = (int *) mem_data(mem_emb_values);
	int nEmbs = mem_shape_dim(mem_emb_values, 0);
	int *lmin_values = (int *) mem_data(mem_lmin_values);
	int nLmins = mem_shape_dim(mem_lmin_values, 0);
	
	if(mem_type(mem_input) == MEM_FLOAT){
		printf("doing float\n");
		float *threshold_values = (float *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		float *input_data = (float *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		float *output = (float *) mem_data(mem_output);
		
		accrqa_DET(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, ACCRQA_METRIC_MAXIMAL, calc_ENTR, error);
	}
	else if(mem_type(mem_input) == MEM_DOUBLE){
		printf("doing double\n");
		double *threshold_values = (double *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		double *input_data = (double *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		double *output = (double *) mem_data(mem_output);
		
		accrqa_DET(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, ACCRQA_METRIC_MAXIMAL, calc_ENTR, error);
	}
	else {
		printf("ERROR! Unsupported data type.\n");
		*error = ERR_DATA_TYPE;
	}
}



void py_accrqa_LAM(Mem *mem_output, Mem *mem_input, Mem *mem_tau_values, Mem *mem_emb_values, Mem *mem_vmin_values, Mem *mem_threshold_values, int distance_type, int calc_ENTR, Accrqa_Error *error){
	if(
		mem_location(mem_output) != MEM_CPU 
		|| mem_location(mem_input) != MEM_CPU 
		|| mem_location(mem_tau_values) != MEM_CPU 
		|| mem_location(mem_emb_values) != MEM_CPU 
		|| mem_location(mem_vmin_values) != MEM_CPU 
		|| mem_location(mem_threshold_values) != MEM_CPU
	){
		printf("ERROR! Data are stored in wrong destination.\n");
		*error = ERR_MEM_LOCATION;
		return;
	}
	
	if( 
		mem_type(mem_input) != mem_type(mem_output) 
		|| mem_type(mem_input) != mem_type(mem_threshold_values) 
	) {
		printf("ERROR! Input, output and threshold values must have the same data type.\n");
		*error = ERR_DATA_TYPE;
		return;
	}
	
	if( 
		mem_type(mem_tau_values) != MEM_INT 
		|| mem_type(mem_emb_values) != MEM_INT
		|| mem_type(mem_vmin_values) != MEM_INT
	) {
		printf("ERROR! tau, emb and lmin must be integers.\n");
		*error = ERR_DATA_TYPE;
		return;
	}
	
	if(
		mem_num_dims(mem_tau_values) != 1 
		|| mem_num_dims(mem_emb_values) != 1 
		|| mem_num_dims(mem_vmin_values) != 1 
		|| mem_num_dims(mem_threshold_values) != 1
		|| mem_num_dims(mem_input) != 1
	) {
		printf("input, tau, emb and threshold must be one-dimensional.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	if( mem_num_dims(mem_output) != 5 ){
		printf("mem_output must be four-dimensional.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	if( 
		mem_shape_dim(mem_input, 0) == 0
		|| mem_shape_dim(mem_tau_values, 0) == 0
		|| mem_shape_dim(mem_emb_values, 0) == 0
		|| mem_shape_dim(mem_vmin_values, 0) == 0
		|| mem_shape_dim(mem_threshold_values, 0) == 0
	) {
		printf("input, tau, emb and threshold number of elements must be non zero.\n");
		*error = ERR_INVALID_ARGUMENT;
		return;
	}
	
	int *tau_values = (int *) mem_data(mem_tau_values);
	int nTaus = mem_shape_dim(mem_tau_values, 0);
	int *emb_values = (int *) mem_data(mem_emb_values);
	int nEmbs = mem_shape_dim(mem_emb_values, 0);
	int *vmin_values = (int *) mem_data(mem_vmin_values);
	int nVmins = mem_shape_dim(mem_vmin_values, 0);
	
	if(mem_type(mem_input) == MEM_FLOAT){
		printf("doing float\n");
		float *threshold_values = (float *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		float *input_data = (float *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		float *output = (float *) mem_data(mem_output);
		
		accrqa_LAM(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, ACCRQA_METRIC_MAXIMAL, calc_ENTR, error);
	}
	else if(mem_type(mem_input) == MEM_DOUBLE){
		printf("doing double\n");
		double *threshold_values = (double *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		double *input_data = (double *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		double *output = (double *) mem_data(mem_output);
		
		accrqa_LAM(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, ACCRQA_METRIC_MAXIMAL, calc_ENTR, error);
	}
	else {
		printf("ERROR! Unsupported data type.\n");
		*error = ERR_DATA_TYPE;
	}
}

#ifdef __cplusplus
}
#endif

