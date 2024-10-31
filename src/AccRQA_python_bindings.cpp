#include "../include/AccRQA_definitions.hpp"
#include "../include/AccRQA_functions.hpp"
#include "../include/AccRQA_utilities_mem.hpp"
#include "../include/AccRQA_utilities_error.hpp"
#include "../include/AccRQA_utilities_comp_platform.hpp"
#include "../include/AccRQA_utilities_distance.hpp"

/**
 * @brief Checks that distance to Line of Identity selected is valid or issues en error if not.
 * @returns Distance type as Accrqa_Distance enum type.
 *
 * @param int_distance_type Input distance type as integer.
 * @param error Error status.
 */
Accrqa_Distance check_distance_type(int int_distance_type, Accrqa_Error *error){
	Accrqa_Distance distance_type;
	if(int_distance_type==1){
		distance_type = DST_EUCLIDEAN;
	}
	else if(int_distance_type==2){
		distance_type = DST_MAXIMAL;
	}
	else {
		distance_type = DST_ERROR;
		printf("Error: Invalid distance type.\n");
		*error = ERR_INVALID_ARGUMENT;
	}
	return(distance_type);
}

/**
 * @brief Checks that computational platform selected is valid or issues en error if not.
 * @returns Computational platform as Accrqa_CompPlatform enum type.
 *
 * @param int_comp_platform Input computational platform as integer.
 * @param error Error status.
 */
Accrqa_CompPlatform check_comp_platform(int int_comp_platform, Accrqa_Error *error){
	Accrqa_CompPlatform comp_platform;
	if(int_comp_platform==1){
		comp_platform = PLT_CPU;
	}
	else if(int_comp_platform==1024){
		comp_platform = PLT_NV_GPU;
	}
	else {
		printf("Error: Invalid computing platform.\n");
		comp_platform = PLT_ERROR;
		*error = ERR_INVALID_ARGUMENT;
	}
	return(comp_platform);
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates RR RQA metric from supplied time-series.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p mem_output_RR is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nThresholds ].
 *
 * - @p mem_input is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p mem_tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p mem_emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p mem_threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param mem_output_RR Multi-dimensional data cube containing RR values.
 * @param mem_input Real-valued array of input time-series samples.
 * @param mem_tau_values Integer array of delay values.
 * @param mem_emb_values Integer array of embedding values.
 * @param mem_threshold_values Real-valued array of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void py_accrqa_RR(
	Mem *mem_output_RR, 
	Mem *mem_input, 
	Mem *mem_tau_values, 
	Mem *mem_emb_values, 
	Mem *mem_threshold_values, 
	int int_distance_type, 
	int int_comp_platform, 
	Accrqa_Error *error
) {
	if(
		mem_location(mem_output_RR) != MEM_CPU 
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
		mem_type(mem_input) != mem_type(mem_output_RR) 
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
	
	if( mem_num_dims(mem_output_RR) != 3 ){
		printf("mem_output_RR must be four-dimensional.\n");
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
	
	Accrqa_Distance distance_type = check_distance_type(int_distance_type, error);
	if(*error != SUCCESS) return;
	
	Accrqa_CompPlatform comp_platform = check_comp_platform(int_comp_platform, error);
	if(*error != SUCCESS) return;
	
	int *tau_values = (int *) mem_data(mem_tau_values);
	int nTaus = mem_shape_dim(mem_tau_values, 0);
	int *emb_values = (int *) mem_data(mem_emb_values);
	int nEmbs = mem_shape_dim(mem_emb_values, 0);
	
	if(mem_type(mem_input) == MEM_FLOAT){
		float *threshold_values = (float *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		float *input_data = (float *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		float *output = (float *) mem_data(mem_output_RR);
		
		accrqa_RR(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, comp_platform, error);
	}
	else if(mem_type(mem_input) == MEM_DOUBLE){
		double *threshold_values = (double *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		double *input_data = (double *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		double *output = (double *) mem_data(mem_output_RR);
		
		accrqa_RR(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, comp_platform, error);
	}
	else {
		printf("ERROR! Unsupported data type.\n");
		*error = ERR_DATA_TYPE;
	}
}


/**
 * @brief Calculates DET, L, Lmax, ENTR and RR RQA metrics from supplied time-series.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p mem_output_DET is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nLmins, @p nThresholds , 5 ].
 *
 * - @p mem_input is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p mem_tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p mem_emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ] 
 *
 * - @p mem_lmin_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p mem_threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param mem_output_DET Multi-dimensional data cube containing RR values.
 * @param mem_input Real-valued array of input time-series samples.
 * @param mem_tau_values Integer array of delay values.
 * @param mem_emb_values Integer array of embedding values.
 * @param mem_lmin_values Integer array of  minimal lengths values.
 * @param mem_threshold_values Real-valued array (float) of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param calc_ENTR Turns calculation of ENTR on (1) and off (0).
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void py_accrqa_DET(
	Mem *mem_output_DET, 
	Mem *mem_input, 
	Mem *mem_tau_values, 
	Mem *mem_emb_values, 
	Mem *mem_lmin_values, 
	Mem *mem_threshold_values, 
	int int_distance_type, 
	int calc_ENTR, 
	int int_comp_platform, 
	Accrqa_Error *error
){
	if(
		mem_location(mem_output_DET) != MEM_CPU 
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
		mem_type(mem_input) != mem_type(mem_output_DET) 
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
	
	if( mem_num_dims(mem_output_DET) != 5 ){
		printf("mem_output_DET must be four-dimensional.\n");
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
	
	Accrqa_Distance distance_type = check_distance_type(int_distance_type, error);
	if(*error != SUCCESS) return;
	
	Accrqa_CompPlatform comp_platform = check_comp_platform(int_comp_platform, error);
	if(*error != SUCCESS) return;
	
	int *tau_values = (int *) mem_data(mem_tau_values);
	int nTaus = mem_shape_dim(mem_tau_values, 0);
	int *emb_values = (int *) mem_data(mem_emb_values);
	int nEmbs = mem_shape_dim(mem_emb_values, 0);
	int *lmin_values = (int *) mem_data(mem_lmin_values);
	int nLmins = mem_shape_dim(mem_lmin_values, 0);
	
	if(mem_type(mem_input) == MEM_FLOAT){
		float *threshold_values = (float *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		float *input_data = (float *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		float *output = (float *) mem_data(mem_output_DET);
		
		accrqa_DET(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, comp_platform, error);
	}
	else if(mem_type(mem_input) == MEM_DOUBLE){
		double *threshold_values = (double *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		double *input_data = (double *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		double *output = (double *) mem_data(mem_output_DET);
		
		accrqa_DET(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, comp_platform, error);
	}
	else {
		printf("ERROR! Unsupported data type.\n");
		*error = ERR_DATA_TYPE;
	}
}



/**
 * @brief Calculates LAM, TT, TTmax, ENTR and RR RQA metrics from supplied time-series.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p mem_output_LAM is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nLmins, @p nThresholds , 5 ].
 *
 * - @p mem_input is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ] 
 *
 * - @p lmin_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param mem_output_LAM Multi-dimensional data cube containing RR values.
 * @param mem_input Real-valued array of input time-series samples.
 * @param mem_tau_values Integer array of delay values.
 * @param mem_emb_values Integer array of embedding values.
 * @param mem_vmin_values Integer array of  minimal lengths values.
 * @param mem_threshold_values Real-valued array (double) of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param calc_ENTR Turns calculation of ENTR on (1) and off (0).
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void py_accrqa_LAM(
	Mem *mem_output_LAM, 
	Mem *mem_input, 
	Mem *mem_tau_values, 
	Mem *mem_emb_values, 
	Mem *mem_vmin_values, 
	Mem *mem_threshold_values, 
	int int_distance_type, 
	int calc_ENTR, 
	int int_comp_platform, 
	Accrqa_Error *error
) {
	if(
		mem_location(mem_output_LAM) != MEM_CPU 
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
		mem_type(mem_input) != mem_type(mem_output_LAM) 
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
	
	if( mem_num_dims(mem_output_LAM) != 5 ){
		printf("mem_output_LAM must be four-dimensional.\n");
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
	
	Accrqa_Distance distance_type = check_distance_type(int_distance_type, error);
	if(*error != SUCCESS) return;
	
	Accrqa_CompPlatform comp_platform = check_comp_platform(int_comp_platform, error);
	if(*error != SUCCESS) return;
	
	int *tau_values = (int *) mem_data(mem_tau_values);
	int nTaus = mem_shape_dim(mem_tau_values, 0);
	int *emb_values = (int *) mem_data(mem_emb_values);
	int nEmbs = mem_shape_dim(mem_emb_values, 0);
	int *vmin_values = (int *) mem_data(mem_vmin_values);
	int nVmins = mem_shape_dim(mem_vmin_values, 0);
	
	if(mem_type(mem_input) == MEM_FLOAT){
		float *threshold_values = (float *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		float *input_data = (float *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		float *output = (float *) mem_data(mem_output_LAM);
		
		accrqa_LAM(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, comp_platform, error);
	}
	else if(mem_type(mem_input) == MEM_DOUBLE){
		double *threshold_values = (double *) mem_data(mem_threshold_values);
		int nThresholds = mem_shape_dim(mem_threshold_values, 0);
		double *input_data = (double *) mem_data(mem_input);
		size_t input_size = (size_t) mem_shape_dim(mem_input, 0);
		double *output = (double *) mem_data(mem_output_LAM);
		
		accrqa_LAM(output, input_data, input_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, comp_platform, error);
	}
	else {
		printf("ERROR! Unsupported data type.\n");
		*error = ERR_DATA_TYPE;
	}
}


#ifdef __cplusplus
}
#endif

