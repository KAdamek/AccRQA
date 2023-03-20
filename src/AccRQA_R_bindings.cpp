#include <R.h>
#include "../include/AccRQA_library.hpp"

#ifdef __cplusplus
extern "C" {
#endif
	void R_double_accrqa_DET(
		double *output, 
		double *input, 
		int    *input_size, 
		int    *tau_values,
		int    *nTaus,
		int    *emb_values,
		int    *nEmbs,
		int    *lmin_values,
		int    *nLmins,
		double *threshold_values, 
		int    *nThresholds, 
		int    *distance_type,
		int    *calc_ENTR
	){
		size_t local_input_size = (size_t) input_size[0];
		int local_nThresholds = nThresholds[0];
		int local_nTaus = nTaus[0];
		int local_nEmbs = nEmbs[0];
		int local_nLmins = nLmins[0];
		int local_distance_type = distance_type[0];
		int local_calc_ENTR = calc_ENTR[0];
		Accrqa_Error error = SUCCESS;
		
		accrqa_DET(
			output, 
			input, 
			local_input_size, 
			tau_values,
			local_nTaus,
			emb_values,
			local_nEmbs,
			lmin_values,
			local_nLmins,
			threshold_values, 
			local_nThresholds, 
			local_distance_type,
			local_calc_ENTR,
			&error
		);
	}
	
	void R_double_accrqa_LAM(
		double *output, 
		double *input, 
		int    *input_size, 
		int    *tau_values,
		int    *nTaus,
		int    *emb_values,
		int    *nEmbs,
		int    *vmin_values,
		int    *nVmins,
		double *threshold_values, 
		int    *nThresholds, 
		int    *distance_type,
		int    *calc_ENTR
	){
		size_t local_input_size = (size_t) input_size[0];
		int local_nThresholds = nThresholds[0];
		int local_nTaus = nTaus[0];
		int local_nEmbs = nEmbs[0];
		int local_nVmins = nVmins[0];
		int local_distance_type = distance_type[0];
		int local_calc_ENTR = calc_ENTR[0];
		Accrqa_Error error = SUCCESS;
		
		accrqa_LAM(
			output, 
			input, 
			local_input_size, 
			tau_values,
			local_nTaus,
			emb_values,
			local_nEmbs,
			vmin_values,
			local_nVmins,
			threshold_values, 
			local_nThresholds, 
			local_distance_type,
			local_calc_ENTR,
			&error
		);
	}

	void R_double_accrqa_RR(
		double *output, 
		double *input, 
		int    *input_size, 
		int    *tau_values,
		int    *nTaus,
		int    *emb_values,
		int    *nEmbs,
		double *threshold_values, 
		int    *nThresholds, 
		int    *distance_type
	){
		size_t local_input_size = (size_t) input_size[0];
		int local_nThresholds = nThresholds[0];
		int local_nTaus = nTaus[0];
		int local_nEmbs = nEmbs[0];
		int local_distance_type = distance_type[0];
		Accrqa_Error error = SUCCESS;
		
		accrqa_RR(
			output,
			input,
			local_input_size,
			tau_values,
			local_nTaus,
			emb_values,
			local_nEmbs,
			threshold_values, 
			local_nThresholds, 
			local_distance_type,
			&error
		);
	}
	
	
#ifdef __cplusplus
}
#endif
