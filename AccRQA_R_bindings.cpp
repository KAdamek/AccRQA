#include <R.h>
#include "AccRQA_library.hpp"

extern "C" {
	void R_double_accrqaDeterminismGPU(
		double *DET, 
		double *L, 
		double *Lmax,
		double *ENTR,
		double *input_data, 
		int *input_size, 
		double *threshold, 
		int *tau, 
		int *emb,
		int *lmin,
		int *distance_type,
		int *device
	){
		size_t local_input_size = (size_t) input_size[0];
		double local_threshold = (double) threshold[0];
		int local_tau = tau[0];
		int local_emb = emb[0];
		int local_lmin = lmin[0];
		int local_distance_type = distance_type[0];
		int local_device = device[0];
		
		accrqaDeterminismGPU(
			DET, 
			L, 
			Lmax,
			ENTR,
			input_data, 
			local_input_size, 
			local_threshold, 
			local_tau, 
			local_emb,
			local_lmin,
			local_distance_type,
			local_device
		);
	}
}

extern "C" {	
	void R_double_accrqaLaminarityGPU(
		double *LAM, 
		double *TT, 
		double *TTmax,
		double *input_data, 
		long long int *input_size, 
		double *threshold, 
		int *tau, 
		int *emb,
		int *vmin,
		int *distance_type,
		int *device
	){
		size_t local_input_size = (size_t) input_size[0];
		double local_threshold = (double) threshold[0];
		int local_tau = tau[0];
		int local_emb = emb[0];
		int local_vmin = vmin[0];
		int local_distance_type = distance_type[0];
		int local_device = device[0];
		
		accrqaLaminarityGPU(
			LAM, 
			TT, 
			TTmax, 
			input_data, 
			local_input_size, 
			local_threshold, 
			local_tau, 
			local_emb,
			local_vmin,
			local_distance_type,
			local_device
		);
	}
}

extern "C" {
	void R_double_accrqaRecurrentRateGPU(
		double *RR, 
		double *input, 
		int    *input_size, 
		double *thresholds, 
		int    *nThresholds, 
		int    *tau, 
		int    *emb, 
		int    *distance_type,
		int    *device
	){
		int local_nThresholds = nThresholds[0];
		size_t local_input_size = (size_t) input_size[0];
		int local_tau = tau[0];
		int local_emb = emb[0];
		int local_distance_type = distance_type[0];
		int local_device = device[0];
		
		accrqaRecurrentRateGPU(
			RR, 
			thresholds, 
			local_nThresholds, 
			input, 
			local_input_size, 
			local_tau, 
			local_emb, 
			local_distance_type, 
			local_device
		);
	}
}