#ifdef ACCRQA_R_FOUND

#include <R.h>
#include "../include/AccRQA_library.hpp"
#include <R_ext/Rdynload.h>


Accrqa_CompPlatform check_comp_platform2(int int_comp_platform, Accrqa_Error *error){
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

Accrqa_Distance check_distance_type2(int int_distance_type, Accrqa_Error *error){
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
		int    *int_distance_type,
		int    *calc_ENTR, 
		int    *int_comp_platform
	){
		size_t local_input_size = (size_t) input_size[0];
		int local_nThresholds = nThresholds[0];
		int local_nTaus = nTaus[0];
		int local_nEmbs = nEmbs[0];
		int local_nLmins = nLmins[0];
		int local_int_distance_type = int_distance_type[0];
		int local_calc_ENTR = calc_ENTR[0];
		int local_int_comp_platform = int_comp_platform[0];
		Accrqa_Error error = SUCCESS;
	
		Accrqa_Distance distance_type = check_distance_type2(local_int_distance_type, &error);
		//std::cout << "Distance type: "<< distance_type << " " << error <<std::endl;
		if(error != SUCCESS)  return;

		Accrqa_CompPlatform comp_platform = check_comp_platform2(local_int_comp_platform, &error);
		//std::cout << "Computation platform: " << comp_platform << " " << error <<std::endl;
		if(error != SUCCESS) return;
		
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
			distance_type,
			local_calc_ENTR,
			comp_platform,
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
  	int    *int_distance_type,
  	int    *calc_ENTR,
  	int    *int_comp_platform
  ){
  	size_t local_input_size = (size_t) input_size[0];
  	int local_nThresholds = nThresholds[0];
  	int local_nTaus = nTaus[0];
  	int local_nEmbs = nEmbs[0];
  	int local_nVmins = nVmins[0];
  	int local_int_distance_type = int_distance_type[0];
  	int local_calc_ENTR = calc_ENTR[0];
  	int local_int_comp_platform = int_comp_platform[0];
  	Accrqa_Error error = SUCCESS;
  	
  	Accrqa_Distance distance_type = check_distance_type2(local_int_distance_type, &error);
  	//std::cout << "Distance type: "<< distance_type << " " << error <<std::endl;
  	if(error != SUCCESS)  return;
  	
  	Accrqa_CompPlatform comp_platform = check_comp_platform2(local_int_comp_platform, &error);
  	//std::cout << "Computation platform: " << comp_platform << " " << error <<std::endl;
  	if(error != SUCCESS) return;
  
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
  		distance_type,
  		local_calc_ENTR,
  		comp_platform,
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
  	int    *int_distance_type,
  	int    *int_comp_platform
  ){
  	size_t local_input_size = (size_t) input_size[0];
  	int local_nThresholds = nThresholds[0];
  	int local_nTaus = nTaus[0];
  	int local_nEmbs = nEmbs[0];
  	int local_int_distance_type = int_distance_type[0];
  	int local_int_comp_platform = int_comp_platform[0];
  	Accrqa_Error error = SUCCESS;
  
    Accrqa_Distance distance_type = check_distance_type2(local_int_distance_type, &error);
    //std::cout << "Distance type: "<< distance_type << " " << error <<std::endl;
    if(error != SUCCESS)  return;
  
    Accrqa_CompPlatform comp_platform = check_comp_platform2(local_int_comp_platform, &error);
    //std::cout << "Computation platform: " << comp_platform << " " << error <<std::endl;
    if(error != SUCCESS) return;
  	
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
  		distance_type,
  		comp_platform,
  		&error
  	);
  }
  
	void R_double_accrqa_RP(
		int *output,
		double *input,
		int    *input_size,
		int    *tau,
		int    *emb,
		double *threshold,
		int    *int_distance_type
	){
		size_t local_input_size = (size_t) input_size[0];
		double local_threshold = threshold[0];
		int local_tau = tau[0];
		int local_emb = emb[0];
		int local_int_distance_type = int_distance_type[0];
		Accrqa_Error error = SUCCESS;
		
		Accrqa_Distance distance_type = check_distance_type2(local_int_distance_type, &error);
		if(error != SUCCESS)  return;

		accrqa_RP(
			output,
			input,
			local_input_size,
			local_tau,
			local_emb,
			local_threshold,
			distance_type,
			&error
		);

	}

  R_CMethodDef cMethods[] = {
    {"R_double_accrqa_DET", (DL_FUNC) &R_double_accrqa_DET, 14},
    {"R_double_accrqa_LAM", (DL_FUNC) &R_double_accrqa_LAM, 14},
    {"R_double_accrqa_RR", (DL_FUNC) &R_double_accrqa_RR, 11},
    {"R_double_accrqa_RP", (DL_FUNC) &R_double_accrqa_RP, 7},
    {NULL, NULL, 0}
  };
  
  void R_init_AccRQA(DllInfo *info){
    R_registerRoutines(info, cMethods, NULL, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
  }	
	
#ifdef __cplusplus
}
#endif

#endif
