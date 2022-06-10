#include "AccRQA_definitions.h"
#include "AccRQA_GPU_function_wrappers.h"
#include "AccRQA_CPU_function.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

template<typename input_type>
class accrqaRecurrentRateResult {
private:
	std::vector<unsigned long long int> rr_count; /** Stores calculated values of recurrent rates */ 
	std::vector<input_type> rr_thresholds;        /** Keeps thresholds at which to calculate the recurrent rate */
	unsigned long long int input_data_size;       /** Number of element in the input data */
	unsigned long long int corrected_size;        /** Corrected number of elements which depend on the time step and embedding */
	bool calculated;                              /** Indicates if the recurrent values were computed */
	int c_tau;                                    /** Time step */
	int c_emb;                                    /** Embedding */
	
public:
	accrqaRecurrentRateResult(std::vector<input_type> &threshold_list, int tau, int emb){
		calculated = false;
		c_tau = tau;
		c_emb = emb;
		int nThresholds = threshold_list.size();
		rr_count.resize(nThresholds, 0);
		for(int f=0; f<(int) threshold_list.size(); f++){
			rr_thresholds.push_back(threshold_list[f]);
		}
	}
	
	void calculate_RR_GPU(input_type *input, size_t input_size, int distance_type, int device){
		input_data_size = input_size;
		corrected_size = input_data_size - (c_emb - 1)*c_tau;
		int nThresholds = (int) rr_thresholds.size();
		double execution_time = 0;
		
		GPU_RQA_RR_metric(rr_count.data(), input, input_size, rr_thresholds.data(), nThresholds, c_tau, c_emb, distance_type, device, &execution_time);
		
		calculated = true;
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == RQA_METRIC_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == RQA_METRIC_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << "0" << " " << (int) rr_thresholds.size() << " " << c_tau << " " << c_emb << " " << "1" << " " << metric << " " << "RR" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	void calculate_RR_CPU(input_type *input, size_t input_size, int distance_type){
		rqa_RR_metric(
				rr_count.data(), 
				rr_thresholds, 
				c_tau, 
				c_emb, 
				input, 
				input_size,
				distance_type
			);
		calculated = true;
	}
	
	size_t nThresholds(){
		return(rr_thresholds.size());
	}
	
	unsigned long long int getRRsum(size_t idx){
		return(rr_count[idx]);
	}
	
	bool getRR(input_type *RR, input_type *threshold, size_t idx){
		if(idx<rr_count.size() && calculated){
			*RR = ((double) rr_count[idx])/((double) (corrected_size*corrected_size));
			*threshold = rr_thresholds[idx];
			return(true);
		}
		else {
			return(false);
		}
	}
	
	void setTau(int tau){
		calculated = false;
		c_tau = tau;
	}
	
	void setEmb(int emb){
		calculated = false;
		c_emb = emb;
	}
};


//-------------------------------------------->
// This must have some tests of correctness of input variables
template<typename input_type>
void template_accrqaRecurrentRateGPU(
		input_type *RR, 
		input_type *thresholds, 
		int nThresholds, 
		input_type *input, 
		size_t input_size, 
		int tau, 
		int emb, 
		int distance_type, 
		int device
	){
	size_t corrected_size = input_size - (emb - 1)*tau;
	std::vector<unsigned long long int> rr_count;
	rr_count.resize(nThresholds, 0);
	double execution_time = 0;
	
	GPU_RQA_RR_metric(rr_count.data(), input, input_size, thresholds, nThresholds, tau, emb, distance_type, device, &execution_time);
	
	for(int idx = 0; idx < (int) rr_count.size(); idx++){
		RR[idx] = ((double) rr_count[idx])/((double) (corrected_size*corrected_size));
	}
	
	rr_count.clear();
}

void accrqaRecurrentRateGPU(float *RR, float *thresholds, int nThresholds, float *input, size_t input_size, int tau, int emb, int distance_type, int device) {
	template_accrqaRecurrentRateGPU<float>(RR, thresholds, nThresholds, input, input_size, tau, emb, distance_type, device);
}
void accrqaRecurrentRateGPU(double *RR, double *thresholds, int nThresholds, double *input, size_t input_size, int tau, int emb, int distance_type, int device) {
	template_accrqaRecurrentRateGPU<double>(RR, thresholds, nThresholds, input, input_size, tau, emb, distance_type, device);
}
//--------------------------------------------<
//-------------------------------------------->
// This must have some tests of correctness of input variables
template<typename input_type>
void template_accrqaRecurrentRateERGPU(
		input_type *RR, 
		input_type threshold, 
		input_type *input, 
		size_t input_size, 
		int tau, 
		int emb, 
		int distance_type, 
		int device
	){

	std::vector<unsigned long long int> rr_count;
	rr_count.resize(emb);
	double execution_time = 0;
	GPU_RQA_RR_ER_metric(rr_count.data(), input, input_size, threshold, tau, emb, distance_type, device, &execution_time);

	for(int k = 0; k < (int) rr_count.size(); k++){
	size_t corrected_size = input_size - (k)*tau;
	RR[k] = ((input_type) (2.0 * rr_count.data()[k]) + input_size - (k)*tau)/((input_type) (corrected_size*corrected_size)); // times the count by 2 and add the diagonal, then normalise
	
	}
	rr_count.clear();
}

void accrqaRecurrentRateERGPU(float *RR, float threshold, float *input, size_t input_size, int tau, int emb, int distance_type, int device) {
	template_accrqaRecurrentRateERGPU<float>(RR, threshold, input, input_size, tau, emb, distance_type, device);
}
void accrqaRecurrentRateERGPU(double *RR, double threshold, double *input, size_t input_size, int tau, int emb, int distance_type, int device) {
	template_accrqaRecurrentRateERGPU<double>(RR, threshold, input, input_size, tau, emb, distance_type, device);
}
//--------------------------------------------<

//-------------------------------------------->
template<typename input_type>
void template_accrqaRecurrentRateCPU(
		input_type *RR, 
		input_type *thresholds, 
		int nThresholds, 
		input_type *input, 
		size_t input_size, 
		int tau, 
		int emb, 
		int distance_type
	){
	size_t corrected_size = input_size - (emb - 1)*tau;
	std::vector<unsigned long long int> rr_count;
	rr_count.resize(nThresholds, 0);
	std::vector<input_type> rr_thresholds;
	rr_thresholds.resize(nThresholds, 0);
	for(int f=0; f<nThresholds; f++){
		rr_thresholds[f] = thresholds[f];
	}
	rqa_RR_metric(
		rr_count.data(), 
		rr_thresholds, 
		tau, 
		emb, 
		input, 
		input_size,
		distance_type
	);
	
	for(int idx = 0; idx < (int) rr_count.size(); idx++){
		RR[idx] = ((double) rr_count[idx])/((double) (corrected_size*corrected_size));
	}
	
	rr_count.clear();
}

void accrqaRecurrentRateCPU(float *RR, float *thresholds, int nThresholds, float *input, size_t input_size, int tau, int emb, int distance_type) {
	template_accrqaRecurrentRateCPU<float>(RR, thresholds, nThresholds, input, input_size, tau, emb, distance_type);
}
void accrqaRecurrentRateCPU(double *RR, double *thresholds, int nThresholds, double *input, size_t input_size, int tau, int emb, int distance_type) {
	template_accrqaRecurrentRateCPU<double>(RR, thresholds, nThresholds, input, input_size, tau, emb, distance_type);
}
//--------------------------------------------<
