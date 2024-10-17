#include "../include/AccRQA_definitions.hpp"
#include "../include/AccRQA_utilities_error.hpp"
#include "../include/AccRQA_utilities_comp_platform.hpp"
#include "../include/AccRQA_utilities_distance.hpp"
#include "AccRQA_GPU_function.hpp"
#include "AccRQA_CPU_function.hpp"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>


//---> Internal classes?
//-------------------------------------------->
template<typename input_type>
class accrqaLengthHistogramResult {
protected:
	std::vector<unsigned long long int> length_histogram; /**  */
	std::vector<unsigned long long int> scan_histogram;
	std::vector<unsigned long long int> metric;
	long int histogram_size;
	long int input_data_size;
	long int corrected_size;
	input_type rqathreshold;
	int tau;
	int emb;
	
	explicit accrqaLengthHistogramResult(size_t input_size, input_type t_threshold, int t_tau, int t_emb) {
		tau = t_tau;
		emb = t_emb;
		input_data_size = input_size;
		corrected_size  = input_size - (emb - 1)*tau;
		histogram_size  = corrected_size + 1;
		length_histogram.resize(histogram_size, 0);
		scan_histogram.resize(histogram_size, 0);
		metric.resize(histogram_size, 0);
		rqathreshold = t_threshold;
	}
	
	double primary_metric(size_t minimum_length) {
		double primary_metric_value;
		
		if(metric[0]>0) primary_metric_value = ((double) metric[minimum_length])/((double) metric[0]);
		else primary_metric_value = NAN;
		
		if(ACCRQA_DEBUG_MODE) printf("primary metric: %e = %e/%e;\n", primary_metric_value, (double) metric[minimum_length], (double) metric[0]);
		
		return(primary_metric_value);
	}
	
	double secondary_metric(size_t minimum_length) {
		double secondary_metric_value;
		
		if(scan_histogram[minimum_length]>0) secondary_metric_value = ((double) metric[minimum_length])/((double) scan_histogram[minimum_length]);
		else secondary_metric_value = NAN;
		
		return(secondary_metric_value);
	}
	
	double tertiary_metric(int is_DET) { // Maximal line lenght
		unsigned long long int last_element;
		unsigned long long int start, mid, end, length;
		unsigned long long int mid_value, midm1_value;
		bool found = false;
		end = histogram_size - 1;
		mid = histogram_size/2;
		if(is_DET == 1) last_element = scan_histogram[end];
		else last_element = 0;
		start = 0;
		length = histogram_size;
		
		while(found==false){
			mid_value = scan_histogram[mid];
			midm1_value = scan_histogram[mid-1];
			if(mid_value == last_element && midm1_value > mid_value) {
				mid = mid - 1;
				found = true;
			}
			else if(length==0) {
				found = true;
				break;
			}
			else if(mid_value>last_element){
				// search upper half
				start = mid;
				length = end - start;
				mid = mid + (length + 1)/2;
			}
			else if(mid_value == last_element && midm1_value == last_element){
				// search lower half
				end = mid;
				length = end - start;
				
				mid = start + (length>>1);
			}
		}
		
		return(mid);
	}
	
	double quaternary_metric(size_t minimum_length) { // ENTR
		double entr = 0;
		for(long int f = minimum_length; f < histogram_size; f++) {
			double probability;
			probability = ((double) length_histogram[f])/((double) scan_histogram[f]);
			if(probability > 0) {
				entr = entr + probability*log(probability);
			}
		}
		return((-1.0)*entr);
	}
	
	double RR_value() { // ENTR
		double entr = 0;
		double RR = ((double) metric[0])/((double) (corrected_size*corrected_size));
		
		if(ACCRQA_DEBUG_MODE) printf("RR: %e = %e/%e;\n", RR, (double) metric[0], (double) (corrected_size*corrected_size));
		
		return( RR );
	}
	
	void calculate_rqa_histogram_horizontal_CPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type) { // Laminarity
		rqa_CPU_LAM_metric_ref(metric.data(), scan_histogram.data(), length_histogram.data(), threshold, tau, emb, time_series, input_size, distance_type);
	}
	
	void calculate_rqa_histogram_horizontal_GPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type) { // Laminarity
		double execution_time = 0;
		
		Accrqa_Error error = SUCCESS;
		GPU_RQA_length_histogram_horizontal(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, &execution_time, &error);
		
		#ifdef MONITOR_PERFORMANCE
		printf("LAM-default execution time: %fms;\n", execution_time);
		char metric[200]; 
		if(distance_type == DST_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == DST_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "LAM" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	void calculate_rqa_histogram_vertical_GPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type) { // Laminarity
		double execution_time = 0;
		
		Accrqa_Error error = SUCCESS;
		GPU_RQA_length_histogram_vertical(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, &execution_time, &error);
		
		#ifdef MONITOR_PERFORMANCE
		printf("LAM-default execution time: %fms;\n", execution_time);
		char metric[200]; 
		if(distance_type == DST_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == DST_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "LAM" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	void calculate_rqa_histogram_diagonal_CPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type) { // Determinism
		rqa_CPU_DET_metric_ref(metric.data(), scan_histogram.data(), length_histogram.data(), threshold, tau, emb, time_series, input_size, distance_type);
	}
	
	void calculate_rqa_histogram_diagonal_GPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type) { 
		// Determinism
		double execution_time = 0;
		
		Accrqa_Error error = SUCCESS;
		GPU_RQA_length_histogram_diagonal(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, &execution_time, &error);
		
		if(ACCRQA_DEBUG_MODE) {
			for(int f=0; f<10; f++){
				printf("-->time series: %e; metric: %lld; scan_histogram: %lld; length_histogram: %lld;\n", time_series[f], metric[f], scan_histogram[f], length_histogram[f]);
			}
		}
		
		#ifdef MONITOR_PERFORMANCE
		printf("DET-default execution time: %fms;\n", execution_time);
		char metric[200]; 
		if(distance_type == DST_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == DST_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "DETmk1" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	
public:
	~accrqaLengthHistogramResult(){
		length_histogram.clear();
		scan_histogram.clear();
		metric.clear();
	}
	
	double threshold(){
		return(rqathreshold);
	}
	
	unsigned long long int *getHistogram(){
		return(length_histogram.data());
	}
	
	unsigned long long int *getHistogramScan(){
		return(scan_histogram.data());
	}
	
	unsigned long long int *getMetric(){
		return(metric.data());
	}
	
	size_t getHistogramSize(){
		return((size_t) histogram_size);
	}
};


template<typename input_type>
class accrqaDeterminismResult : public accrqaLengthHistogramResult<input_type> {
	public:
		accrqaDeterminismResult(size_t input_size, input_type t_threshold, int t_tau, int t_emb) : accrqaLengthHistogramResult<input_type>(input_size, t_threshold, t_tau, t_emb){}
		
		void ProcessData_GPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type){
			this->calculate_rqa_histogram_diagonal_GPU(time_series, input_size, threshold, distance_type);
		}
		
		void ProcessData_CPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type){
			this->calculate_rqa_histogram_diagonal_CPU(time_series, input_size, threshold, distance_type);
		}
		
		double DET(size_t minimum_length) {
			return(this->primary_metric(minimum_length));
		}
		
		double L(size_t minimum_length) {
			return(this->secondary_metric(minimum_length));
		}
		
		double Lmax() {
			return(this->tertiary_metric(1));
		}
		
		double ENTR(size_t minimum_length) {
			return(this->quaternary_metric(minimum_length));
		}
		
		double RR() {
			return(this->RR_value());
		}
};


template<typename input_type>
class accrqaLaminarityResult : public accrqaLengthHistogramResult<input_type> {
	public:
		accrqaLaminarityResult(size_t input_size, input_type t_threshold, int t_tau, int t_emb) : accrqaLengthHistogramResult<input_type>(input_size, t_threshold, t_tau, t_emb){}
		
		void ProcessData_GPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type){
			this->calculate_rqa_histogram_vertical_GPU(time_series, input_size, threshold, distance_type);
		}
		
		void ProcessData_CPU(input_type *time_series, size_t input_size, input_type threshold, Accrqa_Distance distance_type){
			this->calculate_rqa_histogram_horizontal_CPU(time_series, input_size, threshold, distance_type);
		}
	
		double LAM(size_t minimum_length) {
			return(this->primary_metric(minimum_length));
		}
		
		double TT(size_t minimum_length) {
			return(this->secondary_metric(minimum_length));
		}
		
		double TTmax(){
			return(this->tertiary_metric(0));
		}
		
		double ENTR(size_t minimum_length){
			return(this->quaternary_metric(minimum_length));
		}
		
		double RR(){
			return(this->RR_value());
		}
};
//--------------------------------------------<


void accrqa_print_error(Accrqa_Error *error){
	switch(*error) {
		case SUCCESS:
			printf("success");
			break;
		case ERR_RUNTIME:
			printf("generic runtime error");
			break;
		case ERR_INVALID_ARGUMENT:
			printf("wrong arguments");
			break;
		case ERR_DATA_TYPE:
			printf("unsupported data type");
			break;
		case ERR_MEM_ALLOC_FAILURE:
			printf("array not allocated");
			break;
		case ERR_MEM_COPY_FAILURE:
			printf("memory copy failure host<->device");
			break;
		case ERR_MEM_LOCATION:
			printf("wrong memory location");
			break;
		case ERR_CUDA_NOT_FOUND:
			printf("CUDA not found");
			break;
		case ERR_CUDA_DEVICE_NOT_FOUND:
			printf("could not locate CUDA device (GPU)");
			break;
		case ERR_CUDA_NOT_ENOUGH_MEMORY:
			printf("not enough memory on the device (GPU)");
			break;
		case ERR_CUDA:
			printf("other CUDA error");
			break;
		case ERR_INVALID_METRIC_TYPE:
			printf("invalid metric selected");
			break;
		default:
			printf("unrecognised AccRQA error");
	}
}


//==========================================================
//========================= LAM ============================
//==========================================================

template<typename input_type>
void calculate_LAM_GPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_Error *error){
	
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			for(int th_id = 0; th_id < nThresholds; th_id++){
				input_type threshold = threshold_values[th_id];
				
				accrqaLaminarityResult<input_type> LAMresults(data_size, threshold, tau, emb);
				LAMresults.ProcessData_GPU(input_data, data_size, threshold, distance_type);
				
				for(int v_id = 0; v_id < nVmins; v_id++){
					int vmin = vmin_values[v_id];
					
					int pos = tau_id*nVmins*nEmbs*nThresholds + emb_id*nVmins*nThresholds + v_id*nThresholds + th_id;
					output[5*pos + 0] = LAMresults.LAM(vmin);
					output[5*pos + 1] = LAMresults.TT(vmin);
					output[5*pos + 2] = LAMresults.TTmax();
					output[5*pos + 3] = LAMresults.ENTR(vmin);
					output[5*pos + 4] = LAMresults.RR();
				}
			}
		}
	}
	
}

template<typename input_type>
void calculate_LAM_CPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_Error *error){
	
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			for(int th_id = 0; th_id < nThresholds; th_id++){
				input_type threshold = threshold_values[th_id];

				accrqaLaminarityResult<input_type> LAMresults(data_size, threshold, tau, emb);
				LAMresults.ProcessData_CPU(input_data, data_size, threshold, distance_type);
				
				for(int v_id = 0; v_id < nVmins; v_id++){
					int vmin = vmin_values[v_id];
					
					int pos = tau_id*nVmins*nEmbs*nThresholds + emb_id*nVmins*nThresholds + v_id*nThresholds + th_id;
					output[5*pos + 0] = LAMresults.LAM(vmin);
					output[5*pos + 1] = LAMresults.TT(vmin);
					output[5*pos + 2] = LAMresults.TTmax();
					output[5*pos + 3] = LAMresults.ENTR(vmin);
					output[5*pos + 4] = LAMresults.RR();
				}
			}
		}
	}
	
}


template<typename input_type>
void accrqa_LAM_GPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_Error *error) {
	*error = SUCCESS;
	
	#ifdef CUDA_FOUND
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nVmins <= 0 || nThresholds <= 0) *error = ERR_INVALID_ARGUMENT;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || vmin_values == NULL || threshold_values == NULL) *error = ERR_MEM_ALLOC_FAILURE;
	if(*error!=SUCCESS) return;
	
	// Default code
	calculate_LAM_GPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, error);

	#else
		*error = ERR_CUDA_NOT_FOUND;
	#endif
}

template<typename input_type>
void accrqa_LAM_CPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_Error *error) {
	*error = SUCCESS;
	
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nVmins <= 0 || nThresholds <= 0) *error = ERR_INVALID_ARGUMENT;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || vmin_values == NULL || threshold_values == NULL) *error = ERR_MEM_ALLOC_FAILURE;
	if(*error!=SUCCESS) return;
	
	// Default code
	calculate_LAM_CPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, error);
}

void accrqa_LAM(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, float *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform, Accrqa_Error *error) {
	if(comp_platform==PLT_NV_GPU){
		#ifdef CUDA_FOUND
			printf("===> Doing GPU!!\n");
			accrqa_LAM_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#else
			printf("===> Doing CPU!!\n");
			accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#endif
	}
	else {
		if(comp_platform!=PLT_CPU) printf("WARNING: Unknown compute platform. Defaulting to CPU.\n");
		printf("===> Doing CPU!!\n");
		accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	}
}

void accrqa_LAM(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, double *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform, Accrqa_Error *error) {
	if(comp_platform==PLT_NV_GPU){
		#ifdef CUDA_FOUND
			printf("===> Doing GPU!!\n");
			accrqa_LAM_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#else
			printf("WARNING: CUDA capable device not found. Defaulting to CPU.\n");
			printf("===> Doing CPU!!\n");
			accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#endif
	}
	else {
		if(comp_platform!=PLT_CPU) printf("WARNING: Unknown compute platform. Defaulting to CPU.\n");
		printf("===> Doing CPU!!\n");
		accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	}
}

// Calculate output size
int accrqa_LAM_output_size_in_elements(int nTaus, int nEmbs, int nVmins, int nThresholds){
	return(nTaus*nEmbs*nVmins*nThresholds*5);
}



//==========================================================
//========================= DET ============================
//==========================================================

template<typename input_type>
void calculate_DET_GPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_Error *error){
	
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			for(int th_id = 0; th_id < nThresholds; th_id++){
				input_type threshold = threshold_values[th_id];

				accrqaDeterminismResult<input_type> DETresults(data_size, threshold, tau, emb);
				DETresults.ProcessData_GPU(input_data, data_size, threshold, distance_type);
				
				for(int l_id = 0; l_id < nLmins; l_id++){
					int lmin = lmin_values[l_id];
					
					int pos = tau_id*nLmins*nEmbs*nThresholds + emb_id*nLmins*nThresholds + l_id*nThresholds + th_id;
					output[5*pos + 0] = DETresults.DET(lmin);
					output[5*pos + 1] = DETresults.L(lmin);
					output[5*pos + 2] = DETresults.Lmax();
					output[5*pos + 3] = DETresults.ENTR(lmin);
					output[5*pos + 4] = DETresults.RR();
				}
			}
		}
	}
	
}

template<typename input_type>
void calculate_DET_GPU_sum(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_Error *error){
	
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			for(int th_id = 0; th_id < nThresholds; th_id++){
				input_type threshold = threshold_values[th_id];
				for(int l_id = 0; l_id < nLmins; l_id++){
					int lmin = lmin_values[l_id];
					
					input_type h_DET = 0, h_L = 0, h_RR = 0;
					unsigned long long int h_Lmax = 0;
					double execution_time = 0;
					GPU_RQA_length_histogram_diagonal_sum(
						&h_DET, &h_L, &h_Lmax, &h_RR,
						input_data,
						threshold, tau, emb, lmin, 
						data_size, distance_type, 
						&execution_time, error
					);
					int pos = tau_id*nLmins*nEmbs*nThresholds + emb_id*nLmins*nThresholds + l_id*nThresholds + th_id;
					output[5*pos + 0] = h_DET;
					output[5*pos + 1] = h_L;
					output[5*pos + 2] = h_Lmax;
					output[5*pos + 3] = 0;
					output[5*pos + 4] = h_RR;
					
					#ifdef MONITOR_PERFORMANCE
					printf("DET-sum execution time: %fms;\n", execution_time);
					char metric[200]; 
					if(distance_type == DST_EUCLIDEAN) sprintf(metric, "euclidean");
					else if(distance_type == DST_MAXIMAL) sprintf(metric, "maximal");
					std::ofstream FILEOUT;
					FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
					FILEOUT << std::fixed << std::setprecision(8) << data_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "DETsum" << " " << execution_time << std::endl;
					FILEOUT.close();
					#endif
				}
			}
		}
	}
	
}

template<typename input_type>
void calculate_DET_CPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_Error *error){
	
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			for(int th_id = 0; th_id < nThresholds; th_id++){
				input_type threshold = threshold_values[th_id];

				accrqaDeterminismResult<input_type> DETresults(data_size, threshold, tau, emb);
				DETresults.ProcessData_CPU(input_data, data_size, threshold, distance_type);
				
				for(int l_id = 0; l_id < nLmins; l_id++){
					int lmin = lmin_values[l_id];
					
					int pos = tau_id*nLmins*nEmbs*nThresholds + emb_id*nLmins*nThresholds + th_id*nLmins + l_id;
					output[5*pos + 0] = DETresults.DET(lmin);
					output[5*pos + 1] = DETresults.L(lmin);
					output[5*pos + 2] = DETresults.Lmax();
					output[5*pos + 3] = DETresults.ENTR(lmin);
					output[5*pos + 4] = DETresults.RR();
				}
			}
		}
	}
	
}

template<typename input_type>
void accrqa_DET_GPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_Error *error) {
	*error = SUCCESS;
	
	#ifdef CUDA_FOUND
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nLmins <= 0 || nThresholds <= 0) *error = ERR_INVALID_ARGUMENT;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || lmin_values == NULL || threshold_values == NULL) *error = ERR_MEM_ALLOC_FAILURE;
	if(*error!=SUCCESS) return;
	
	// Default code
	calculate_DET_GPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, error);
	//calculate_DET_GPU_sum(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, error);

	#else
		*error = ERR_CUDA_NOT_FOUND;
	#endif
}

template<typename input_type>
void accrqa_DET_CPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_Error *error) {
	*error = SUCCESS;
	
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nLmins <= 0 || nThresholds <= 0) *error = ERR_INVALID_ARGUMENT;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || lmin_values == NULL || threshold_values == NULL) *error = ERR_MEM_ALLOC_FAILURE;
	if(*error!=SUCCESS) return;
	
	// Default code
	calculate_DET_CPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, error);

}

void accrqa_DET(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, float *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform, Accrqa_Error *error) {
	if(comp_platform==PLT_NV_GPU){
		#ifdef CUDA_FOUND
			printf("===> Doing GPU!!\n");
			accrqa_DET_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#else
			printf("WARNING: CUDA capable device not found. Defaulting to CPU.\n");
			printf("===> Doing CPU!!\n");
			accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#endif
	}
	else {
		if(comp_platform!=PLT_CPU) printf("WARNING: Unknown compute platform. Defaulting to CPU.\n");
		printf("===> Doing CPU!!\n");
		accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	}
}

void accrqa_DET(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, double *threshold_values, int nThresholds, Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform, Accrqa_Error *error) {
	if(comp_platform==PLT_NV_GPU){
		#ifdef CUDA_FOUND
			printf("===> Doing GPU!!\n");
			accrqa_DET_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#else
			printf("WARNING: CUDA capable device not found. Defaulting to CPU.\n");
			printf("===> Doing CPU!!\n");
			accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
		#endif
	}
	else {
		if(comp_platform!=PLT_CPU) printf("WARNING: Unknown compute platform. Defaulting to CPU.\n");
		printf("===> Doing CPU!!\n");
		accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	}
}

int accrqa_DET_output_size_in_elements(int nTaus, int nEmbs, int nLmins, int nThresholds){
	return(nTaus*nEmbs*nLmins*nThresholds*5);
}

//==========================================================
//========================== RR ============================
//==========================================================

// This class needs to be converted into class for accessing results 
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
	
	void calculate_RR_GPU(input_type *input, size_t input_size, Accrqa_Distance distance_type, int device){
		input_data_size = input_size;
		corrected_size = input_data_size - (c_emb - 1)*c_tau;
		int nThresholds = (int) rr_thresholds.size();
		double execution_time = 0;
		
		Accrqa_Error error = SUCCESS;
		GPU_RQA_RR_metric(rr_count.data(), input, input_size, rr_thresholds.data(), nThresholds, c_tau, c_emb, distance_type, device, &execution_time, &error);
		
		calculated = true;
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == DST_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == DST_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << "0" << " " << (int) rr_thresholds.size() << " " << c_tau << " " << c_emb << " " << "1" << " " << metric << " " << "RR" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	void calculate_RR_CPU(input_type *input, size_t input_size, Accrqa_Distance distance_type){
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

template<typename input_type>
void calculate_RR_CPU_default(
	input_type *RR, 
	input_type *input_data,
	size_t input_size,
	int *tau_values, 
	int nTaus, 
	int *emb_values, 
	int nEmbs, 
	input_type *threshold_values,
	int nThresholds, 
	Accrqa_Distance distance_type
){
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			for(int th_id = 0; th_id < nThresholds; th_id++){
				input_type RR_value = 0;
				input_type threshold = threshold_values[th_id];
				
				rqa_CPU_RR_metric_ref_parallel(
					&RR_value, 
					input_data, 
					(unsigned long long int) input_size, 
					threshold, 
					tau, 
					emb, 
					distance_type
				);
				
				int pos = tau_id*nEmbs*nThresholds + emb_id*nThresholds + th_id;
				RR[pos] = RR_value;
			}
		}
	}
}

template<typename input_type>
void calculate_RR_GPU_default(
	input_type *RR, 
	input_type *input_data,
	size_t input_size,
	int *tau_values, 
	int nTaus, 
	int *emb_values, 
	int nEmbs, 
	input_type *threshold_values,
	int nThresholds, 
	Accrqa_Distance distance_type
){
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			
			long int corrected_size = input_size - (emb - 1)*tau;
			std::vector<unsigned long long int> rr_count;
			rr_count.resize(nThresholds, 0);
			double execution_time = 0;
			Accrqa_Error error = SUCCESS;
			GPU_RQA_RR_metric_integer(rr_count.data(), input_data, input_size, threshold_values, nThresholds, tau, emb, distance_type, &execution_time, &error);
	
			for(int th_id = 0; th_id < nThresholds; th_id++){
				int pos = tau_id*nEmbs*nThresholds + emb_id*nThresholds + th_id;
				RR[pos] = (input_type) ((double) rr_count[th_id])/((double) (corrected_size*corrected_size));
			}
			
			rr_count.clear();
		}
	}
}


template<typename input_type>
void accrqa_RR_GPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_Error *error) {
	*error = SUCCESS;
	
	#ifdef CUDA_FOUND
	if(data_size == 0 || nTaus <=0 || nEmbs <=0 || nThresholds <=0) *error = ERR_INVALID_ARGUMENT;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || threshold_values == NULL) *error = ERR_MEM_ALLOC_FAILURE;
	if(*error!=SUCCESS) return;
	
	// Default code
	calculate_RR_GPU_default(
		output, 
		input_data,
		data_size,
		tau_values, 
		nTaus, 
		emb_values, 
		nEmbs, 
		threshold_values,
		nThresholds, 
		distance_type
	);
	#else
		*error = ERR_CUDA_NOT_FOUND;
	#endif
}


template<typename input_type>
void accrqa_RR_CPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, input_type *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_Error *error) {
	*error = SUCCESS;
	
	if(data_size == 0 || nTaus <=0 || nEmbs <=0 || nThresholds <=0) *error = ERR_INVALID_ARGUMENT;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || threshold_values == NULL) *error = ERR_MEM_ALLOC_FAILURE;
	if(*error!=SUCCESS) return;
	
	// Default code
	calculate_RR_CPU_default(
		output, 
		input_data,
		data_size,
		tau_values, 
		nTaus, 
		emb_values, 
		nEmbs, 
		threshold_values,
		nThresholds, 
		distance_type
	);
}


void accrqa_RR(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, float *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_CompPlatform comp_platform, Accrqa_Error *error) {
	if(comp_platform==PLT_NV_GPU){
		#ifdef CUDA_FOUND
			printf("===> Doing GPU!!\n");
			accrqa_RR_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
		#else
			printf("WARNING: CUDA capable device not found. Defaulting to CPU.\n");
			printf("===> Doing CPU!!\n");
			accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
		#endif
	}
	else {
		if(comp_platform!=PLT_CPU) printf("WARNING: Unknown compute platform. Defaulting to CPU.\n");
		printf("===> Doing CPU!!\n");
		accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	}
}

void accrqa_RR(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, double *threshold_values, int nThresholds, Accrqa_Distance distance_type, Accrqa_CompPlatform comp_platform, Accrqa_Error *error) {
	if(comp_platform==PLT_NV_GPU){
		#ifdef CUDA_FOUND
			printf("===> Doing GPU!!\n");
			accrqa_RR_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
		#else
			printf("WARNING: CUDA capable device not found. Defaulting to CPU.\n");
			printf("===> Doing CPU!!\n");
			accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
		#endif
	}
	else {
		if(comp_platform!=PLT_CPU) printf("WARNING: Unknown compute platform. Defaulting to CPU.\n");
		printf("===> Doing CPU!!\n");
		accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	}
}


int accrqa_RR_output_size_in_elements(
	int nTaus, int nEmbs, int nThresholds
) {
	return(nTaus*nEmbs*nThresholds);
}

//=============================<



