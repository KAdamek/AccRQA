#include "../include/AccRQA_definitions.hpp"
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
		return( RR );
	}
	
	void calculate_rqa_histogram_horizontal_CPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type) { // Laminarity
		rqa_CPU_LAM_metric_ref(metric.data(), scan_histogram.data(), length_histogram.data(), threshold, tau, emb, time_series, input_size, distance_type);
	}
	
	void calculate_rqa_histogram_horizontal_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type) { // Laminarity
		double execution_time = 0;
		
		int error = ACCRQA_SUCCESS;
		GPU_RQA_length_histogram_horizontal(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, &execution_time, &error);
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == ACCRQA_METRIC_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == ACCRQA_METRIC_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "LAM" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	void calculate_rqa_histogram_vertical_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type) { // Laminarity
		double execution_time = 0;
		
		int error = ACCRQA_SUCCESS;
		GPU_RQA_length_histogram_vertical(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, &execution_time, &error);
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == ACCRQA_METRIC_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == ACCRQA_METRIC_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "LAM" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	void calculate_rqa_histogram_diagonal_CPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type) { // Determinism
		rqa_CPU_DET_metric_ref(metric.data(), scan_histogram.data(), length_histogram.data(), threshold, tau, emb, time_series, input_size, distance_type);
	}
	
	void calculate_rqa_histogram_diagonal_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type) { 
		// Determinism
		double execution_time = 0;
		
		int error = ACCRQA_SUCCESS;
		GPU_RQA_length_histogram_diagonal(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, &execution_time, &error);
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == ACCRQA_METRIC_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == ACCRQA_METRIC_MAXIMAL) sprintf(metric, "maximal");
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
		
		void ProcessData_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type){
			this->calculate_rqa_histogram_diagonal_GPU(time_series, input_size, threshold, distance_type);
		}
		
		void ProcessData_CPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type){
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
		
		void ProcessData_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type){
			this->calculate_rqa_histogram_vertical_GPU(time_series, input_size, threshold, distance_type);
		}
		
		void ProcessData_CPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type){
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


void accrqa_print_error(int *error){
	switch(*error) {
		case ACCRQA_SUCCESS:
			printf("success");
			break;
		case ACCRQA_ERROR_CUDA_NOT_FOUND:
			printf("CUDA not found");
			break;
		case ACCRQA_ERROR_WRONG_ARGUMENTS:
			printf("wrong arguments");
			break;
		case ACCRQA_ERROR_ARRAYS_NOT_ALLOCATED:
			printf("array not allocated");
			break;
		case ACCRQA_ERROR_WRONG_METRIC_TYPE:
			printf("wrong metric type");
			break;
		case ACCRQA_ERROR_CUDA_DEVICE_NOT_FOUND:
			printf("CUDA device not found");
			break;
		case ACCRQA_ERROR_CUDA_NOT_ENOUGH_MEMORY:
			printf("not enough memory on device (GPU)");
			break;
		case ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION:
			printf("could not allocate memory on the device (GPU)");
			break;
		case ACCRQA_ERROR_CUDA_MEMORY_COPY:
			printf("could not copy memory host<->device");
			break;
		case ACCRQA_ERROR_CUDA_KERNEL:
			printf("CUDA kernel error");
			break;
		default:
			printf("unrecognised AccRQA error");
	}
}



//==========================================================
//========================= LAM ============================
//==========================================================

template<typename input_type>
void calculate_LAM_GPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, int distance_type, int *error){
	
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
					
					int pos = tau_id*nVmins*nEmbs*nThresholds + emb_id*nVmins*nThresholds + th_id*nVmins + v_id;
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
void calculate_LAM_CPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, int distance_type, int *error){
	
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
					
					int pos = tau_id*nVmins*nEmbs*nThresholds + emb_id*nVmins*nThresholds + th_id*nVmins + v_id;
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
void accrqa_LAM_GPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nVmins <= 0 || nThresholds <= 0) *error = ACCRQA_ERROR_WRONG_ARGUMENTS;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || vmin_values == NULL || threshold_values == NULL) *error = ACCRQA_ERROR_ARRAYS_NOT_ALLOCATED;
	if(*error!=ACCRQA_SUCCESS) return;
	
	// Default code
	calculate_LAM_GPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, error);

	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}

template<typename input_type>
void accrqa_LAM_CPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, input_type *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nVmins <= 0 || nThresholds <= 0) *error = ACCRQA_ERROR_WRONG_ARGUMENTS;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || vmin_values == NULL || threshold_values == NULL) *error = ACCRQA_ERROR_ARRAYS_NOT_ALLOCATED;
	if(*error!=ACCRQA_SUCCESS) return;
	
	// Default code
	calculate_LAM_CPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, error);
}



// Thin wrappers that call either accrqa_LAM_CPU_t or accrqa_LAM_GPU_t
void accrqa_LAM_GPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		accrqa_LAM_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}
void accrqa_LAM_GPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		accrqa_LAM_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}
void accrqa_LAM_CPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
}
void accrqa_LAM_CPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
}
void accrqa_LAM(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		printf("===> Doing GPU!!\n");
		accrqa_LAM_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		printf("===> Doing CPU!!\n");
		accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#endif
}
void accrqa_LAM(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *vmin_values, int nVmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		printf("===> Doing GPU!!\n");
		accrqa_LAM_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		printf("===> Doing CPU!!\n");
		accrqa_LAM_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, vmin_values, nVmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#endif
}

// Calculate output size
int accrqa_LAM_output_size_in_elements(int nTaus, int nEmbs, int nVmins, int nThresholds){
	return(nTaus*nEmbs*nVmins*nThresholds*5);
}



//==========================================================
//========================= DET ============================
//==========================================================

// Do we need access class?

template<typename input_type>
void calculate_DET_GPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, int distance_type, int *error){
	
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
					
					int pos = tau_id*nLmins*nEmbs*nThresholds + emb_id*nLmins*nThresholds + th_id*nLmins + l_id;
					output[5*pos + 0] = DETresults.DET(lmin);;
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
void calculate_DET_CPU_default(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, int distance_type, int *error){
	
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
void accrqa_DET_GPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nLmins <= 0 || nThresholds <= 0) *error = ACCRQA_ERROR_WRONG_ARGUMENTS;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || lmin_values == NULL || threshold_values == NULL) *error = ACCRQA_ERROR_ARRAYS_NOT_ALLOCATED;
	if(*error!=ACCRQA_SUCCESS) return;
	
	// Default code
	calculate_DET_GPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, error);

	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}

template<typename input_type>
void accrqa_DET_CPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, input_type *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	if(data_size == 0 || nTaus <= 0 || nEmbs <= 0 || nLmins <= 0 || nThresholds <= 0) *error = ACCRQA_ERROR_WRONG_ARGUMENTS;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || lmin_values == NULL || threshold_values == NULL) *error = ACCRQA_ERROR_ARRAYS_NOT_ALLOCATED;
	if(*error!=ACCRQA_SUCCESS) return;
	
	// Default code
	calculate_DET_CPU_default(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, error);

}

// Thin wrappers that call either accrqa_DET_CPU_t or accrqa_DET_GPU_t
void accrqa_DET_GPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		accrqa_DET_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}
void accrqa_DET_GPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		accrqa_DET_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}
void accrqa_DET_CPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
}
void accrqa_DET_CPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
}
void accrqa_DET(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, float *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		printf("===> Doing GPU!!\n");
		accrqa_DET_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		printf("===> Doing CPU!!\n");
		accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#endif
}
void accrqa_DET(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, int *lmin_values, int nLmins, double *threshold_values, int nThresholds, int distance_type, int calc_ENTR, int *error) {
	#ifdef CUDA_FOUND
		printf("===> Doing GPU!!\n");
		accrqa_DET_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#else
		printf("===> Doing CPU!!\n");
		accrqa_DET_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, lmin_values, nLmins, threshold_values, nThresholds, distance_type, calc_ENTR, error);
	#endif
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
	
	void calculate_RR_GPU(input_type *input, size_t input_size, int distance_type, int device){
		input_data_size = input_size;
		corrected_size = input_data_size - (c_emb - 1)*c_tau;
		int nThresholds = (int) rr_thresholds.size();
		double execution_time = 0;
		
		int error = ACCRQA_SUCCESS;
		GPU_RQA_RR_metric(rr_count.data(), input, input_size, rr_thresholds.data(), nThresholds, c_tau, c_emb, distance_type, device, &execution_time, &error);
		
		calculated = true;
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == ACCRQA_METRIC_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == ACCRQA_METRIC_MAXIMAL) sprintf(metric, "maximal");
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
	int distance_type
){
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			for(int th_id = 0; th_id < nThresholds; th_id++){
				input_type RR_value = 0;
				input_type threshold = threshold_values[th_id];
				
				rqa_CPU_RR_metric_ref(
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
	int distance_type
){
	for(int tau_id = 0; tau_id < nTaus; tau_id++){
		int tau = tau_values[tau_id];
		for(int emb_id = 0; emb_id < nEmbs; emb_id++){
			int emb = emb_values[emb_id];
			
			long int corrected_size = input_size - (emb - 1)*tau;
			std::vector<unsigned long long int> rr_count;
			rr_count.resize(nThresholds, 0);
			double execution_time = 0;
			int error = ACCRQA_SUCCESS;
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
void accrqa_RR_GPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, input_type *threshold_values, int nThresholds, int distance_type, int *error) {
	#ifdef CUDA_FOUND
	if(data_size == 0 || nTaus <=0 || nEmbs <=0 || nThresholds <=0) *error = ACCRQA_ERROR_WRONG_ARGUMENTS;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || threshold_values == NULL) *error = ACCRQA_ERROR_ARRAYS_NOT_ALLOCATED;
	if(*error!=ACCRQA_SUCCESS) return;
	
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
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}

template<typename input_type>
void accrqa_RR_CPU_t(input_type *output, input_type *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, input_type *threshold_values, int nThresholds, int distance_type, int *error) {
	if(data_size == 0 || nTaus <=0 || nEmbs <=0 || nThresholds <=0) *error = ACCRQA_ERROR_WRONG_ARGUMENTS;
	if(output == NULL || input_data == NULL || tau_values == NULL || emb_values == NULL || threshold_values == NULL) *error = ACCRQA_ERROR_ARRAYS_NOT_ALLOCATED;
	if(*error!=ACCRQA_SUCCESS) return;
	
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

// Thin wrappers that call either accrqa_RR_CPU_t or accrqa_RR_GPU_t
void accrqa_RR_GPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, float *threshold_values, int nThresholds, int distance_type, int *error) {
	#ifdef CUDA_FOUND
		accrqa_RR_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}
void accrqa_RR_GPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, double *threshold_values, int nThresholds, int distance_type, int *error) {
	#ifdef CUDA_FOUND
		accrqa_RR_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	#else
		*error = ACCRQA_ERROR_CUDA_NOT_FOUND;
	#endif
}
void accrqa_RR_CPU(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, float *threshold_values, int nThresholds, int distance_type, int *error) {
	accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
}
void accrqa_RR_CPU(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, double *threshold_values, int nThresholds, int distance_type, int *error) {
	accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
}
void accrqa_RR(float *output, float *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, float *threshold_values, int nThresholds, int distance_type, int *error) {
	#ifdef CUDA_FOUND
		printf("===> Doing GPU!!\n");
		accrqa_RR_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	#else
		printf("===> Doing CPU!!\n");
		accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	#endif
}
void accrqa_RR(double *output, double *input_data, size_t data_size, int *tau_values, int nTaus, int *emb_values, int nEmbs, double *threshold_values, int nThresholds, int distance_type, int *error) {
	#ifdef CUDA_FOUND
		printf("===> Doing GPU!!\n");
		accrqa_RR_GPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	#else
		printf("===> Doing CPU!!\n");
		accrqa_RR_CPU_t(output, input_data, data_size, tau_values, nTaus, emb_values, nEmbs, threshold_values, nThresholds, distance_type, error);
	#endif
}
int accrqa_RR_output_size_in_elements(int nTaus, int nEmbs, int nThresholds){
	return(nTaus*nEmbs*nThresholds);
}

//=============================>

// Python interface will decide based on numpy/cupy
// R interface will decide based on availability of a GPU
// C++ interface could be called directly? or based on availability
// For R and C++ interface it is assumed that CUDA_VISIBLE_​DEVICES will be set to something reasonable thus by default we use device 0



// Input: 
//  -- float *data; or double *data;
//  -- std::vector<int> tau;
//  -- std::vector<int> emb;
//  -- std::vector<float> threshold; or std::vector<double> threshold;
//  -- What to calculate will be determined by DET or LAM call and additional flag for entropy
//  
// Output:
//  -- Four dimensional array [tau][emb][threshold][M1,M2,M3,M4]. Where M1 = DET/LAM; M2 = L/TT; M3 = Lmax/TTmax; M4 = ENTR;


// How to do GPU thing?
// No idea at the moment.

