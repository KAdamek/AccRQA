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
		else primary_metric_value = -1.0;
		
		return(primary_metric_value);
	}
	
	double secondary_metric(size_t minimum_length) {
		double secondary_metric_value;
		
		if(scan_histogram[minimum_length]>0) secondary_metric_value = ((double) metric[minimum_length])/((double) scan_histogram[minimum_length]);
		else secondary_metric_value = -1.0;
		
		return(secondary_metric_value);
	}
	
	double tertiary_metric() {
		unsigned long long int last_element = 0;
		//binary search for last update
		
		// initial
		unsigned long long int start, mid, end, length;
		unsigned long long int mid_value, midm1_value;
		bool found = false;
		end = histogram_size - 1; 
		mid = histogram_size/2; 
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
				// search higher half
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
		// Then find place where it occurs first that is greatest length
		return(mid);
	}
	
	void calculate_rqa_histogram_horizontal_CPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type) { // Laminarity
		rqa_LAM_metric_CPU(metric.data(), scan_histogram.data(), length_histogram.data(), threshold, tau, emb, time_series, input_size, distance_type);
	}
	
	void calculate_rqa_histogram_horizontal_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type, int device) { // Laminarity
		double execution_time = 0;
		
		GPU_RQA_length_histogram_horizontal(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, device, &execution_time);
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == RQA_METRIC_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == RQA_METRIC_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "LAM" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	void calculate_rqa_histogram_diagonal_CPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type) { // Determinism
		rqa_DET_metric_CPU(metric.data(), scan_histogram.data(), length_histogram.data(), threshold, tau, emb, time_series, input_size, distance_type);
	}
	
	void calculate_rqa_histogram_diagonal_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type, int device) { 
		// Determinism
		double execution_time = 0;
		
		GPU_RQA_length_histogram_diagonal(length_histogram.data(), scan_histogram.data(), metric.data(), time_series, threshold, tau, emb, input_size, distance_type, device, &execution_time);
		
		#ifdef MONITOR_PERFORMANCE
		char metric[200]; 
		if(distance_type == RQA_METRIC_EUCLIDEAN) sprintf(metric, "euclidean");
		else if(distance_type == RQA_METRIC_MAXIMAL) sprintf(metric, "maximal");
		std::ofstream FILEOUT;
		FILEOUT.open ("RQA_results.txt", std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << input_size << " " << threshold << " " << "1" << " " << tau << " " << emb << " " << "1" << " " << metric << " " << "DET" << " " << execution_time << std::endl;
		FILEOUT.close();
		#endif
	}
	
	
public:
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
		
		void ProcessData_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type, int device){
			this->calculate_rqa_histogram_diagonal_GPU(time_series, input_size, threshold, distance_type, device);
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
			return(this->tertiary_metric());
		}
};

template<typename input_type>
class accrqaLaminarityResult : public accrqaLengthHistogramResult<input_type> {
	public:
		accrqaLaminarityResult(size_t input_size, input_type t_threshold, int t_tau, int t_emb) : accrqaLengthHistogramResult<input_type>(input_size, t_threshold, t_tau, t_emb){}
		
		void ProcessData_GPU(input_type *time_series, size_t input_size, input_type threshold, int distance_type, int device){
			this->calculate_rqa_histogram_horizontal_GPU(time_series, input_size, threshold, distance_type, device);
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
			return(this->tertiary_metric());
		}
};

//-------------------------------------------->
template<typename input_type>
void template_accrqaDeterminismGPU(
		input_type *DET, 
		input_type *L, 
		input_type *Lmax, 
		input_type *input_data, 
		size_t input_size, 
		input_type threshold, 
		int tau, 
		int emb,
		int lmin,
		int distance_type,
		int device
	){
	accrqaDeterminismResult<input_type> DETresults(input_size, threshold, tau, emb);
	DETresults.ProcessData_GPU(input_data, input_size, threshold, distance_type, device);
	*DET  = DETresults.DET(lmin);
	*L    = DETresults.L(lmin);
	*Lmax = DETresults.Lmax();
}

void accrqaDeterminismGPU(float *DET, float *L, float *Lmax, float *input_data, size_t input_size,  float threshold,  int tau, int emb, int lmin, int distance_type, int device) {
	template_accrqaDeterminismGPU<float>(DET, L, Lmax, input_data, input_size, threshold, tau, emb, lmin, distance_type, device);
}

void accrqaDeterminismGPU(double *DET, double *L, double *Lmax, double *input_data, size_t input_size,  double threshold,  int tau, int emb, int lmin, int distance_type, int device) {
	template_accrqaDeterminismGPU<double>(DET, L, Lmax, input_data, input_size, threshold, tau, emb, lmin, distance_type, device);
}
//--------------------------------------------<

//-------------------------------------------->
template<typename input_type>
void template_accrqaLaminarityGPU(
		input_type *LAM, 
		input_type *TT, 
		input_type *TTmax, 
		input_type *input_data, 
		size_t input_size, 
		input_type threshold, 
		int tau, 
		int emb,
		int vmin,
		int distance_type,
		int device
	){
	accrqaLaminarityResult<input_type> LAMresults(input_size, threshold, tau, emb);
	LAMresults.ProcessData_GPU(input_data, input_size, threshold, distance_type, device);
	*LAM   = LAMresults.LAM(vmin);
	*TT    = LAMresults.TT(vmin);
	*TTmax = LAMresults.TTmax();
}

void accrqaLaminarityGPU(float *LAM, float *TT, float *TTmax, float *input_data, size_t input_size,  float threshold,  int tau, int emb, int vmin, int distance_type, int device) {
	template_accrqaLaminarityGPU<float>(LAM, TT, TTmax, input_data, input_size, threshold, tau, emb, vmin, distance_type, device);
}

void accrqaLaminarityGPU(double *LAM, double *TT, double *TTmax, double *input_data, size_t input_size,  double threshold,  int tau, int emb, int vmin, int distance_type, int device) {
	template_accrqaLaminarityGPU<double>(LAM, TT, TTmax, input_data, input_size, threshold, tau, emb, vmin, distance_type, device);
}
//--------------------------------------------<

//-------------------------------------------->
template<typename input_type>
void template_accrqaDeterminismCPU(
		input_type *DET, 
		input_type *L, 
		input_type *Lmax, 
		input_type *input_data, 
		size_t input_size, 
		input_type threshold, 
		int tau, 
		int emb,
		int lmin,
		int distance_type
	){
	accrqaDeterminismResult<input_type> DETresults(input_size, threshold, tau, emb);
	DETresults.ProcessData_CPU(input_data, input_size, threshold, distance_type);
	*DET  = DETresults.DET(lmin);
	*L    = DETresults.L(lmin);
	*Lmax = DETresults.Lmax();
}

void accrqaDeterminismCPU(float *DET, float *L, float *Lmax, float *input_data, size_t input_size,  float threshold,  int tau, int emb, int lmin, int distance_type) {
	template_accrqaDeterminismCPU<float>(DET, L, Lmax, input_data, input_size, threshold, tau, emb, lmin, distance_type);
}

void accrqaDeterminismCPU(double *DET, double *L, double *Lmax, double *input_data, size_t input_size,  double threshold,  int tau, int emb, int lmin, int distance_type) {
	template_accrqaDeterminismCPU<double>(DET, L, Lmax, input_data, input_size, threshold, tau, emb, lmin, distance_type);
}
//--------------------------------------------<

//-------------------------------------------->
template<typename input_type>
void template_accrqaLaminarityCPU(
		input_type *LAM, 
		input_type *TT, 
		input_type *TTmax, 
		input_type *input_data, 
		size_t input_size, 
		input_type threshold, 
		int tau, 
		int emb,
		int vmin,
		int distance_type
	){
	accrqaLaminarityResult<input_type> LAMresults(input_size, threshold, tau, emb);
	LAMresults.ProcessData_CPU(input_data, input_size, threshold, distance_type);
	*LAM   = LAMresults.LAM(vmin);
	*TT    = LAMresults.TT(vmin);
	*TTmax = LAMresults.TTmax();
}

void accrqaLaminarityCPU(float *LAM, float *TT, float *TTmax, float *input_data, size_t input_size,  float threshold,  int tau, int emb, int vmin, int distance_type) {
	template_accrqaLaminarityCPU<float>(LAM, TT, TTmax, input_data, input_size, threshold, tau, emb, vmin, distance_type);
}

void accrqaLaminarityCPU(double *LAM, double *TT, double *TTmax, double *input_data, size_t input_size,  double threshold,  int tau, int emb, int vmin, int distance_type) {
	template_accrqaLaminarityCPU<double>(LAM, TT, TTmax, input_data, input_size, threshold, tau, emb, vmin, distance_type);
}
//--------------------------------------------<

// End


















