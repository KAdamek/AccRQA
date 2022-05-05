#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

//TODO: osamostatnit rqaLengthHistogramResult do hpp

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <dirent.h>

#include "debug.h"
#include "results.h"
#include "AccRQA_library.hpp"

typedef double RQAdp;

bool UNIT_TESTS = false;

bool CPU_RQA = false;
bool CPU_RQA_RR = true;
bool CPU_RQA_DET = false;
bool CPU_RQA_LAM = false;

bool GPU_RQA = true;
bool GPU_RQA_RR = false;
bool GPU_RQA_DET = false;
bool GPU_RQA_LAM = false;
bool GPU_RQA_ALL = true;

//-----------------------------------------------
//---------- Data exports
//---------- NOT USED!
void Export_R_matrix(int *R_matrix, size_t size_rows, size_t size_columns, const char *filename){
	std::ofstream FILEOUT;
	FILEOUT.open(filename);
	for(size_t i = 0; i<size_rows; i++){
		for(size_t j = 0; j<size_columns; j++){
			size_t pos = i*size_columns + j;
			FILEOUT << R_matrix[pos] << " ";
		}
		FILEOUT << std::endl;
	}
	FILEOUT.close();
}

void print_LAM_histogram(std::vector<unsigned int> *TEST_histogram){
	printf("Histogram size: %zu;\n", TEST_histogram->size());
	for (size_t f = 0; f<TEST_histogram->size(); f++) {
		if (TEST_histogram->operator[](f)>0) {
			printf("length: %zu; count: %d;\n", f, TEST_histogram->operator[](f));
		}
	}	
}

void print_LAM_histogram(unsigned int *TEST_histogram, int size){
	for (int f = 0; f<size; f++) {
		if (TEST_histogram[f]>0) {
			printf("length: %d; count: %d;\n", f, TEST_histogram[f]);
		}
	}	
}

void write_to_file(accrqaRecurrentRateResult<RQAdp> *RR_class, const char *filename){
	bool no_error;
	std::ofstream FILEOUT;
	FILEOUT.open(filename);
	size_t size = RR_class->nThresholds();
	for(size_t i = 0; i<size; i++){
		RQAdp RR, th;
		no_error = RR_class->getRR(&RR, &th, i);
		if(no_error) FILEOUT << th << " " << RR << std::endl;
	}
	FILEOUT.close();
}
//------------------------------------------------<

//-----------------------------------------------
//---------- Data checks and generation
double max_error = 1.0e-4;

template<typename input_type>
double get_error(input_type iA, input_type iB){
	double error, div_error=10000, per_error=10000, order=0;
	double A = (double) iA;
	double B = (double) iB;
	int power;
	if(A<0) A = -A;
	if(B<0) B = -B;
	
	if (A>B) {
		div_error = A-B;
		if(B>10){
			power = (int) log10(B);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	else {
		div_error = B-A;
		if(A>10){
			power = (int) log10(A);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	
	if(div_error<per_error) error = div_error;
	else error = per_error;
	return(error);
}

size_t Compare_data(int *CPU_data, int *GPU_data, size_t size){
	size_t nErrors = 0;
	
	for(size_t i = 0; i<size; i++){
		for(size_t j = 0; j<size; j++){
			size_t pos = i*size + j;
			if(CPU_data[pos]!=GPU_data[pos]) nErrors++;
		}
	}
	return(nErrors);
}

size_t Compare_data(int *CPU_data, int *GPU_data, size_t rows, size_t columns){
	size_t nErrors = 0;
	
	for(size_t i = 0; i<rows; i++){
		for(size_t j = 0; j<columns; j++){
			size_t pos = i*columns + j;
			if(CPU_data[pos]!=GPU_data[pos]) nErrors++;
			if(CPU_data[pos]!=GPU_data[pos] && DEBUG && nErrors<25) {
				printf("Error at [%zu|%zu] CPU=%d; GPU=%d;\n", i, j, CPU_data[pos], GPU_data[pos]);
			}
		}
	}
	return(nErrors);
}

size_t Compare_data(size_t *CPU_data, size_t *GPU_data, size_t rows, size_t columns){
	size_t nErrors = 0;
	
	for(size_t i = 0; i<rows; i++){
		for(size_t j = 0; j<columns; j++){
			size_t pos = i*columns + j;
			if(CPU_data[pos]!=GPU_data[pos]) nErrors++;
		}
	}
	return(nErrors);
}

template <typename input_type>
size_t Compare_data_histograms(input_type *CPU_data, input_type *GPU_data, size_t size){
	size_t nErrors = 0;
	
	for(size_t i = 0; i<size; i++){
		if(CPU_data[i]!=GPU_data[i]) {
			nErrors++;
			long long int CPU = CPU_data[i];
			long long int GPU = GPU_data[i];
			long long int dif = CPU_data[i] - GPU_data[i];
			if(DEBUG) printf("Error in the histogram at %zu: CPU=%lld; GPU=%lld; difference=%lld;\n", i, CPU, GPU, dif);
		}
		
	}
	return(nErrors);
}

template<typename input_type>
int Generate_random(input_type *h_input, size_t size){
	for(size_t i = 0; i<size; i++){
		h_input[i] = rand() / (input_type)RAND_MAX;
	}
	return(0);
}

template<typename input_type>
int Generate_random(std::vector<input_type> *input){
	for(size_t i = 0; i<input->size(); i++){
		input->operator[](i) = rand() / (input_type)RAND_MAX;
	}
	return(0);
}

//---------------------------------------------------<




// ==================================================================
// ============================ CPU =================================
template <typename input_type>
double euclidean_distance(input_type *A, input_type *B, int emb){
	double sum = 0;
	for(int m=0; m<emb; m++){
		sum = sum + (A[m] - B[m])*(A[m] - B[m]);
	}
	return(sqrt(sum));
}


template <typename input_type>
double max_distance(input_type *A, input_type *B, int emb){
	double max = 0;
	for(int m=0; m<emb; m++){
		double dist = abs(A[m] - B[m]);
		if(dist > max) max = dist;
	}
	return(max);
}


template <typename input_type>
int Calculate_R_matrix(int *R_matrix, input_type *time_series, long int corrected_size, input_type threshold, int tau, int emb){
	input_type *A, *B;
	A = new input_type[emb];
	B = new input_type[emb];
	
	for(long int i = 0; i < corrected_size; i++){
		for(long int j = 0; j < corrected_size; j++){
			for(long int m = 0; m < emb; m++){
				A[m] = time_series[i + m*tau];
				B[m] = time_series[j + m*tau];
			}
			//double difference = euclidean_distance(A, B, emb);
			double difference = max_distance(A, B, emb);
			
			size_t r_pos = i*corrected_size + j;
			R_matrix[r_pos] = (difference < threshold ? 1 : 0 );
		}
	}
	
	delete[] A;
	delete[] B;
	return(0);
}


//==========================================================
//======================= Unit tests =======================
//==========================================================

//---------------- R-matrix unit test
int test_R_matrix(size_t size, RQAdp threshold, int tau, int emb, int device){
	std::vector<RQAdp> input(size, 0);
	Generate_random(&input);
	
	long int corrected_size = size - (emb - 1)*tau;
	std::vector<int> CPU_R_matrix(corrected_size*corrected_size, 0);
	std::vector<int> GPU_R_matrix(corrected_size*corrected_size, 0);
	
	double execution_time = 0;
	char metric[20]; sprintf(metric, "cartesian");
	Performance_results RQA_results(size, threshold, 1, tau, emb, 1, metric, "Rmatrix", "RQA_results.txt");
	GPU_RQA_R_matrix(GPU_R_matrix.data(), input.data(), input.size(), threshold, tau, emb, RQA_METRIC_MAXIMAL, device, &execution_time);
	RQA_results.GPU_time = execution_time;
	RQA_results.Save();
		
	int nErrors = 0;
	Calculate_R_matrix(CPU_R_matrix.data(), input.data(), size, threshold, tau, emb);
	
	nErrors = Compare_data(CPU_R_matrix.data(), GPU_R_matrix.data(), size);
	return(nErrors);
}

void unit_test_R_matrix(int device){
	printf("\n== R matrix unit test ==\n");
	printf("Testing different thresholds:"); fflush(stdout);
	int total_nErrors = 0;
	int nErrors = 0;
	int tau = 1;
	int emb = 1;
	for(RQAdp t=0; t<1.0; t = t + 0.1){
		nErrors = test_R_matrix(1000, t, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	printf("\n");
	printf("Testing different sizes:"); fflush(stdout);
	total_nErrors = 0;
	nErrors = 0;	
	for(size_t s = 1014; s < 32768; s = s*2){
		nErrors = test_R_matrix(1000, 0.33, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<

//---------------- diagonal R-matrix unit test
int test_diagonal_R_matrix(long int input_size, RQAdp threshold, int tau, int emb, int device){
	std::vector<RQAdp> input(input_size, 0);
	Generate_random(&input);
	
	int *CPU_diagonal_R_matrix, *GPU_diagonal_R_matrix;
	unsigned long long int corrected_size = input_size - (emb - 1)*tau;
	size_t size_rows = 2*corrected_size - 1;
	size_t size_columns = corrected_size;
	size_t matrix_size = size_rows*size_columns;
	size_t matrix_size_bytes = size_rows*size_columns*sizeof(int);
	CPU_diagonal_R_matrix = new int[matrix_size];
	GPU_diagonal_R_matrix = new int[matrix_size];
	memset(CPU_diagonal_R_matrix, 0, matrix_size_bytes);
	memset(GPU_diagonal_R_matrix, 0, matrix_size_bytes);
	
	rqa_R_matrix_diagonal(CPU_diagonal_R_matrix, input.data(), corrected_size, threshold, tau, emb, RQA_METRIC_MAXIMAL);
	
	double execution_time = 0;
	GPU_RQA_diagonal_R_matrix(GPU_diagonal_R_matrix, input.data(), threshold, tau, emb, input_size, RQA_METRIC_MAXIMAL, device, &execution_time);
	
	size_t nErrors = 0;
	nErrors = Compare_data(CPU_diagonal_R_matrix, GPU_diagonal_R_matrix, size_rows, size_columns);
	if(DEBUG) {
		if(nErrors==0) printf("     Comparison of diagonal R matrices:\033[1;32mPASSED\033[0m\n");
		else printf("     Comparison of diagonal R matrices:\033[1;31mFAILED\033[0m\n");
	}
	
	delete[] CPU_diagonal_R_matrix;
	delete[] GPU_diagonal_R_matrix;
	
	return(nErrors);
}

void unit_test_diagonal_R_matrix(int device){
	printf("\n== diagonal R matrix unit test ==\n");
	printf("Testing different thresholds:"); fflush(stdout);
	int total_nErrors = 0;
	int nErrors = 0;
	int tau = 1;
	int emb = 1;
	
	for(RQAdp t=0; t<1.0; t = t + 0.1){
		nErrors = test_diagonal_R_matrix(1000, t, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	printf("\n");
	printf("Testing different sizes:"); fflush(stdout);
	total_nErrors = 0;
	nErrors = 0;	
	for(size_t s = 1014; s < 32768; s = s*2){
		nErrors = test_diagonal_R_matrix(1000, 0.33, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<

//---------------- Recurrent rate unit test
int test_recurrent_rate(size_t input_size, RQAdp threshold_low, RQAdp threshold_high, RQAdp threshold_step, int tau, int emb, int device){
	std::vector<RQAdp> input(input_size, 0);
	Generate_random(&input);
	
	std::vector<RQAdp> threshold_list;
	for (RQAdp threshold = threshold_low; threshold < threshold_high; threshold = threshold + threshold_step) threshold_list.push_back(threshold);
	
	//-------> GPU
	accrqaRecurrentRateResult<RQAdp> GPU_RRresults(threshold_list, tau, emb);
	GPU_RRresults.calculate_RR_GPU(input.data(), input.size(), RQA_METRIC_MAXIMAL, device);
	
	//-------> CPU
	accrqaRecurrentRateResult<RQAdp> CPU_RRresults(threshold_list, tau, emb);
	CPU_RRresults.calculate_RR_CPU(input.data(), input.size(), RQA_METRIC_MAXIMAL);
	
	
	int nErrors = 0;
	size_t *GPU_results, *CPU_results;
	size_t nTresholds = threshold_list.size();
	GPU_results = new size_t[nTresholds];
	CPU_results = new size_t[nTresholds];
	for(size_t f=0; f < threshold_list.size(); f++){
		GPU_results[f] = (size_t) GPU_RRresults.getRRsum(f);
		CPU_results[f] = (size_t) CPU_RRresults.getRRsum(f);
	}
	nErrors = Compare_data(GPU_results, GPU_results, 1, nTresholds);
	if(nErrors>0 && DEBUG){
		printf("------> Errors detected:\n");
		for(size_t f=0; f<threshold_list.size(); f++){
			long long int CPU = CPU_results[f];
			long long int GPU = GPU_results[f];
			long long int dif = CPU - GPU;
			if(dif>0) printf("f=%zu; CPU:%lld; GPU:%lld; diff:%lld \n", f, CPU, GPU, dif);
		}
	}
	
	delete[] GPU_results;
	delete[] CPU_results;
	return(nErrors);
}

void unit_test_RR(int device){
	printf("\n== Recurrent rate unit test ==\n");
	int nErrors = 0, total_nErrors = 0;
	
	printf("Recurrent rate with different number of thresholds:"); fflush(stdout);
	nErrors = 0; total_nErrors = 0;
	for(int exp=1; exp<9; exp++){
		int nThresholds = (1<<exp);
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		RQAdp threshold_low = 0.1;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	
	printf("\n");
	printf("Recurrent rate with different sizes:"); fflush(stdout);
	nErrors = 0; total_nErrors = 0;	
	for(size_t s = 1014; s < 32768; s = s*2){
		int nThresholds = 10;
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		RQAdp threshold_low = 0;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	printf("\n");
	printf("Recurrent rate with different time steps and embeddings:"); fflush(stdout);
	nErrors = 0; total_nErrors = 0;
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			int nThresholds = 10;
			size_t size = 10000;
			RQAdp threshold_low = 0.1;
			RQAdp threshold_high = 1.0;
			RQAdp threshold_step = 1.0/((double) nThresholds);
			nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb, device);
			total_nErrors = total_nErrors + nErrors;
			printf("."); fflush(stdout);
		}
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<

//---------------- Histogram creation unit test
int test_histogram(int *input, int input_size, int device){
	size_t histogram_size = input_size + 1;
	std::vector<unsigned long long int> CPU_histogram(histogram_size, 0);
	std::vector<unsigned long long int> GPU_histogram(histogram_size, 0);

	get_length_histogram(CPU_histogram.data(), input, input_size);
		
	double execution_time = 0;
	GPU_RQA_length_start_end_test(GPU_histogram.data(), input, input_size, device, 1, &execution_time);

	int nErrors = 0;
	nErrors = Compare_data_histograms(CPU_histogram.data(), GPU_histogram.data(), histogram_size);
	if(DEBUG) {
		if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
		else printf("     Test:\033[1;31mFAILED\033[0m\n");	
	}
	return(nErrors);
	CPU_histogram.clear();
	GPU_histogram.clear();
}

void unit_test_histogram(int device){
	unsigned int input_size = 10000;
	int *input;
	input = new int[input_size];
	
	printf("\n== Histogram creation unit test ==\n");
	int nErrors = 0, total_nErrors = 0;
	
	for(unsigned int length_size = 1; length_size<input_size; length_size++){
		if(DEBUG) printf("Testing with length %d\n", length_size);
		memset(input, 0, input_size*sizeof(int));
		for (unsigned int f = 0; f<input_size; f++) {
			if (f%(length_size + 1)==0) {
				for(unsigned int i=0; i<length_size; i++){
					if( (f+i)<input_size ){
						input[f+i] = 1;
					}
				}
			}
		}
		
		nErrors = test_histogram(input, input_size, device);
		total_nErrors = total_nErrors + nErrors;
		
		if(length_size%100 == 0) {printf("."); fflush(stdout);}
	}
	
	if(DEBUG) printf("Testing with length %d\n", input_size);
	memset(input, 0, input_size*sizeof(int));
	for (unsigned int f = 0; f<input_size; f++) input[f]=1;
	nErrors = test_histogram(input, input_size, device);
	total_nErrors = total_nErrors + nErrors;
	
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
	
	delete[] input;
}
// ---------------------------------------------<


//---------------- DET and LAM unit test
int test_laminarity(long int input_size, RQAdp threshold, int tau, int emb, int device){
	std::vector<RQAdp> input(input_size, 0);
	Generate_random(&input);
	
	// GPU
	accrqaLaminarity<RQAdp> GPU_LAMresults(input_size, threshold, tau, emb);
	GPU_LAMresults.ProcessData_GPU(input.data(), input_size, threshold, RQA_METRIC_MAXIMAL, device);
	
	// CPU
	accrqaLaminarity<RQAdp> CPU_LAMresults(input_size, threshold, tau, emb);
	CPU_LAMresults.ProcessData_CPU(input.data(), input_size, threshold, RQA_METRIC_MAXIMAL);
	
	size_t histogram_size = GPU_LAMresults.getHistogramSize();
	
	unsigned long long int *GPU_histogram = GPU_LAMresults.getHistogram();
	unsigned long long int *CPU_histogram = CPU_LAMresults.getHistogram();
	int nErrors = 0;
	nErrors = Compare_data_histograms(CPU_histogram, GPU_histogram, histogram_size);
	if(DEBUG) {
		if(nErrors==0) printf("     Length histogram test:\033[1;32mPASSED\033[0m\n");
		else printf("     Length histogram test:\033[1;31mFAILED\033[0m\n");
	}
	
	unsigned long long int *GPU_histogram_scan = GPU_LAMresults.getHistogramScan();
	unsigned long long int *CPU_histogram_scan = CPU_LAMresults.getHistogramScan();
	nErrors = 0;
	if(DEBUG) printf("\n");
	nErrors += Compare_data_histograms(CPU_histogram_scan, GPU_histogram_scan, histogram_size);
	if(DEBUG) {
		if(nErrors==0) printf("     Length histogram after scan test:\033[1;32mPASSED\033[0m\n");
		else printf("     Length histogram after scan test:\033[1;31mFAILED\033[0m\n");
	}
	
	unsigned long long int *GPU_metric = GPU_LAMresults.getMetric();
	unsigned long long int *CPU_metric = CPU_LAMresults.getMetric();
	nErrors = 0;
	if(DEBUG) printf("\n");
	nErrors += Compare_data_histograms(CPU_metric, GPU_metric, histogram_size);
	if(DEBUG) {
		if(nErrors==0) printf("     Laminarity metric test:\033[1;32mPASSED\033[0m\n");
		else printf("     Laminarity metric test:\033[1;31mFAILED\033[0m\n");
	}

	return(nErrors);
}

int test_determinism(long int input_size, RQAdp threshold, int tau, int emb, int device){
	std::vector<RQAdp> input(input_size, 0);
	Generate_random(&input);
	
	// GPU
	accrqaDeterminism<RQAdp> GPU_DETresults(input_size, threshold, tau, emb);
	GPU_DETresults.ProcessData_GPU(input.data(), input_size, threshold, RQA_METRIC_MAXIMAL, device);
	
	// CPU
	accrqaDeterminism<RQAdp> CPU_DETresults(input_size, threshold, tau, emb);
	CPU_DETresults.ProcessData_CPU(input.data(), input_size, threshold, RQA_METRIC_MAXIMAL);
	
	size_t histogram_size = GPU_DETresults.getHistogramSize();
	
	unsigned long long int *GPU_histogram = GPU_DETresults.getHistogram();
	unsigned long long int *CPU_histogram = CPU_DETresults.getHistogram();
	int nErrors = 0;
	nErrors += Compare_data_histograms(CPU_histogram, GPU_histogram, histogram_size);
	if(DEBUG) {
		if(nErrors==0) printf("     Length histogram test:\033[1;32mPASSED\033[0m\n");
		else printf("     Length histogram test:\033[1;31mFAILED\033[0m\n");
	}
	
	unsigned long long int *GPU_histogram_scan = GPU_DETresults.getHistogramScan();
	unsigned long long int *CPU_histogram_scan = CPU_DETresults.getHistogramScan();
	if(DEBUG) printf("\n");
	nErrors += Compare_data_histograms(CPU_histogram_scan, GPU_histogram_scan, histogram_size);
	if(DEBUG) {
		if(nErrors==0) printf("     Length histogram after scan test:\033[1;32mPASSED\033[0m\n");
		else printf("     Length histogram after scan test:\033[1;31mFAILED\033[0m\n");
	}
	
	unsigned long long int *GPU_metric = GPU_DETresults.getMetric();
	unsigned long long int *CPU_metric = CPU_DETresults.getMetric();
	if(DEBUG) printf("\n");
	nErrors += Compare_data_histograms(CPU_metric, GPU_metric, histogram_size);
	if(DEBUG) {
		if(nErrors==0) printf("     Determinism test:\033[1;32mPASSED\033[0m\n");
		else printf("     Determinism test:\033[1;31mFAILED\033[0m\n");
	}

	return(nErrors);
}	
	
void unit_test_DET(int device){
	printf("\n== Determinism unit test ==\n");
	int nErrors = 0, total_nErrors = 0;

	std::vector<RQAdp> threshold_list;
	for(int t = 0; t <= 11; t++){
		threshold_list.push_back((RQAdp)t/10.0);
	}

	printf("Determinism with different number of thresholds:"); fflush(stdout);
	if(DEBUG) printf("\n");
	for(int t = 0; t<(int)threshold_list.size(); t++){
		RQAdp threshold = threshold_list[t];
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		if(DEBUG) printf("Testing with threshold=%f;\n", threshold);
		nErrors = test_determinism(size, threshold, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	
	nErrors = 0; total_nErrors = 0;
	printf("Determinism with different input sizes:"); fflush(stdout);
	if(DEBUG) printf("\n");
	for(size_t s = 1014; s < 32768; s = s*2){
		for(int t = 0; t<(int)threshold_list.size(); t++){
			RQAdp threshold = threshold_list[t];
			int tau = 1;
			int emb = 1;
			if(DEBUG) printf("Testing with size=%zu;\n", s);
			nErrors = test_determinism(s, threshold, tau, emb, device);
			total_nErrors = total_nErrors + nErrors;
			printf("."); fflush(stdout);
		}
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	

	nErrors = 0; total_nErrors = 0;
	printf("Determinism with different time steps and embeddings:"); fflush(stdout);
	if(DEBUG) printf("\n");
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			for(int t = 0; t<(int)threshold_list.size(); t++){
				RQAdp threshold = threshold_list[t];
				size_t size = 10000;
				if(DEBUG) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d\n", size, threshold, tau, emb);
				nErrors = test_determinism(size, threshold, tau, emb, device);
				total_nErrors = total_nErrors + nErrors;
				printf("."); fflush(stdout);
			}
		}
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}

void unit_test_LAM(int device){
	printf("\n== Laminarity unit test ==\n");
	int nErrors = 0, total_nErrors = 0;
	
	std::vector<RQAdp> threshold_list;
	for(int t = 0; t <= 11; t++){
		threshold_list.push_back((RQAdp)t/10.0);
	}
	
	nErrors = 0; total_nErrors = 0;
	printf("Laminarity with different number of thresholds:"); fflush(stdout);
	if(DEBUG) printf("\n");
	for(int t = 0; t<(int)threshold_list.size(); t++){
		RQAdp threshold = threshold_list[t];
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		if(DEBUG) printf("Testing with threshold=%f;\n", threshold);
		nErrors = test_laminarity(size, threshold, tau, emb, device);
		total_nErrors = total_nErrors + nErrors;
		printf("."); fflush(stdout);
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	
	nErrors = 0; total_nErrors = 0;
	printf("Laminarity with different input sizes:"); fflush(stdout);
	if(DEBUG) printf("\n");
	for(size_t s = 1014; s < 32768; s = s*2){
		for(int t = 0; t<(int)threshold_list.size(); t++){
			RQAdp threshold = threshold_list[t];
			int tau = 1;
			int emb = 1;
			if(DEBUG) printf("Testing with size=%zu and threshold=%f\n", s, threshold);
			nErrors = test_laminarity(s, threshold, tau, emb, device);
			total_nErrors = total_nErrors + nErrors;
			printf("."); fflush(stdout);
		}
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");


	nErrors = 0; total_nErrors = 0;
	printf("Laminarity with different time step and embedding:"); fflush(stdout);
	if(DEBUG) printf("\n");
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			for(int t = 0; t<(int)threshold_list.size(); t++){
				size_t size = 10000;
				RQAdp threshold = threshold_list[t];
				if(DEBUG) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d \n", size, threshold, tau, emb);
				nErrors = test_laminarity(size, threshold, tau, emb, device);
				total_nErrors = total_nErrors + nErrors;
				printf("."); fflush(stdout);
			}
		}
	}
	printf("\n");
	if(nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<

//==========================================================
//==========================================================
//==========================================================

int main(int argc, char* argv[]) {
		// all unit tests work
	if(UNIT_TESTS==true){
		unit_test_R_matrix(device);
		unit_test_diagonal_R_matrix(device);
		unit_test_RR(device);
		unit_test_histogram(device);
		unit_test_DET(device);
		unit_test_LAM(device);
	}
}