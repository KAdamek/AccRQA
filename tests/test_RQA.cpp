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

#include <dirent.h>

#include "../include/AccRQA_library.hpp"

typedef double RQAdp;

bool TEST_RQA_RECURRENT_MATRIX = false;
bool TEST_RQA_RECURRENT_MATRIX_DIAGONAL = false;
bool TEST_RQA_DET = false;
bool TEST_RQA_LAM = false;

bool DEBUG_MODE = false;
bool CHECK = true;
bool GPU_UNIT_TEST = true;
bool CPU_UNIT_TEST = false;
bool RR_EXTENDED_TEST = true;

//-----------------------------------------------
//---------- Data checks and generation
double max_error = 1.0e-3;

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
			if(CPU_data[pos]!=GPU_data[pos]) {
				if(DEBUG_MODE && nErrors<20) {
					printf("Reference: %d; Tested: %d; pos:[%zu;%zu];\n", CPU_data[pos], GPU_data[pos], i, j);
				}
				nErrors++;
			}
		}
	}
	return(nErrors);
}

size_t Compare_data(int *CPU_data, int *GPU_data, size_t rows, size_t columns){
	size_t nErrors = 0;
	
	for(size_t i = 0; i<rows; i++){
		for(size_t j = 0; j<columns; j++){
			size_t pos = i*columns + j;
			if(CPU_data[pos]!=GPU_data[pos]) {
				nErrors++;
				if(DEBUG_MODE && nErrors<20) {
					printf("Error at [%zu|%zu] CPU=%d; GPU=%d;\n", i, j, CPU_data[pos], GPU_data[pos]);
				}
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
size_t Compare_data(input_type *CPU_data, input_type *GPU_data, size_t size){
	size_t nErrors = 0;
	
	
	for(size_t f = 0; f<size; f++){
		double error = get_error(CPU_data[f], GPU_data[f]);
		if(error > max_error) {
			nErrors++;
			if(DEBUG_MODE && nErrors<20) {
				printf("Error at %zu: CPU %e; GPU: %e;\n", f, CPU_data[f], GPU_data[f]);
			}
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
			if(DEBUG_MODE) printf("Error in the histogram at %zu: CPU=%lld; GPU=%lld; difference=%lld;\n", i, CPU, GPU, dif);
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




//---------------- Recurrent rate unit test
int test_recurrent_rate(size_t input_size, RQAdp threshold_low, RQAdp threshold_high, RQAdp threshold_step, int tau, int emb){
	std::vector<RQAdp> input_data(input_size, 0);
	Generate_random(&input_data);
	
	std::vector<RQAdp> threshold_list;
	for (RQAdp threshold = threshold_low; threshold < threshold_high; threshold = threshold + threshold_step) threshold_list.push_back(threshold);
	int nThresholds = (int) threshold_list.size(); 
	
	RQAdp *CPU_RR_result, *GPU_RR_result, *GPU_DET_RR_result, *GPU_LAM_RR_result;
	int RR_output_size = accrqa_RR_output_size_in_elements(1, 1, threshold_list.size());
	CPU_RR_result = new RQAdp[nThresholds];
	GPU_RR_result = new RQAdp[nThresholds];
	GPU_DET_RR_result = new RQAdp[nThresholds];
	GPU_LAM_RR_result = new RQAdp[nThresholds];

	int tau_values = tau;
	int emb_values = emb;
	Accrqa_Error error;
	
	//-------> GPU
	if(GPU_UNIT_TEST) {
		accrqa_RR_GPU(GPU_RR_result, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, ACCRQA_METRIC_MAXIMAL, &error);
	}
	
	//-------> CPU
	accrqa_RR_CPU(CPU_RR_result, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, ACCRQA_METRIC_MAXIMAL, &error);
	
	//-------> GPU LAM
	if(RR_EXTENDED_TEST){
		RQAdp *output_GPU;
		int output_size = accrqa_LAM_output_size_in_elements(1, 1, 1, 1);
		output_GPU = new RQAdp[output_size];
		int vmin_values = 2;
		int calc_ENTR = 1;
		for(size_t th_idx = 0; th_idx < threshold_list.size(); th_idx++){
			RQAdp threshold_values = threshold_list[th_idx];
			accrqa_LAM_GPU(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, ACCRQA_METRIC_MAXIMAL, calc_ENTR, &error);
			GPU_LAM_RR_result[th_idx] = output_GPU[4];
		}
		delete[] output_GPU;
	}
	
	//-------> GPU DET
	if(RR_EXTENDED_TEST){
		RQAdp *output_GPU;
		int output_size = accrqa_LAM_output_size_in_elements(1, 1, 1, 1);
		output_GPU = new RQAdp[output_size];
		int lmin_values = 2;
		int calc_ENTR = 1;
		for(size_t th_idx = 0; th_idx < threshold_list.size(); th_idx++){
			RQAdp threshold_values = threshold_list[th_idx];
			accrqa_DET_GPU(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, ACCRQA_METRIC_MAXIMAL, calc_ENTR, &error);
			GPU_DET_RR_result[th_idx] = output_GPU[4];
		}
		delete[] output_GPU;
	}
	
	int nErrors = 0;
	if(CHECK) {
		nErrors = Compare_data(GPU_RR_result, CPU_RR_result, nThresholds);
		if(nErrors>0 || DEBUG_MODE){
			if(nErrors>0) printf("------> Errors detected:\n");
			for(size_t f=0; f<threshold_list.size(); f++){
				double CPU = CPU_RR_result[f];
				double GPU = GPU_RR_result[f];
				double dif = CPU - GPU;
				printf("f=%zu; CPU:%e; GPU:%e; diff:%e \n", f, CPU, GPU, dif);
			}
		}
		if(RR_EXTENDED_TEST){
			nErrors += Compare_data(GPU_DET_RR_result, CPU_RR_result, nThresholds);
			if(nErrors>0 || DEBUG_MODE){
				if(nErrors>0) printf("------> Errors detected:\n");
				for(size_t f=0; f<threshold_list.size(); f++){
					double CPU = CPU_RR_result[f];
					double GPU = GPU_DET_RR_result[f];
					double dif = CPU - GPU;
					printf("f=%zu; CPU:%e; GPU DET:%e; diff:%e \n", f, CPU, GPU, dif);
				}
			}
			nErrors += Compare_data(GPU_LAM_RR_result, CPU_RR_result, nThresholds);
			if(nErrors>0 || DEBUG_MODE){
				if(nErrors>0) printf("------> Errors detected:\n");
				for(size_t f=0; f<threshold_list.size(); f++){
					double CPU = CPU_RR_result[f];
					double GPU = GPU_LAM_RR_result[f];
					double dif = CPU - GPU;
					printf("f=%zu; CPU:%e; GPU LAM:%e; diff:%e \n", f, CPU, GPU, dif);
				}
			}
		}
	}
	
	delete[] CPU_RR_result;
	delete[] GPU_RR_result;
	delete[] GPU_DET_RR_result;
	delete[] GPU_LAM_RR_result;
	return(nErrors);
}

void unit_test_RR(){
	printf("\n== Recurrent rate unit test ==\n");
	int total_GPU_nErrors = 0, GPU_nErrors = 0;
	
	printf("Recurrent rate with different number of thresholds:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int exp=1; exp<9; exp++){
		int nThresholds = (1<<exp);
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		RQAdp threshold_low = 0.0;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		GPU_nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb);
		total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
		if(GPU_nErrors==0) {
			printf("\033[1;32m.\033[0m");
		}
		else {
			printf("\033[1;31m.\033[0m");
		}
		fflush(stdout);
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	
	printf("\n");
	printf("Recurrent rate with different sizes:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(size_t s = 1014; s < 32768; s = s*2){
		int nThresholds = 10;
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		RQAdp threshold_low = 0;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		GPU_nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb);
		total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
		if(GPU_nErrors==0) {
			printf("\033[1;32m.\033[0m");
		}
		else {
			printf("\033[1;31m.\033[0m");
		}
		fflush(stdout);
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	printf("\n");
	printf("Recurrent rate with different time steps and embeddings:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			int nThresholds = 10;
			size_t size = 10000;
			RQAdp threshold_low = 0.1;
			RQAdp threshold_high = 1.0;
			RQAdp threshold_step = 1.0/((double) nThresholds);
			GPU_nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb);
			total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
			if(GPU_nErrors==0) {
				printf("\033[1;32m.\033[0m");
			}
			else {
				printf("\033[1;31m.\033[0m");
			}
			fflush(stdout);
		}
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<


//---------------- DET and LAM unit test
int test_determinism(long int input_size, RQAdp threshold, int tau, int emb, int lmin){
	std::vector<RQAdp> input_data(input_size, 0);
	Generate_random(&input_data);
	
	// GPU
	RQAdp GPU_DET = 0, GPU_L = 0, GPU_Lmax = 0, GPU_ENTR = 0;
	
	int output_size = accrqa_LAM_output_size_in_elements(1, 1, 1, 1);
	int tau_values = tau;
	int emb_values = emb;
	RQAdp threshold_values = threshold;
	int lmin_values = lmin;
	int calc_ENTR = 1;
	Accrqa_Error error;
	
	if(GPU_UNIT_TEST) {
		RQAdp *output_GPU;
		output_GPU = new RQAdp[output_size];
		accrqa_DET_GPU(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, ACCRQA_METRIC_MAXIMAL, calc_ENTR, &error);
		GPU_DET  = output_GPU[0];
		GPU_L    = output_GPU[1];
		GPU_Lmax = output_GPU[2];
		GPU_ENTR = output_GPU[3];
		delete[] output_GPU;
	}
	
	// CPU
	RQAdp ref_DET = 0, ref_L = 0, ref_Lmax = 0, ref_ENTR = 0;
	{
		RQAdp *output_CPU;
		output_CPU = new RQAdp[output_size];
		accrqa_DET_CPU(output_CPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, ACCRQA_METRIC_MAXIMAL, calc_ENTR, &error);
		ref_DET  = output_CPU[0];
		ref_L    = output_CPU[1];
		ref_Lmax = output_CPU[2];
		ref_ENTR = output_CPU[3];
		delete[] output_CPU;
	}
	
	int nErrors = 0;
	double ferror = 0;
	if(CHECK) {
		if(isnan(GPU_DET) && isnan(ref_DET)) ferror = 0;
		ferror = get_error(ref_DET, GPU_DET);
		if(ferror > max_error) nErrors++;
		if(DEBUG_MODE) printf("DET ref: %e; GPU: %e; diff: %e;\n", ref_DET, GPU_DET, ref_DET - GPU_DET);
		
		if(isnan(GPU_L) && isnan(ref_L)) ferror = 0;
		ferror = get_error(ref_L, GPU_L);
		if(ferror > max_error) nErrors++;
		if(DEBUG_MODE) printf("L ref: %e; GPU: %e; diff: %e;\n", ref_L, GPU_L, ref_L - GPU_L);
		
		if(isnan(GPU_Lmax) && isnan(ref_Lmax)) ferror = 0;
		ferror = get_error(ref_Lmax, GPU_Lmax);
		if(ferror > max_error) nErrors++;
		if(DEBUG_MODE) printf("TTmax ref: %e; GPU: %e; diff: %e;\n", ref_Lmax, GPU_Lmax, ref_Lmax - GPU_Lmax);
		
		if(isnan(GPU_ENTR) && isnan(ref_ENTR)) ferror = 0;
		ferror = get_error(ref_ENTR, GPU_ENTR);
		if(ferror > max_error) nErrors++;
		if(DEBUG_MODE) printf("TTmax ref: %e; GPU: %e; diff: %e;\n", ref_ENTR, GPU_ENTR, ref_ENTR - GPU_ENTR);
	}
	return(nErrors);
}	

void unit_test_DET(){
	printf("\n== Determinism unit test ==\n");
	int total_GPU_nErrors = 0, GPU_nErrors = 0;

	std::vector<RQAdp> threshold_list;
	for(int t = 0; t <= 11; t++){
		threshold_list.push_back((RQAdp)t/10.0);
	}

	printf("Determinism with different number of thresholds:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int t = 0; t<(int)threshold_list.size(); t++){
		RQAdp threshold = threshold_list[t];
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		int lmin = 2;
		if(DEBUG_MODE) printf("Testing with threshold=%f;\n", threshold);
		GPU_nErrors = test_determinism(size, threshold, tau, emb, lmin);
		total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
		if(GPU_nErrors==0) {
			printf("\033[1;32m.\033[0m");
		}
		else {
			printf("\033[1;31m.\033[0m");
		}
		fflush(stdout);
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	printf("Determinism with different input sizes:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(size_t s = 1014; s < 32768; s = s*2){
		for(int t = 0; t<(int)threshold_list.size(); t++){
			RQAdp threshold = threshold_list[t];
			int tau = 1;
			int emb = 1;
			int lmin = 2;
			if(DEBUG_MODE) printf("Testing with size=%zu;\n", s);
			GPU_nErrors = test_determinism(s, threshold, tau, emb, lmin);
			total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
			if(GPU_nErrors==0) {
				printf("\033[1;32m.\033[0m");
			}
			else {
				printf("\033[1;31m.\033[0m");
			}
			fflush(stdout);
		}
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	

	total_GPU_nErrors = 0; GPU_nErrors = 0;
	printf("Determinism with different time steps and embeddings:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			for(int t = 0; t<(int)threshold_list.size(); t++){
				RQAdp threshold = threshold_list[t];
				size_t size = 10000;
				if(DEBUG_MODE) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d\n", size, threshold, tau, emb);
				int lmin = 2;
				GPU_nErrors = test_determinism(size, threshold, tau, emb, lmin);
				total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
				if(GPU_nErrors==0) {
					printf("\033[1;32m.\033[0m");
				}
				else {
					printf("\033[1;31m.\033[0m");
				}
				fflush(stdout);
			}
		}
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<


//---------------- DET and LAM unit test
int test_laminarity(long int input_size, RQAdp threshold, int tau, int emb, int vmin){
	std::vector<RQAdp> input_data(input_size, 0);
	Generate_random(&input_data);
	
	int output_size = accrqa_LAM_output_size_in_elements(1, 1, 1, 1);
	int tau_values = tau;
	int emb_values = emb;
	RQAdp threshold_values = threshold;
	int vmin_values = vmin;
	int calc_ENTR = 0;
	Accrqa_Error error;
	
	RQAdp GPU_LAM, GPU_TT, GPU_TTmax;
	if(GPU_UNIT_TEST) {
		RQAdp *output_GPU;
		output_GPU = new RQAdp[output_size];
		accrqa_LAM_GPU(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, ACCRQA_METRIC_MAXIMAL, calc_ENTR, &error);
		GPU_LAM   = output_GPU[0];
		GPU_TT    = output_GPU[1];
		GPU_TTmax = output_GPU[2];
		delete[] output_GPU;
	}
	
	// Reference
	RQAdp ref_LAM, ref_TT, ref_TTmax;
	{
		RQAdp *output_CPU;
		output_CPU = new RQAdp[output_size];
		accrqa_LAM_CPU(output_CPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, ACCRQA_METRIC_MAXIMAL, calc_ENTR, &error);
		ref_LAM   = output_CPU[0];
		ref_TT    = output_CPU[1];
		ref_TTmax = output_CPU[2];
		delete[] output_CPU;
	}
	
	//Potentially add histogram tests as well
	int nErrors = 0;
	double ferror = 0;
	if(CHECK) {
		if(isnan(GPU_LAM) && isnan(ref_LAM)) ferror = 0;
		else ferror = get_error(ref_LAM, GPU_LAM);
		if(ferror > max_error) nErrors++;
		if(DEBUG_MODE) printf("LAM ref: %e; GPU: %e; diff: %e;\n", ref_LAM, GPU_LAM, ref_LAM - GPU_LAM);
		
		if(isnan(GPU_TT) && isnan(ref_TT)) ferror = 0;
		else ferror = get_error(ref_TT, GPU_TT);
		if(ferror > max_error) nErrors++;
		if(DEBUG_MODE) printf("TT ref: %e; GPU: %e; diff: %e;\n", ref_TT, GPU_TT, ref_TT - GPU_TT);
		
		if(isnan(GPU_TTmax) && isnan(ref_TTmax)) ferror = 0;
		else ferror = get_error(ref_TTmax, GPU_TTmax);
		if(ferror > max_error) nErrors++;
		if(DEBUG_MODE) printf("TTmax ref: %e; GPU: %e; diff: %e;\n", ref_TTmax, GPU_TTmax, ref_TTmax - GPU_TTmax);
	}
	return(nErrors);
}

void unit_test_LAM(void){
	printf("\n== Laminarity unit test ==\n");
	int total_GPU_nErrors = 0, GPU_nErrors = 0;
	
	std::vector<RQAdp> threshold_list;
	for(int t = 0; t <= 11; t++){
		threshold_list.push_back((RQAdp)t/10.0);
	}
	
	GPU_nErrors = 0; total_GPU_nErrors = 0;
	printf("Laminarity with different number of thresholds:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int t = 0; t<(int)threshold_list.size(); t++){
		RQAdp threshold = threshold_list[t];
		size_t size = 10000;
		int tau = 1;
		int emb = 1;
		int vmin = 2;
		if(DEBUG_MODE) printf("Testing with threshold=%f;\n", threshold);
		GPU_nErrors = test_laminarity(size, threshold, tau, emb, vmin);
		total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
		if(GPU_nErrors==0) {
			printf("\033[1;32m.\033[0m");
		}
		else {
			printf("\033[1;31m.\033[0m");
		}
		fflush(stdout);
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	
	
	GPU_nErrors = 0; total_GPU_nErrors = 0;
	printf("Laminarity with different input sizes:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(size_t s = 1014; s < 32768; s = s*2){
		for(int t = 0; t<(int)threshold_list.size(); t++){
			RQAdp threshold = threshold_list[t];
			int tau = 1;
			int emb = 1;
			int vmin = 2;
			if(DEBUG_MODE) printf("Testing with size=%zu and threshold=%f\n", s, threshold);
			GPU_nErrors = test_laminarity(s, threshold, tau, emb, vmin);
			total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
			if(GPU_nErrors==0) {
				printf("\033[1;32m.\033[0m");
			}
			else {
				printf("\033[1;31m.\033[0m");
			}
			fflush(stdout);
		}
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");


	GPU_nErrors = 0; total_GPU_nErrors = 0;
	printf("Laminarity with different time step and embedding:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			for(int t = 0; t<(int)threshold_list.size(); t++){
				size_t size = 10000;
				int vmin = 2;
				RQAdp threshold = threshold_list[t];
				if(DEBUG_MODE) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d \n", size, threshold, tau, emb);
				GPU_nErrors = test_laminarity(size, threshold, tau, emb, vmin);
				total_GPU_nErrors = total_GPU_nErrors + GPU_nErrors;
				if(GPU_nErrors==0) {
					printf("\033[1;32m.\033[0m");
				}
				else {
					printf("\033[1;31m.\033[0m");
				}
				fflush(stdout);
			}
		}
	}
	printf("\n");
	if(total_GPU_nErrors==0) printf("     Test:\033[1;32mPASSED\033[0m\n");
	else printf("     Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<


int main(void) {
	
	unit_test_RR();
	//unit_test_DET();
	//unit_test_LAM();
	
	return (0);
}

