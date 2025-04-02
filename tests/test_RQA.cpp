#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <limits>

using namespace std;

#include <dirent.h>

#include "../include/AccRQA_library.hpp"

typedef float RQAdp;

bool DEBUG_MODE = false;
bool CHECK = true;
bool GPU_UNIT_TEST = true;
bool CPU_UNIT_TEST = true;

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
		if(error > max_error) nErrors++;
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
int test_recurrent_rate(size_t input_size, RQAdp threshold_low, RQAdp threshold_high, RQAdp threshold_step, int tau, int emb, Accrqa_Distance distance_type){
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
	
	if(DEBUG_MODE) printf("--> RR unit test: data size = %zu; Thresholds=%d; emb=%d; tau=%d;\n", input_size, nThresholds, emb, tau);
	
	//-------> GPU
	if(GPU_UNIT_TEST) {
		accrqa_RR(GPU_RR_result, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, distance_type, PLT_NV_GPU, &error);
		if(DEBUG_MODE) {
			printf("---->ACCRQA Error (accrqa_RR_GPU):");
			accrqa_print_error(&error);
			printf("\n");
		}
	}
	
	//-------> CPU
	accrqa_RR(CPU_RR_result, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, distance_type, PLT_CPU, &error);
	
	int nErrors = 0;
	if(CHECK) {
		nErrors = Compare_data(GPU_RR_result, CPU_RR_result, nThresholds);
		if(nErrors>0 || DEBUG_MODE){
			if(nErrors>0) printf("------> Errors detected:\n");
			for(size_t f=0; f<threshold_list.size(); f++){
				double CPU = CPU_RR_result[f];
				double GPU = GPU_RR_result[f];
				double dif = CPU - GPU;
				printf("RR: f=%zu; CPU:%e; GPU:%e; diff:%e \n", f, CPU, GPU, dif);
			}
		}
	}
	
	delete[] CPU_RR_result;
	delete[] GPU_RR_result;
	delete[] GPU_DET_RR_result;
	delete[] GPU_LAM_RR_result;
	return(nErrors);
}

int test_recurrent_rate_extended(size_t input_size, RQAdp threshold_low, RQAdp threshold_high, RQAdp threshold_step, int tau, int emb){
	std::vector<RQAdp> input_data(input_size, 0);
	Generate_random(&input_data);
	
	std::vector<RQAdp> threshold_list;
	for (RQAdp threshold = threshold_low; threshold < threshold_high; threshold = threshold + threshold_step) threshold_list.push_back(threshold);
	int nThresholds = (int) threshold_list.size(); 
	
	RQAdp *CPU_RR_result, *GPU_DET_RR_result, *GPU_LAM_RR_result;
	int RR_output_size = accrqa_RR_output_size_in_elements(1, 1, threshold_list.size());
	CPU_RR_result = new RQAdp[nThresholds];
	GPU_DET_RR_result = new RQAdp[nThresholds];
	GPU_LAM_RR_result = new RQAdp[nThresholds];

	int tau_values = tau;
	int emb_values = emb;
	Accrqa_Error error;
	
	if(DEBUG_MODE) printf("--> RR unit test: data size = %zu; Thresholds=%d; emb=%d; tau=%d;\n", input_size, nThresholds, emb, tau);
	
	//-------> CPU
	accrqa_RR(CPU_RR_result, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, DST_MAXIMAL, PLT_CPU, &error);
	
	//-------> GPU LAM
	{
		RQAdp *output_GPU;
		int output_size = accrqa_LAM_output_size_in_elements(1, 1, 1, 1);
		output_GPU = new RQAdp[output_size];
		int vmin_values = 2;
		int calc_ENTR = 1;
		for(size_t th_idx = 0; th_idx < threshold_list.size(); th_idx++){
			RQAdp threshold_values = threshold_list[th_idx];
			accrqa_LAM(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_NV_GPU, &error);
			GPU_LAM_RR_result[th_idx] = output_GPU[4];
			if(DEBUG_MODE) {
				printf("---->ACCRQA Error (accrqa_LAM_GPU):");
				accrqa_print_error(&error);
				printf("\n");
			}
		}
		delete[] output_GPU;
	}
	
	//-------> GPU DET
	{
		RQAdp *output_GPU;
		int output_size = accrqa_LAM_output_size_in_elements(1, 1, 1, 1);
		output_GPU = new RQAdp[output_size];
		int lmin_values = 2;
		int calc_ENTR = 1;
		for(size_t th_idx = 0; th_idx < threshold_list.size(); th_idx++){
			RQAdp threshold_values = threshold_list[th_idx];
			accrqa_DET(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_NV_GPU, &error);
			GPU_DET_RR_result[th_idx] = output_GPU[4];
			if(DEBUG_MODE) {
				printf("---->ACCRQA Error (accrqa_DET_GPU):");
				accrqa_print_error(&error);
				printf("\n");
			}
		}
		delete[] output_GPU;
	}
	
	int nErrors = 0;
	if(CHECK) {
		nErrors += Compare_data(GPU_DET_RR_result, CPU_RR_result, nThresholds);
		if(nErrors>0 || DEBUG_MODE){
			if(nErrors>0) printf("------> Errors detected:\n");
			for(size_t f=0; f<threshold_list.size(); f++){
				double CPU = CPU_RR_result[f];
				double GPU = GPU_DET_RR_result[f];
				double dif = CPU - GPU;
				printf("DET: f=%zu; CPU:%e; GPU DET:%e; diff:%e \n", f, CPU, GPU, dif);
			}
		}
		nErrors += Compare_data(GPU_LAM_RR_result, CPU_RR_result, nThresholds);
		if(nErrors>0 || DEBUG_MODE){
			if(nErrors>0) printf("------> Errors detected:\n");
			for(size_t f=0; f<threshold_list.size(); f++){
				double CPU = CPU_RR_result[f];
				double GPU = GPU_LAM_RR_result[f];
				double dif = CPU - GPU;
				printf("LAM: f=%zu; CPU:%e; GPU LAM:%e; diff:%e \n", f, CPU, GPU, dif);
			}
		}
	}
	
	delete[] CPU_RR_result;
	delete[] GPU_DET_RR_result;
	delete[] GPU_LAM_RR_result;
	return(nErrors);
}


void unit_test_RR(){
	printf("\n== Recurrent rate unit test ==\n");
	int total_GPU_nErrors = 0, GPU_nErrors = 0;
	
	printf("  Recurrent rate with different number of thresholds:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int exp=1; exp<9; exp++){
		int nThresholds = (1<<exp);
		size_t size = 1000;
		int tau = 1;
		int emb = 1;
		Accrqa_Distance distance_type = DST_MAXIMAL;
		RQAdp threshold_low = 0.0;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		GPU_nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	
	printf("  Recurrent rate with different sizes:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(size_t s = 1014; s < 32768; s = s*2){
		int nThresholds = 10;
		size_t size = s;
		int tau = 1;
		int emb = 1;
		Accrqa_Distance distance_type = DST_MAXIMAL;
		RQAdp threshold_low = 0;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		GPU_nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	printf("  Recurrent rate with different time steps and embeddings:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			int nThresholds = 10;
			size_t size = 1000;
			Accrqa_Distance distance_type = DST_MAXIMAL;
			RQAdp threshold_low = 0.1;
			RQAdp threshold_high = 1.0;
			RQAdp threshold_step = 1.0/((double) nThresholds);
			GPU_nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	printf("  Recurrent rate with different distance types:\n"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int distance_type=1; distance_type<=2; distance_type++){
		if(distance_type==DST_EUCLIDEAN) printf("    DST_EUCLIDEAN: ");
		else if(distance_type==DST_MAXIMAL) printf("    DST_MAXIMAL: ");
		else printf("Unknown metric!\n");
		for(int tau = 1; tau < 6; tau++){
			for(int emb = 1; emb < 12; emb++){
				int nThresholds = 10;
				size_t size = 1000;
				RQAdp threshold_low = 0.1;
				RQAdp threshold_high = 1.0;
				RQAdp threshold_step = 1.0/((double) nThresholds);
				GPU_nErrors = test_recurrent_rate(size, threshold_low, threshold_high, threshold_step, tau, emb, (Accrqa_Distance) distance_type);
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
	}
	if(total_GPU_nErrors==0) printf("      Test:\033[1;32mPASSED\033[0m\n");
	else printf("      Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}

void unit_test_RR_extended(){
	printf("\n== Recurrent rate through DET and LAM unit test ==\n");
	int total_GPU_nErrors = 0, GPU_nErrors = 0;
	
	printf("  Recurrent rate DET and LAM with different number of thresholds:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int exp=1; exp<9; exp++){
		int nThresholds = (1<<exp);
		size_t size = 1000;
		int tau = 1;
		int emb = 1;
		RQAdp threshold_low = 0.0;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		GPU_nErrors = test_recurrent_rate_extended(size, threshold_low, threshold_high, threshold_step, tau, emb);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	
	printf("  Recurrent rate DET and LAM with different sizes:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(size_t s = 1014; s < 32768; s = s*2){
		int nThresholds = 10;
		size_t size = s;
		int tau = 1;
		int emb = 1;
		RQAdp threshold_low = 0;
		RQAdp threshold_high = 1.0;
		RQAdp threshold_step = 1.0/((double) nThresholds);
		GPU_nErrors = test_recurrent_rate_extended(size, threshold_low, threshold_high, threshold_step, tau, emb);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	printf("  Recurrent rate using DET and LAM with different time steps and embeddings:"); fflush(stdout);
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			int nThresholds = 10;
			size_t size = 1000;
			RQAdp threshold_low = 0.1;
			RQAdp threshold_high = 1.0;
			RQAdp threshold_step = 1.0/((double) nThresholds);
			GPU_nErrors = test_recurrent_rate_extended(size, threshold_low, threshold_high, threshold_step, tau, emb);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<


//---------------- DET and LAM unit test
int test_determinism(long int input_size, RQAdp threshold, int tau, int emb, int lmin, Accrqa_Distance distance_type){
	std::vector<RQAdp> input_data(input_size, 0);
	Generate_random(&input_data);
	
	// GPU
	RQAdp GPU_DET = 0, GPU_L = 0, GPU_Lmax = 0, GPU_ENTR = 0;
	
	int output_size = accrqa_LAM_output_size_in_elements(1, 1, 1, 1);
	int tau_values = tau;
	int emb_values = emb;
	RQAdp threshold_values = threshold;
	int lmin_values = lmin;
	int calc_ENTR = 0;
	Accrqa_Error error;
	
	if(GPU_UNIT_TEST) {
		RQAdp *output_GPU;
		output_GPU = new RQAdp[output_size];
		accrqa_DET(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, distance_type, calc_ENTR, PLT_NV_GPU, &error);
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
		accrqa_DET(output_CPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, distance_type, calc_ENTR, PLT_CPU, &error);
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
		
		if( calc_ENTR == 1) {
			if(isnan(GPU_Lmax) && isnan(ref_Lmax)) ferror = 0;
			ferror = get_error(ref_Lmax, GPU_Lmax);
			if(ferror > max_error) nErrors++;
			if(DEBUG_MODE) printf("TTmax ref: %e; GPU: %e; diff: %e;\n", ref_Lmax, GPU_Lmax, ref_Lmax - GPU_Lmax);
			
			if(isnan(GPU_ENTR) && isnan(ref_ENTR)) ferror = 0;
			ferror = get_error(ref_ENTR, GPU_ENTR);
			if(ferror > max_error) nErrors++;
			if(DEBUG_MODE) printf("ENTR ref: %e; GPU: %e; diff: %e;\n", ref_ENTR, GPU_ENTR, ref_ENTR - GPU_ENTR);
		}
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
	
	printf("  Determinism with different number of thresholds:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	for(int t = 0; t<(int)threshold_list.size(); t++){
		RQAdp threshold = threshold_list[t];
		size_t size = 500;
		int tau = 1;
		int emb = 1;
		int lmin = 2;
		Accrqa_Distance distance_type = DST_MAXIMAL;
		if(DEBUG_MODE) printf("Testing with threshold=%f;\n", threshold);
		GPU_nErrors = test_determinism(size, threshold, tau, emb, lmin, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	printf("  Determinism with different input sizes:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(size_t s = 1014; s < 65538; s = s*2){
		for(int t = 0; t<(int)threshold_list.size(); t++){
			RQAdp threshold = threshold_list[t];
			int tau = 1;
			int emb = 1;
			int lmin = 2;
			Accrqa_Distance distance_type = DST_MAXIMAL;
			if(DEBUG_MODE) printf("Testing with size=%zu;\n", s);
			GPU_nErrors = test_determinism(s, threshold, tau, emb, lmin, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	total_GPU_nErrors = 0; GPU_nErrors = 0;
	printf("  Determinism with different time steps and embeddings:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			for(int t = 0; t<(int)threshold_list.size(); t++){
				RQAdp threshold = threshold_list[t];
				size_t size = 1000;
				Accrqa_Distance distance_type = DST_MAXIMAL;
				if(DEBUG_MODE) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d\n", size, threshold, tau, emb);
				int lmin = 2;
				GPU_nErrors = test_determinism(size, threshold, tau, emb, lmin, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	GPU_nErrors = 0; total_GPU_nErrors = 0;
	printf("  Determinism with different distance types:\n"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int distance_type=1; distance_type<=2; distance_type++){
		if(distance_type==DST_EUCLIDEAN) printf("    DST_EUCLIDEAN: ");
		else if(distance_type==DST_MAXIMAL) printf("    DST_MAXIMAL: ");
		else printf("Unknown metric!\n");
		for(int tau = 1; tau < 6; tau++){
			for(int emb = 1; emb < 12; emb++){
				RQAdp threshold = threshold_list[5];
				size_t size = 1000;
				if(DEBUG_MODE) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d\n", size, threshold, tau, emb);
				int lmin = 2;
				GPU_nErrors = test_determinism(size, threshold, tau, emb, lmin, (Accrqa_Distance) distance_type);
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
	}
	if(total_GPU_nErrors==0) printf("      Test:\033[1;32mPASSED\033[0m\n");
	else printf("      Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<


//---------------- DET and LAM unit test
int test_laminarity(long int input_size, RQAdp threshold, int tau, int emb, int vmin, Accrqa_Distance distance_type){
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
		accrqa_LAM(output_GPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, distance_type, calc_ENTR, PLT_NV_GPU, &error);
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
		accrqa_LAM(output_CPU, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, distance_type, calc_ENTR, PLT_CPU, &error);
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
		
		if(calc_ENTR==1){
			if(isnan(GPU_TTmax) && isnan(ref_TTmax)) ferror = 0;
			else ferror = get_error(ref_TTmax, GPU_TTmax);
			if(ferror > max_error) nErrors++;
			if(DEBUG_MODE) printf("TTmax ref: %e; GPU: %e; diff: %e;\n", ref_TTmax, GPU_TTmax, ref_TTmax - GPU_TTmax);
		}
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
	printf("  Laminarity with different number of thresholds:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int t = 0; t<(int)threshold_list.size(); t++){
		RQAdp threshold = threshold_list[t];
		size_t size = 1000;
		int tau = 1;
		int emb = 1;
		int vmin = 2;
		Accrqa_Distance distance_type = DST_MAXIMAL;
		if(DEBUG_MODE) printf("Testing with threshold=%f;\n", threshold);
		GPU_nErrors = test_laminarity(size, threshold, tau, emb, vmin, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	
	GPU_nErrors = 0; total_GPU_nErrors = 0;
	printf("  Laminarity with different input sizes:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(size_t s = 1014; s < 32768; s = s*2){
		for(int t = 0; t<(int)threshold_list.size(); t++){
			RQAdp threshold = threshold_list[t];
			int tau = 1;
			int emb = 1;
			int vmin = 2;
			Accrqa_Distance distance_type = DST_MAXIMAL;
			if(DEBUG_MODE) printf("Testing with size=%zu and threshold=%f\n", s, threshold);
			GPU_nErrors = test_laminarity(s, threshold, tau, emb, vmin, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");


	GPU_nErrors = 0; total_GPU_nErrors = 0;
	printf("  Laminarity with different time step and embedding:"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int tau = 1; tau < 6; tau++){
		for(int emb = 1; emb < 12; emb++){
			for(int t = 0; t<(int)threshold_list.size(); t++){
				size_t size = 1000;
				int vmin = 2;
				Accrqa_Distance distance_type = DST_MAXIMAL;
				RQAdp threshold = threshold_list[t];
				if(DEBUG_MODE) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d \n", size, threshold, tau, emb);
				GPU_nErrors = test_laminarity(size, threshold, tau, emb, vmin, distance_type);
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
	if(total_GPU_nErrors==0) printf("    Test:\033[1;32mPASSED\033[0m\n");
	else printf("    Test:\033[1;31mFAILED\033[0m\n");
	
	GPU_nErrors = 0; total_GPU_nErrors = 0;
	printf("  Laminarity with different distance types:\n"); fflush(stdout);
	if(DEBUG_MODE) printf("\n");
	for(int distance_type=1; distance_type<=2; distance_type++){
		if(distance_type==DST_EUCLIDEAN) printf("    DST_EUCLIDEAN: ");
		else if(distance_type==DST_MAXIMAL) printf("    DST_MAXIMAL: ");
		else printf("Unknown metric!\n");
		for(int tau = 1; tau < 6; tau++){
			for(int emb = 1; emb < 12; emb++){
				size_t size = 1000;
				int vmin = 2;
				RQAdp threshold = threshold_list[5];
				if(DEBUG_MODE) printf("Testing with size=%zu, threshold=%f, tau=%d and emb=%d \n", size, threshold, tau, emb);
				GPU_nErrors = test_laminarity(size, threshold, tau, emb, vmin, (Accrqa_Distance) distance_type);
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
	}
	if(total_GPU_nErrors==0) printf("      Test:\033[1;32mPASSED\033[0m\n");
	else printf("      Test:\033[1;31mFAILED\033[0m\n");
	printf("----------------------------------<\n");
}
// ---------------------------------------------<




template <typename input_type>
inline double distance_euclidean(input_type *input, size_t i, size_t j, int tau, int emb){
	input_type sum = 0;
	for(int m = 0; m < emb; m++){
		input_type A = input[i + m*tau];
		input_type B = input[j + m*tau];
		sum += (A - B)*(A - B);
	}
	sum = sqrt(sum);
	return(sum);
}


template <typename input_type>
inline double distance_maximum(input_type *input, size_t i, size_t j, int tau, int emb){
	input_type max = 0;
	for(int m = 0; m < emb; m++){
		input_type A = input[i + (size_t) (m*tau)];
		input_type B = input[j + (size_t) (m*tau)];
		input_type dist = abs(A - B);
		if(dist > max) max = dist;
	}
	return(max);
}

template <typename input_type>
int R_matrix_element(input_type *input, size_t i, size_t j, input_type threshold, int tau, int emb, int distance_type){
	input_type distance = 0;
	if(distance_type == 1) distance = distance_euclidean(input, i, j, tau, emb);
	if(distance_type == 2) distance = distance_maximum(input, i, j, tau, emb);
	int R_element = ( (threshold - distance)>=0 ? 1 : 0 );
	//printf("[%d;%d]=%d; ", (int) i, (int) j, R_element);
	return ( R_element );
}



size_t calculate_k(size_t last_block_end){
	if(last_block_end == 0) return (0);
	double a = 1.0;
	double b = 1.0; // always positive
	double c = -2.0*last_block_end; // always negative
	double D = 1.0 - 4.0*a*c;
	double n1 = (sqrt(D)-1.0)/2.0;
	return( (size_t) n1);
}

void calculate_coordinates(size_t *i, size_t *j, size_t pos, size_t corrected_size){
	size_t pk = calculate_k(pos);
	*i = pos - (pk*(pk-1))/2 - pk;
	*j = corrected_size - pk - 1 + *i;
}


size_t test_coordinates(size_t corrected_size){
	size_t nErrors = 0;
	std::vector<size_t> msizes;
	for(size_t f=1; f<corrected_size; f++) msizes.push_back(f);
	std::vector<size_t> mscan;
	mscan.push_back(0);
	for(size_t f=0; f<=corrected_size; f++) mscan.push_back(mscan[f] + msizes[f]);
	for(size_t f=0; f<corrected_size-1; f++) {
		size_t i, j;
		calculate_coordinates(&i, &j, mscan[f], corrected_size);
		if(i != 0 || j != corrected_size-1-f) nErrors++;
	}
	return(nErrors);
}


template <typename input_type>
size_t test_indexing(input_type *time_series, size_t data_size, input_type threshold, int tau, int emb, int distance_type){
	size_t corrected_size = data_size - (emb - 1)*tau;
	size_t nErrors = 0;
	std::vector<int> linearised_R_matrix;
	for (size_t r = corrected_size-1; r>0; r--) {
		size_t distance_from_diagonal = r;
		size_t diagonal_length = corrected_size-distance_from_diagonal;
		for(size_t s=0; s<diagonal_length; s++) {
			long int row = s;
			long int column = s + distance_from_diagonal;
			int R_matrix_value = R_matrix_element(time_series, row, column, threshold, tau, emb, distance_type);
			linearised_R_matrix.push_back(R_matrix_value);
		}
	}
	
	size_t block_size = 1024;
	size_t total_elements = ((corrected_size-1)*corrected_size)/2;
	size_t nBlocks = (total_elements + block_size - 1)/block_size;
	for(size_t b=0; b<nBlocks; b++){
		for(size_t th_id = 0; th_id < block_size; th_id++){
			size_t pos = b*block_size + th_id;
			if(pos < total_elements){
				size_t i, j;
				calculate_coordinates(&i, &j, pos, corrected_size);
				int R_matrix_value = R_matrix_element(time_series, i, j, threshold, tau, emb, distance_type);
				if(R_matrix_value != linearised_R_matrix[pos]) {
					if(nErrors < 20 ){
						printf("th_id = %zu; b=%zu; pos=%zu; i = %zu; j = %zu;\n", th_id, b, pos, i, j);
						printf("%d != %d;\n", R_matrix_value, linearised_R_matrix[pos]);
					}
					nErrors++;
				}
			}
		}
	}
	
	return(nErrors);
}


void init_index_testing(){
	int tau = 1;
	int emb = 1;
	int lmin = 2;
	size_t data_size = 10;
	
	{
		size_t nErrors = 0;
		printf("Test of coordinate calculation: ");
		for(size_t s = 10; s <= 100; s=s*10){
			nErrors += test_coordinates(s);
		}
		if(nErrors>0) {
			printf("\033[1;31mFAILED\033[0m with number of errors = %zu;\n", nErrors);
		}
		else printf("\033[1;32mPASSED\033[0m\n");
	}
	
	std::vector<RQAdp> threshold_list;
	for(int t = 0; t <= 11; t++){
		threshold_list.push_back((RQAdp)t/10.0);
	}
	
	for(int s = 10; s <= 10000; s=s*10){
		size_t nErrors = 0;
		std::vector<RQAdp> input_data(s, 0);
		Generate_random(&input_data);
		printf("Test of regularized indexing at size %d: ", s); fflush(stdout);
		for(size_t t = 0; t<threshold_list.size(); t++){
			nErrors += test_indexing(input_data.data(), input_data.size(), threshold_list[t], tau, emb, DST_MAXIMAL);
		}
		input_data.clear();
		if(nErrors>0) {
			printf("\033[1;31mFAILED\033[0m with number of errors = %zu;\n", nErrors);
		}
		else printf("\033[1;32mPASSED\033[0m\n");
	}
}

void add_line(std::vector<int> *matrix, int lenght){
	matrix->push_back(0);
	for(int f=0; f<lenght;f++){
		matrix->push_back(1);
	}
}

void delete_lines(std::vector<int> *matrix, int lenght){
	size_t size = matrix->size();
	for(size_t f=0;f<size;f++){
		int sum = 0;
		for(size_t l=0; l<lenght && (f+l)<size;l++){
			sum += matrix->operator[](f+l);
		}
		matrix->operator[](f) = (int) (sum/(lenght));
	}
}

void print_line(std::vector<int> matrix){
	size_t size = matrix.size();
	for(size_t f=0;f<size;f++){
		printf("%d ", matrix[f]);
	}
	printf("\n");
}

int sum_line(std::vector<int> matrix){
	size_t size = matrix.size();
	int sum = 0;
	for(size_t f=0;f<size;f++){
		sum += matrix[f];
	}
	return(sum);
}

void apply_filter(std::vector<int> *matrix, int length){
	size_t size = matrix->size();
	for(size_t f=0;f<size-1;f++){
		if(f==0 && matrix->operator[](f)==1){
			matrix->operator[](f) = length;
		}
		else {
			if(matrix->operator[](f) == 0 && matrix->operator[](f+1) == 1){
				matrix->operator[](f+1) = length;
			}
		}
	}
}

void test_delete_lines(){
	printf("Testing line deletion and sum correction: ");
	fflush(stdout);
	std::vector<int> matrix;
	std::vector<int> lengths={1,2,1,2,3,4,2,1,4,4,5,7,9};
	for(size_t f=0; f<lengths.size(); f++){
		add_line(&matrix, lengths[f]);
	}
	int sum_before = sum_line(matrix);
	delete_lines(&matrix, 2);
	int sum_deleted = sum_line(matrix);
	apply_filter(&matrix, 2);
	int sum_corrected = sum_line(matrix);
	if(sum_before != 45 || sum_deleted != 32 || sum_corrected != 42) {
		printf("\033[1;31mFAILED\033[0m\n");
	}
	else {
		printf("\033[1;32mPASSED\033[0m\n");
	}
}

void test_starting_point(){
	int block_size = 1024;
	int lmin = 2;
	int start_overlap = 1;
	int nElements = block_size - lmin + 1 - 1;
	
	for(int f=0; f<10; f++){
		int global_pos = f*nElements;
		int start = global_pos;
		int end = global_pos + block_size - lmin;
	}
}






int main(void) {
	
	init_index_testing();
	test_delete_lines();
	test_starting_point();
	
	unit_test_RR();
	unit_test_RR_extended();
	unit_test_DET();
	unit_test_LAM();
	
	return (0);
}

