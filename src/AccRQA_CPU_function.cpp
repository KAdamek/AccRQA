//#include "AccRQA_CPU_function.h"

#include "debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <math.h>

//---------------------- Utilities ------------------------->
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
	return ( ( (threshold - distance)>=0 ? 1 : 0 ) );
}

template <typename input_type>
void get_length_histogram(unsigned long long int *LAM_histogram, input_type *time_series, size_t size) {
	for (size_t f = 0; f<size; f++) {
		if (time_series[f] == 1) {
			// Start counting line length
			size_t line_length = 1;
			while (time_series[f + line_length]==1 && (f + line_length) < size) {
				line_length++;
			}
			//printf("Adding line_length: %d;\n", line_length);
			LAM_histogram[line_length]++;
			f = f + line_length;
		}
	}
}


template<typename input_type>
void serialScanInclusive(input_type *result, input_type *h_input, size_t data_size){
	int running_sum = 0;
	
	for(size_t f=0; f<data_size; f++){
		running_sum = running_sum + h_input[f];
		result[f] = running_sum;
	}
}

/*
unsigned long long int reduce(unsigned int *histogram, size_t histogram_size){
	unsigned long long int sum = 0;
	for(size_t f=1; f<histogram_size; f++){
		sum = sum + histogram[f];
	}
	return(sum);
}
*/

template<typename input_type>
void reverseArrayAndMultiply(input_type *destination, input_type *source, unsigned int input_size){
	for(unsigned int f=0; f<input_size; f++){
		unsigned long long int pos = input_size - 1 - f;
		destination[f] = pos*source[pos];
	}
}


template<typename input_type>
void reverseArray(input_type *destination, input_type *source, unsigned int input_size){
	for(unsigned int f=0; f<input_size; f++){
		destination[f] = source[input_size - 1 - f];
	}
}


//---------------------- Recurrent matrix -------------------->
template <typename input_type>
int rqa_CPU_R_matrix_ref(
	int *R_matrix, 
	input_type *time_series, 
	long int corrected_size, 
	input_type threshold, 
	int tau, 
	int emb, 
	int distance_type
){
	for(long int i = 0; i < corrected_size; i++){
		for(long int j = 0; j < corrected_size; j++){
			size_t r_pos = i*corrected_size + j;
			R_matrix[r_pos] = R_matrix_element(time_series, i, j, threshold, tau, emb, distance_type);
		}
	}
	return(0);
}


template <typename input_type>
int rqa_CPU_R_matrix(
	int *R_matrix, 
	input_type *time_series, 
	long int corrected_size, 
	input_type threshold, 
	int tau, 
	int emb, 
	int distance_type
){
	return(0);
}


template <typename input_type>
int rqa_CPU_R_matrix_diagonal_ref(
	int *R_matrix_diagonal, 
	input_type *time_series, 
	long int corrected_size, 
	input_type threshold, 
	int tau, 
	int emb, 
	int distance_type
){
	long int line_count = 0;
	memset(R_matrix_diagonal, 0, (2*corrected_size - 1)*corrected_size*sizeof(int));
	// upper triangle
	for (long int r = corrected_size-1; r>0; r--) {
		for(long int s=0; s<corrected_size; s++) {
			long int row = s;
			long int column = s + r;
			if(s<corrected_size) {
				size_t r_pos = line_count*corrected_size + s;
				R_matrix_diagonal[r_pos] = R_matrix_element(time_series, row, column, threshold, tau, emb, distance_type);
			}
			else break;
		}
		line_count++;
	}
	
	//diagonal:
	for(long int s=0; s<corrected_size; s++) {
		R_matrix_diagonal[line_count*corrected_size + s] = R_matrix_element(time_series, s, s, threshold, tau, emb, distance_type);
	}
	line_count++;
	
	// lower triangle
	for (long int r = 1; r<corrected_size; r++) {
		for(long int s=0; s<corrected_size; s++) {
			long int row = s + r;
			long int column = s;
			if(row < corrected_size){
				size_t r_pos = line_count*corrected_size + s;
				R_matrix_diagonal[r_pos] = R_matrix_element(time_series, row, column, threshold, tau, emb, distance_type);
			}
			else break;
		}
		line_count++;
	}
	return(0);
}


template <typename input_type>
int rqa_CPU_R_matrix_diagonal(
	int *R_matrix_diagonal, 
	input_type *time_series, 
	long int corrected_size, 
	input_type threshold, 
	int tau, 
	int emb, 
	int distance_type
){
	return(0);
}
//----------------------------------------------------------<


//---------------------- Recurrent rate -------------------->
template <typename input_type>
void rqa_CPU_RR_metric_ref(
	input_type *output_RR, 
	input_type *time_series, 
	unsigned long long int input_size, 
	input_type threshold, 
	int tau, 
	int emb, 
	int distance_type
) {
	long int corrected_size = input_size - (emb - 1)*tau;
	
	unsigned long long int sum = 0;
	for(long int i=0; i<corrected_size; i++) {
		for(long int j=0; j<corrected_size; j++) {
			sum = sum + R_matrix_element(time_series, i, j, threshold, tau, emb, distance_type);
		}
	}
	*output_RR = ((input_type) sum)/((input_type) (corrected_size*corrected_size));
	
	if(DEBUG) {printf("|"); fflush( stdout );}
}

template <typename input_type>
void rqa_CPU_RR_metric(
	unsigned long long int *recurrent_rate_integers, 
	std::vector<input_type> threshold_list, 
	int tau, 
	int emb, 
	input_type *time_series, 
	unsigned long long int input_size, 
	int distance_type
) {
	
}
//----------------------------------------------------------<


//------------------------ LAM metric ---------------------->
template<typename input_type>
void rqa_CPU_LAM_metric_ref(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
) {
	int *R_matrix;
	unsigned long long int *temp_histogram;
	unsigned long long int *temp_metric;
	
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t matrix_size = corrected_size*corrected_size;
	size_t histogram_size = corrected_size + 1;
	
	R_matrix = new int[matrix_size];
	temp_histogram = new unsigned long long int[histogram_size];
	temp_metric = new unsigned long long int[histogram_size];
	
	rqa_CPU_R_matrix_ref(R_matrix, time_series, corrected_size, threshold, tau, emb, distance_type);

	for (long int r = 0; r<corrected_size; r++) {
		get_length_histogram(length_histogram, &R_matrix[r*corrected_size], corrected_size);
	}
	
	// metric
	reverseArrayAndMultiply(temp_histogram, length_histogram, histogram_size);
	serialScanInclusive(temp_metric, temp_histogram, histogram_size);
	reverseArray(metric, temp_metric, histogram_size);
	// histogram
	reverseArray(temp_histogram, length_histogram, histogram_size);
	serialScanInclusive(temp_metric, temp_histogram, histogram_size);
	reverseArray(scan_histogram, temp_metric, histogram_size);

	delete[] R_matrix;
	delete[] temp_histogram;
	delete[] temp_metric;
}

template<typename input_type>
void rqa_CPU_LAM_metric(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
) {
	
}
//----------------------------------------------------------<


//------------------------ DET metric ---------------------->
template<typename input_type>
void rqa_CPU_DET_metric_ref(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
) {
	int *R_matrix, *matrix_line;
	unsigned long long int *temp_histogram, *temp_metric;
	
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t matrix_size = corrected_size*corrected_size;
	size_t histogram_size = corrected_size + 1;
	
	R_matrix = new int[matrix_size];
	matrix_line = new int[corrected_size];
	temp_histogram = new unsigned long long int[histogram_size];
	temp_metric = new unsigned long long int[histogram_size];
	
	rqa_CPU_R_matrix_ref(R_matrix, time_series, corrected_size, threshold, tau, emb, distance_type);
	
	// upper triangle
	for (long int r = corrected_size-1; r>0; r--) {
		
		for(long int s=0; s<corrected_size; s++) matrix_line[s] = 0;
		
		for(long int s=0; s<corrected_size; s++) {
			long int row = s;
			long int column = s + r;
			if(column<corrected_size){
				size_t pos = row*corrected_size + column;
				matrix_line[s] = R_matrix[pos];
			}
		}
		get_length_histogram(length_histogram, matrix_line, corrected_size);
	}
	
	//diagonal:
	for(long int f=0; f<corrected_size; f++) matrix_line[f] = R_matrix[f*corrected_size + f];
	get_length_histogram(length_histogram, matrix_line, corrected_size);
	
	// lower triangle
	for (long int r = 1; r<corrected_size; r++) {
		for(long int s=0; s<corrected_size; s++) matrix_line[s] = 0;
		
		for(long int s=0; s<corrected_size; s++) {
			long int row = s + r;
			long int column = s;
			if(row<corrected_size){
				size_t pos = row*corrected_size + column;
				matrix_line[s] = R_matrix[pos];
			}
		}
		get_length_histogram(length_histogram, matrix_line, corrected_size);
	}
	
	// metric
	reverseArrayAndMultiply(temp_histogram, length_histogram, histogram_size);
	serialScanInclusive(temp_metric, temp_histogram, histogram_size);
	reverseArray(metric, temp_metric, histogram_size);
	// histogram
	reverseArray(temp_histogram, length_histogram, histogram_size);
	serialScanInclusive(temp_metric, temp_histogram, histogram_size);
	reverseArray(scan_histogram, temp_metric, histogram_size);


	delete[] temp_histogram;
	delete[] temp_metric;
	delete[] R_matrix;
	delete[] matrix_line;
}

template<typename input_type>
void rqa_CPU_DET_metric(
	unsigned long long int *metric, 
	unsigned long long int *scan_histogram, 
	unsigned long long int *length_histogram, 
	input_type threshold, 
	int tau, 
	int emb, 
	input_type *time_series, 
	long int input_size, 
	int distance_type
) {
	
}
//----------------------------------------------------------<

