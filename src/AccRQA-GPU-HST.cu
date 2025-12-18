#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "GPU_timer.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip> 
#include <vector>
#include <limits>

#include "../include/AccRQA_definitions.hpp"
#include "../include/AccRQA_utilities_distance.hpp"
#include "../include/AccRQA_utilities_error.hpp"
#include "../include/AccRQA_printf.hpp"
#include "GPU_scan.cuh"
#include "GPU_reduction.cuh"
#include "AccRQA_metrics.cuh"


#define DET_FAST_MUL 1

using namespace std;


#define DEBUG_GPU_HST false

// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************


template<class const_params>
__inline__ __device__ void create_start_end_arrays_compact(int *s_start, int *s_end, int *start_count, int *end_count, int first, int second, int block, long int nSamples, int *warp_scan_partial_sums){
	int gl_index = block*blockDim.x + threadIdx.x - 1;
	int address = 0, predicate = 0;

	//----------- Start -----------
	if (first == 0 && second == 1) {
		predicate = 1;
	}

	__syncthreads();	
	address = threadblock_scan_exclusive(predicate, warp_scan_partial_sums) + (*start_count);

	__syncthreads();
	if(predicate) {
		s_start[address] = gl_index + 2;
	}
	
	__syncthreads();
	if(threadIdx.x==blockDim.x-1){
		(*start_count) = address + predicate;
	}
	__syncthreads();
	
	//------------ End ----------
	predicate = 0;
	address = 0;
	if (first == 1 && second == 0) {
		predicate = 1;
	}
	
	address = threadblock_scan_exclusive(predicate, warp_scan_partial_sums) + (*end_count);
	if(predicate) {
		s_end[address] = gl_index + 2;
	}
	__syncthreads();
	if(threadIdx.x==blockDim.x-1){
		(*end_count) = address + predicate;
	}
	__syncthreads();
}

//-----------------------------------------------------------------
//---------------------------> Histogram directly from one block
__inline__ __device__ void GPU_RQA_HST_load_data(int *first, int *second, int const* __restrict__ d_input, int block, unsigned int nSamples){
	int gl_index = block*blockDim.x + threadIdx.x - 1;
	
	// loading data like this avoids border case at the beginning
	// -------------------- first ---------------------	
	if( gl_index >= 0 && gl_index < nSamples ) {
		(*first) = d_input[gl_index];
	}
	else {
		(*first)= 0;
	}
	
	// -------------------- second ---------------------
	if( (gl_index + 1) >= 0 && (gl_index + 1) < nSamples ) {
		(*second) = d_input[gl_index + 1];
	}	
	else {
		(*second) = 0;
	}
}

template<class const_params, typename IOtype>
__inline__ __device__ void GPU_RQA_HST_get_R_matrix_element_horizontal(
		int *first, 
		int *second, 
		IOtype const* __restrict__ d_input, 
		IOtype threshold, 
		int tau, 
		int emb, 
		int block, 
		long int corrected_size
	){
	long int gl_index = (long int)(block*blockDim.x + threadIdx.x) - 1;
	//float seed = d_input[blockIdx.y];
	
	// loading data like this avoids border case at the beginning
	// -------------------- first ---------------------	
	if( gl_index >= 0 && gl_index < corrected_size ) {
		//(*first) = R_element_cartesian(seed, d_input[gl_index], threshold);
		(*first) = get_RP_element<const_params>(d_input, blockIdx.x, gl_index, threshold, tau, emb);
	}
	else {
		(*first)= 0;
	}
	
	// -------------------- second ---------------------
	if( (gl_index + 1) >= 0 && (gl_index + 1) < corrected_size ) {
		//(*second) = R_element_cartesian(seed, d_input[gl_index + 1], threshold);
		(*second) = get_RP_element<const_params>(d_input, blockIdx.x, gl_index + 1, threshold, tau, emb);
	}	
	else {
		(*second) = 0;
	}
}

template<class const_params, typename IOtype>
__inline__ __device__ void GPU_RQA_HST_get_R_matrix_element_vertical(int *first, int *second, IOtype const* __restrict__ d_input, IOtype threshold, int tau, int emb, int block, long int corrected_size){
	long int gl_index = (long int)(block*blockDim.x + threadIdx.x) - 1;
	//float seed = d_input[blockIdx.y];
	
	// loading data like this avoids border case at the beginning
	// -------------------- first ---------------------	
	if( gl_index >= 0 && gl_index < corrected_size ) {
		(*first) = get_RP_element<const_params>(d_input, gl_index, blockIdx.x, threshold, tau, emb);
	}
	else {
		(*first)= 0;
	}
	
	// -------------------- second ---------------------
	if( (gl_index + 1) >= 0 && (gl_index + 1) < corrected_size ) {
		//(*second) = R_element_cartesian(d_input[gl_index + 1], seed, threshold);
		(*second) = get_RP_element<const_params>(d_input, gl_index + 1, blockIdx.x, threshold, tau, emb);
	}
	else {
		(*second) = 0;
	}
}

template<class const_params, typename IOtype>
__inline__ __device__ int GPU_RQA_HST_get_R_matrix_element_diagonal(IOtype const* __restrict__ d_input, IOtype threshold, int tau, int emb, int block, long long int gl_index, long long int corrected_size){
	long int block_y  = (long int) (blockIdx.x) - corrected_size + 1;
	
	// This is stored first as blockIdx.x < size
	// it is stored as blockIdx.x = 0 => row 0 
	if(block_y<0){
		long long int row = gl_index;
		long long int column = gl_index - block_y; //block_y is negative thus it is like + block_y
		if(row >= 0 && row < corrected_size && column  >= 0 && column < corrected_size) {
			int value = get_RP_element<const_params>(d_input, row, column, threshold, tau, emb);
			return(value);
		}
		else return(0);
	}
	
	if(block_y>0){
		long long int row = gl_index + block_y;
		long long int column = gl_index;
		if(row >= 0 && row < corrected_size && column  >= 0 && column < corrected_size) {
			int value = get_RP_element<const_params>(d_input, row, column, threshold, tau, emb);
			return(value);
		}
		else return(0);
	}
	
	if(block_y==0){ // diagonal
		long long int row = gl_index;
		long long int column = gl_index;
		if(row >= 0 && row < corrected_size && column  >= 0 && column < corrected_size) {
			int value = get_RP_element<const_params>(d_input, row, column, threshold, tau, emb);
			return(value);
		}
	}
	
	return(0);
}

template<class const_params>
__inline__ __device__ void GPU_RQA_HST_clean(int *s_start, int *s_end, int *start_count, int *end_count){
	// Setting default values
	if( threadIdx.x < const_params::shared_memory_size ){
		s_start[threadIdx.x] = 0;
		s_end[threadIdx.x] = 0;
	}
	
	if(threadIdx.x==const_params::shared_memory_size){
		(*start_count) = 0;
		(*end_count) = 0;
	}
}

template<class const_params>
__inline__ __device__ void GPU_RQA_HST_reset_data(int *s_start, int *s_end, int *start_count, int *end_count, int bl){
	// Setting default values
	__syncthreads();
	//if(threadIdx.x==0) ACCRQA_PRINT("bl=%d; start_count=%d; end_count=%d;\n", bl, (*start_count), (*end_count));
	if( (*start_count) == (*end_count) ) GPU_RQA_HST_clean<const_params>(s_start, s_end, start_count, end_count);
	else {
		if( threadIdx.x < (*end_count) ){
			s_start[threadIdx.x] = 0;
			s_end[threadIdx.x] = 0;
		}
		__syncthreads();
		if(threadIdx.x==0){
			if( (*end_count)>0 ){
				s_start[0] = s_start[(*end_count)];
				s_start[(*end_count)] = 0;
			}
			//if(s_start[0]==0) (*start_count) = 0;
			//else (*start_count) = 1;
			(*start_count) = 1;
			(*end_count) = 0;
		}
		__syncthreads();
	}
}

__device__ void GPU_RQA_HST_create_histogram(unsigned long long int *d_histogram, int *s_start, int *s_end, int end_count){
	if(threadIdx.x<end_count){
		int line_length = s_end[threadIdx.x] - s_start[threadIdx.x];
		int old_value = atomicAdd(&d_histogram[line_length], 1);
	}
}

__device__ void GPU_RQA_HST_calculate_sum(
		unsigned long long int *s_S_all, 
		unsigned long long int *s_S_lmin, 
		unsigned long long int *s_lmax, 
		unsigned long long int *s_N_lmin, 
		int *s_start, 
		int *s_end, 
		int end_count, 
		int lmin
){
	if(threadIdx.x<end_count){
		int line_length = s_end[threadIdx.x] - s_start[threadIdx.x];
		
		atomicAdd(s_S_all, line_length);
		atomicMax(s_lmax, line_length);
		if(line_length >= lmin) {
			atomicAdd(s_S_lmin, line_length);
			atomicAdd(s_N_lmin, 1);
		}
	}
}

__device__ size_t calculate_k(size_t last_block_end){
	if(last_block_end <= 0) return (0);
	double c = -2.0*last_block_end; // always negative
	double D = 1.0 - 4.0*c;
	double n1 = (__dsqrt_rd(D)-1.0)/2.0;
	return( (size_t) n1);
}

__device__ void calculate_coordinates(size_t *i, size_t *j, size_t pos, size_t corrected_size){
	size_t pk = calculate_k(pos);
	*i = pos - (pk*(pk-1))/2 - pk;
	*j = corrected_size - pk - 1 + *i;
}

//---------------------------------------------------------------------------------
//-------------------------- GPU kernels ------------------------------------------
//---------------------------------------------------------------------------------

template<typename Type>
__global__ void reverse_array_and_multiply(Type *d_output, Type *d_input, unsigned int nElements){
	int read_index  = nElements - 1 - blockIdx.x*blockDim.x - threadIdx.x;
	int write_index = blockIdx.x*blockDim.x + threadIdx.x;
	if(read_index>=0 && write_index<nElements) {
		d_output[write_index] = read_index*d_input[read_index];
	}
}

template<typename Type>
__global__ void reverse_array(Type *d_output, Type *d_input, unsigned int nElements){
	int read_index  = nElements - 1 - blockIdx.x*blockDim.x - threadIdx.x;
	int write_index = blockIdx.x*blockDim.x + threadIdx.x;
	if(read_index>=0 && write_index<nElements) {
		d_output[write_index] = d_input[read_index];
	}
}

template<typename Type>
__global__ void GPU_RQA_HST_correction_diagonal_half(Type *d_histogram, unsigned int nElements){
	int read_index  = blockIdx.x*blockDim.x + threadIdx.x;
	// Correction because we created histogram from only half of the recurrent matrix
	if(read_index >= 0 && read_index < (nElements - 1)) {
		d_histogram[read_index] = 2*d_histogram[read_index];
	}
	// Correction because we did not include diagonal
	if(read_index == (nElements - 1)) {
		d_histogram[read_index] = d_histogram[read_index] + 1;
	}
}


template<class const_params, typename IOtype>
__global__ void GPU_RQA_generate_diagonal_R_matrix_kernel(int *d_diagonal_R_matrix, IOtype const* __restrict__ d_input, IOtype threshold, int tau, int emb, long int corrected_size){
	int nBlocks = (corrected_size/blockDim.x) + 1;
	for(int bl=0; bl<nBlocks; bl++){
		long int gl_index = (long int)(bl*blockDim.x + threadIdx.x) - 1;
		int value = GPU_RQA_HST_get_R_matrix_element_diagonal<const_params>( d_input, threshold, tau, emb, bl, gl_index, corrected_size);
	
		if(gl_index>=0 && gl_index<corrected_size) {
			d_diagonal_R_matrix[blockIdx.x*corrected_size + gl_index] = value;
		}
	}
}

template<class const_params>
__global__ void GPU_RQA_HST_length_histogram_direct(unsigned long long int *d_histogram, int const* __restrict__ d_input, unsigned int nSamples){
	__shared__ int s_start[const_params::shared_memory_size];
	__shared__ int s_end[const_params::shared_memory_size];
	__shared__ int warp_scan_partial_sums[const_params::warp];
	__shared__ int start_count;
	__shared__ int end_count;
	
	// read or create data and store them into shared memory
	int first, second;
	int nBlocks = (nSamples/blockDim.x) + 1;
	
	GPU_RQA_HST_clean<const_params>(s_start, s_end, &start_count, &end_count);
	
	for(int bl=0; bl<nBlocks; bl++){
		GPU_RQA_HST_load_data(&first, &second, d_input, bl, nSamples);	
		
		create_start_end_arrays_compact<const_params>(s_start, s_end, &start_count, &end_count, first, second, bl, nSamples, warp_scan_partial_sums);
		__syncthreads();
		
		// Now we need to process the histogram
//		if(threadIdx.x<start_count){
//			ACCRQA_PRINT("th:%d; bl=%d; s_start=%d; s_end=%d; s_end[]=%d; s_start[]=%d; length=%d;\n", threadIdx.x, bl, start_count, end_count, s_end[threadIdx.x], s_start[threadIdx.x], s_end[threadIdx.x] - s_start[threadIdx.x]);
//		}
		GPU_RQA_HST_create_histogram(d_histogram, s_start, s_end, end_count);
		__syncthreads();
		// in case the line started in this block but did not end 
		// the number of starts > number of ends
		// The histogram will process only closed lines, that is it will use end_count.
		GPU_RQA_HST_reset_data<const_params>(s_start, s_end, &start_count, &end_count, bl);
		
		// Now we can start the loop again
	}
}

template<class const_params, typename IOtype>
__global__ void GPU_RQA_HST_length_histogram_horizontal(
		unsigned long long int *d_histogram, 
		IOtype const* __restrict__ d_input, 
		IOtype threshold, 
		int tau, 
		int emb, 
		long int nSamples
	){
	__shared__ int s_start[const_params::shared_memory_size];
	__shared__ int s_end[const_params::shared_memory_size];
	__shared__ int warp_scan_partial_sums[const_params::warp];
	__shared__ int start_count;
	__shared__ int end_count;
	
	// read or create data and store them into shared memory
	int first, second;
	int nBlocks = (nSamples/blockDim.x) + 1;
	
	GPU_RQA_HST_clean<const_params>(s_start, s_end, &start_count, &end_count);
	
	for(int bl=0; bl<nBlocks; bl++){
		GPU_RQA_HST_get_R_matrix_element_horizontal<const_params>(&first, &second, d_input, threshold, tau, emb, bl, nSamples);
		
		create_start_end_arrays_compact<const_params>(s_start, s_end, &start_count, &end_count, first, second, bl, nSamples, warp_scan_partial_sums);
		__syncthreads();
		
		// Now we need to process the histogram
		GPU_RQA_HST_create_histogram(d_histogram, s_start, s_end, end_count);
		__syncthreads();
		// in case the line started in this block but did not end 
		// the number of starts > number of ends
		// The histogram will process only closed lines, that is it will use end_count.
		GPU_RQA_HST_reset_data<const_params>(s_start, s_end, &start_count, &end_count, bl);
		
		// Now we can start the loop again
	}
}

template<class const_params, typename IOtype>
__global__ void GPU_RQA_HST_length_histogram_vertical(
		unsigned long long int *d_histogram, 
		IOtype const* __restrict__ d_input, 
		IOtype threshold, 
		int tau, 
		int emb, 
		long int nSamples
	){
	__shared__ int s_start[const_params::shared_memory_size];
	__shared__ int s_end[const_params::shared_memory_size];
	__shared__ int warp_scan_partial_sums[const_params::warp];
	__shared__ int start_count;
	__shared__ int end_count;
	
	// read or create data and store them into shared memory
	int first, second;
	int nBlocks = (nSamples/blockDim.x) + 1;
	
	GPU_RQA_HST_clean<const_params>(s_start, s_end, &start_count, &end_count);
	
	for(int bl=0; bl<nBlocks; bl++){
		GPU_RQA_HST_get_R_matrix_element_vertical<const_params>(&first, &second, d_input, threshold, tau, emb, bl, nSamples);
		
		create_start_end_arrays_compact<const_params>(s_start, s_end, &start_count, &end_count, first, second, bl, nSamples, warp_scan_partial_sums);
		__syncthreads();
		
		// Now we need to process the histogram
		GPU_RQA_HST_create_histogram(d_histogram, s_start, s_end, end_count);
		__syncthreads();
		// in case the line started in this block but did not end 
		// the number of starts > number of ends
		// The histogram will process only closed lines, that is it will use end_count.
		GPU_RQA_HST_reset_data<const_params>(s_start, s_end, &start_count, &end_count, bl);
		
		// Now we can start the loop again
	}
}

template<class const_params, typename IOtype>
__global__ void GPU_RQA_HST_length_histogram_diagonal(
		unsigned long long int *d_histogram, 
		IOtype const* __restrict__ d_input, 
		IOtype threshold, 
		int tau, 
		int emb, 
		long int nSamples
	){
	__shared__ int s_start[const_params::shared_memory_size];
	__shared__ int s_end[const_params::shared_memory_size];
	__shared__ int warp_scan_partial_sums[const_params::warp];
	__shared__ int start_count;
	__shared__ int end_count;
	
	// read or create data and store them into shared memory
	int first, second;
	int nBlocks = (nSamples/blockDim.x) + 1;
	
	GPU_RQA_HST_clean<const_params>(s_start, s_end, &start_count, &end_count);
	for(int bl=0; bl<nBlocks; bl++){
		// Generate data for diagonal LAM metric
		long int gl_index = (long int)(bl*blockDim.x + threadIdx.x) - 1;
		first  = GPU_RQA_HST_get_R_matrix_element_diagonal<const_params>(d_input, threshold, tau, emb, bl, gl_index, nSamples);
		second = GPU_RQA_HST_get_R_matrix_element_diagonal<const_params>(d_input, threshold, tau, emb, bl, gl_index + 1, nSamples);
		
		create_start_end_arrays_compact<const_params>(s_start, s_end, &start_count, &end_count, first, second, bl, nSamples, warp_scan_partial_sums);
		__syncthreads();
		
		// Now we need to process the histogram
		GPU_RQA_HST_create_histogram(d_histogram, s_start, s_end, end_count);
		__syncthreads();
		// in case the line started in this block but did not end 
		// the number of starts > number of ends
		// The histogram will process only closed lines, that is it will use end_count.
		GPU_RQA_HST_reset_data<const_params>(s_start, s_end, &start_count, &end_count, bl);
		
		// Now we can start the loop again
		if(bl*blockDim.x > blockIdx.x) break;
	}
}

template<class const_params, typename IOtype>
__global__ void GPU_RQA_HST_length_histogram_diagonal_sum(
		unsigned long long int *d_S_all, 
		unsigned long long int *d_S_lmin, 
		unsigned long long int *d_lmax, 
		unsigned long long int *d_N_lmin, 
		IOtype const* __restrict__ d_input, 
		IOtype threshold, 
		int tau, 
		int emb, 
		int lmin, 
		long int nSamples
	){
	__shared__ int s_start[const_params::shared_memory_size];
	__shared__ int s_end[const_params::shared_memory_size];
	__shared__ int warp_scan_partial_sums[const_params::warp];
	__shared__ int start_count;
	__shared__ int end_count;
	__shared__ unsigned long long int s_S_all;
	__shared__ unsigned long long int s_S_lmin; 
	__shared__ unsigned long long int s_lmax; 
	__shared__ unsigned long long int s_N_lmin; 
	
	if (threadIdx.x==0) {
		s_S_all = 0;
		s_S_lmin = 0;
		s_lmax = 0;
		s_N_lmin = 0;
	}
	
	// read or create data and store them into shared memory
	int first, second;
	int nBlocks = (nSamples/blockDim.x) + 1;
	
	
	GPU_RQA_HST_clean<const_params>(s_start, s_end, &start_count, &end_count);
	for(int bl=0; bl<nBlocks; bl++){
		// Generate data for diagonal LAM metric
		long int gl_index = (long int)(bl*blockDim.x + threadIdx.x) - 1;
		first  = GPU_RQA_HST_get_R_matrix_element_diagonal<const_params>(d_input, threshold, tau, emb, bl, gl_index, nSamples);
		second = GPU_RQA_HST_get_R_matrix_element_diagonal<const_params>(d_input, threshold, tau, emb, bl, gl_index + 1, nSamples);
		
		create_start_end_arrays_compact<const_params>(s_start, s_end, &start_count, &end_count, first, second, bl, nSamples, warp_scan_partial_sums);
		__syncthreads();
		
		// Now we need to process the histogram
		GPU_RQA_HST_calculate_sum(&s_S_all, &s_S_lmin, &s_lmax, &s_N_lmin, s_start, s_end, end_count, lmin);
		__syncthreads();
		// in case the line started in this block but did not end 
		// the number of starts > number of ends
		// The histogram will process only closed lines, that is it will use end_count.
		GPU_RQA_HST_reset_data<const_params>(s_start, s_end, &start_count, &end_count, bl);
		
		// Now we can start the loop again
		if(bl*blockDim.x > blockIdx.x) break;
	}
	
	if(threadIdx.x==0){
		atomicAdd(&d_S_all[0], s_S_all);
		atomicAdd(&d_S_lmin[0], s_S_lmin);
		atomicMax(&d_lmax[0], s_lmax);
		atomicAdd(&d_N_lmin[0], s_N_lmin);
	}
}

template<class const_params, typename IOtype>
__global__ void GPU_RQA_DET_boxcar(
		unsigned long long int *d_S_all, 
		unsigned long long int *d_S_lmin, 
		unsigned long long int *d_N_lmin, 
		IOtype const* __restrict__ d_input, 
		IOtype threshold, 
		int tau, 
		int emb, 
		int lmin, 
		size_t corrected_size
){
	extern __shared__ int R_values[];
	__shared__ int warp_scan_partial_sums[3*const_params::warp];
	
	// Populate shared memory with values
	size_t elements_per_block = DET_FAST_MUL*blockDim.x - lmin;
	size_t global_pos = blockIdx.x*elements_per_block + threadIdx.x;
	size_t total_elements = ((corrected_size-1)*corrected_size)/2;
	int sum_before = 0;
	for(size_t f=0; f<DET_FAST_MUL; f++){
		int s_pos = blockDim.x*f + threadIdx.x;
		size_t i = 0, j = corrected_size-1;
		if(global_pos < total_elements) {
			calculate_coordinates(
				&i, &j, 
				global_pos + f*blockDim.x, corrected_size
			);
		}
		int R_value;
		if(const_params::dst_type==DST_EUCLIDEAN){
			R_value= R_element_euc(
				d_input,
				i, j,
				threshold, tau, emb
			);
		}
		else if(const_params::dst_type==DST_MAXIMAL){
			R_value= R_element_max(
				d_input,
				i, j,
				threshold, tau, emb
			);
		}
		R_values[s_pos] = R_value;
		if(s_pos > 0 && s_pos < (DET_FAST_MUL*blockDim.x - lmin + 1)) {
			sum_before += R_value;
		}
	}
	__syncthreads();
	
	
	// ==================> Apply boxcar filter
	int boxcar[DET_FAST_MUL];
	for(int f=0; f<DET_FAST_MUL - 1; f++){
		int s_pos = f*blockDim.x + threadIdx.x;
		boxcar[f] = R_values[s_pos];
		for(int l=1; l<lmin; l++){
			boxcar[f] += R_values[s_pos + l];
		}
	}
	boxcar[DET_FAST_MUL - 1] = 0;
	if(threadIdx.x <= blockDim.x - lmin){
		int s_pos = (DET_FAST_MUL - 1)*blockDim.x + threadIdx.x;
		boxcar[DET_FAST_MUL - 1] = R_values[s_pos];
		for(int l=1; l<lmin; l++){
			boxcar[DET_FAST_MUL - 1] += R_values[s_pos + l];
		}
	}
	__syncthreads();
	for(int f=0; f<DET_FAST_MUL; f++){
		int s_pos = f*blockDim.x + threadIdx.x;
		R_values[s_pos] = (int) (boxcar[f]/lmin);
	}
	__syncthreads();
	//======================================<
	
	
	
	// ==================> Apply correction
	int first = 0, second = 0;
	int sum_after = 0, num_lines = 0;
	for(int f=0; f<DET_FAST_MUL - 1; f++){
		int s_pos = f*blockDim.x + threadIdx.x;
		first  = R_values[s_pos];
		second = R_values[s_pos + 1];
		if(first == 0 && second == 1) {
			second = second*lmin;
			num_lines++;
		}
		sum_after += second;
	}
	// last iteration is unique because last lmin threads do not participate
	first = 0; second = 0;
	if(threadIdx.x <= blockDim.x - lmin){
		int s_pos = (DET_FAST_MUL - 1)*blockDim.x + threadIdx.x;
		first  = R_values[s_pos];
		second = R_values[s_pos + 1];
		if(first == 0 && second == 1) {
			second = second*lmin;
			num_lines++;
		}
		sum_after += second;
	}
	//======================================<
	
	
	
	// ==================> Reductions
	Reduce_WARP(&sum_after);
	Reduce_WARP(&sum_before);
	Reduce_WARP(&num_lines);
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	if(local_id == 0){
		warp_scan_partial_sums[warp_id] = sum_after;
		warp_scan_partial_sums[warp_id + const_params::warp] = sum_before;
		warp_scan_partial_sums[warp_id + 2*const_params::warp] = num_lines;
	}
	__syncthreads();
	if(warp_id==0){
		sum_after = warp_scan_partial_sums[local_id];
		sum_before = warp_scan_partial_sums[const_params::warp + local_id];
		num_lines = warp_scan_partial_sums[2*const_params::warp + local_id];
		Reduce_WARP(&sum_after);
		Reduce_WARP(&sum_before);
		Reduce_WARP(&num_lines);
		if(threadIdx.x==0){
			atomicAdd(&d_S_all[0], sum_before);
			atomicAdd(&d_S_lmin[0], sum_after);
			atomicAdd(&d_N_lmin[0], num_lines);
		}
	}
	//======================================<
}


template<class const_params, typename IOtype>
__global__ void GPU_RQA_DET_boxcar_square_double(
		unsigned long long int *d_S_all, 
		unsigned long long int *d_S_lmin, 
		unsigned long long int *d_N_lmin, 
		IOtype const* __restrict__ d_input, 
		double threshold, 
		int tau, 
		int emb, 
		int lmin, 
		size_t corrected_size
){
	__shared__ int R_values[const_params::width*const_params::width];
	__shared__ int warp_scan_partial_sums[3*const_params::warp];
	// problem is tau and emb therefore I need larger cache
	extern __shared__ double dcached[];
	int cache_size = const_params::width + (emb - 1)*tau;
	
	if(blockIdx.y > blockIdx.x) return; // Calculating only upper triangle
	
	size_t global_pos_x = blockIdx.x*(32 - lmin) + threadIdx.x;
	size_t global_pos_y = blockIdx.y*(32 - lmin) + threadIdx.x;
	
	if(threadIdx.x < cache_size) {
		size_t input_size = corrected_size + (emb - 1)*tau;
		if(global_pos_y < input_size){
			// i-th elements that is row
			dcached[threadIdx.x] = d_input[global_pos_y];
		}
		if(global_pos_x < input_size){
			// j-th elements that is column
			dcached[cache_size + threadIdx.x] = d_input[global_pos_x];
		}
	}
	__syncthreads();
	

	// Calculate R values and store them
	int sum_before = 0;
	int th_x = threadIdx.x&31;
	int th_y = threadIdx.x>>5;
	int R_value = 0;
	if(
		th_y + blockIdx.y*(32 - lmin) < corrected_size &&
		th_x + blockIdx.x*(32 - lmin) < corrected_size
	){
		R_value = get_RP_element_cache<const_params>(dcached, th_y, th_x, threshold, tau, emb, cache_size);
		
		//if(const_params::dst_type==DST_EUCLIDEAN){
		//	R_value = R_element_euc_cache(
		//		dcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
		//else if(const_params::dst_type==DST_MAXIMAL){
		//	R_value = R_element_max_cache(
		//		dcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
	}
	int s_pos = th_y*32 + th_x;
	R_values[s_pos] = R_value;
	__syncthreads();
	

	if(
		th_x > 0 && th_x < (32 - lmin + 1) &&
		th_y > 0 && th_y < (32 - lmin + 1)
	) {
		sum_before += R_value;
	}
	__syncthreads();
	
	
	// ==================> Apply boxcar filter
	int boxcar;
	s_pos = th_y*32 + th_x;
	boxcar = R_values[s_pos];
	// Doing DET
	for(int l=1; l<lmin; l++){
		// Calculating new coordinates for boxcar filter
		// If these overflow it does not matter because of the apron data
		int new_th_y = (th_y + l)&(const_params::width-1);
		int new_th_x = (th_x + l)&(const_params::width-1);
		s_pos = new_th_y*32 + new_th_x;
		boxcar += R_values[s_pos];
	}
	__syncthreads();
	s_pos = th_y*32 + th_x;
	R_values[s_pos] = (int) (boxcar/lmin);
	__syncthreads();
	//======================================<
	
	
	
	// ==================> Apply correction
	int first = 0, second = 0;
	int sum_after = 0, num_lines = 0;
	if(
		th_x > 0 && th_x < (32 - lmin + 1) &&
		th_y > 0 && th_y < (32 - lmin + 1)
	) {
		s_pos = th_y*32 + th_x;
		first  = R_values[(th_y - 1)*32 + (th_x - 1)];
		second = R_values[s_pos];
		if(first == 0 && second == 1) {
			second = second*lmin;
			num_lines++;
		}
		sum_after += second;
		
		R_values[s_pos] = second;
	}
	//======================================<
	
	int g_pos_x = blockIdx.x*(32 - lmin) + th_x;
	int g_pos_y = blockIdx.y*(32 - lmin) + th_y;
	if( g_pos_y >= g_pos_x){
		sum_before = 0;
		sum_after = 0;
		num_lines = 0;
	}
	// ==================> Reductions
	Reduce_WARP(&sum_after);
	Reduce_WARP(&sum_before);
	Reduce_WARP(&num_lines);
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	if(local_id == 0){
		warp_scan_partial_sums[warp_id] = sum_after;
		warp_scan_partial_sums[warp_id + const_params::warp] = sum_before;
		warp_scan_partial_sums[warp_id + 2*const_params::warp] = num_lines;
	}
	__syncthreads();
	if(warp_id==0){
		sum_after = warp_scan_partial_sums[local_id];
		sum_before = warp_scan_partial_sums[const_params::warp + local_id];
		num_lines = warp_scan_partial_sums[2*const_params::warp + local_id];
		Reduce_WARP(&sum_after);
		Reduce_WARP(&sum_before);
		Reduce_WARP(&num_lines);
		if(threadIdx.x==0){
			atomicAdd(&d_S_all[0], sum_before);
			atomicAdd(&d_S_lmin[0], sum_after);
			atomicAdd(&d_N_lmin[0], num_lines);
		}
	}
	//======================================<
}


template<class const_params, typename IOtype>
__global__ void GPU_RQA_DET_boxcar_square_float(
		unsigned long long int *d_S_all, 
		unsigned long long int *d_S_lmin, 
		unsigned long long int *d_N_lmin, 
		IOtype const* __restrict__ d_input, 
		float threshold, 
		int tau, 
		int emb, 
		int lmin, 
		size_t corrected_size
){
	__shared__ int R_values[const_params::width*const_params::width];
	__shared__ int warp_scan_partial_sums[3*const_params::warp];
	// problem is tau and emb therefore I need larger cache
	extern __shared__ float fcached[];
	int cache_size = const_params::width + (emb - 1)*tau;
	
	if(blockIdx.y > blockIdx.x) return; // Calculating only upper triangle
	
	size_t global_pos_x = blockIdx.x*(32 - lmin) + threadIdx.x;
	size_t global_pos_y = blockIdx.y*(32 - lmin) + threadIdx.x;
	
	if(threadIdx.x < cache_size) {
		size_t input_size = corrected_size + (emb - 1)*tau;
		if(global_pos_y < input_size){
			// i-th elements that is row
			fcached[threadIdx.x] = d_input[global_pos_y];
		}
		if(global_pos_x < input_size){
			// j-th elements that is column
			fcached[cache_size + threadIdx.x] = d_input[global_pos_x];
		}
	}
	__syncthreads();
	

	// Calculate R values and store them
	int sum_before = 0;
	int th_x = threadIdx.x&31;
	int th_y = threadIdx.x>>5;
	int R_value = 0;
	if(
		th_y + blockIdx.y*(32 - lmin) < corrected_size &&
		th_x + blockIdx.x*(32 - lmin) < corrected_size
	){
		R_value = get_RP_element_cache<const_params>(fcached, th_y, th_x, threshold, tau, emb, cache_size);
		
		//if(const_params::dst_type==DST_EUCLIDEAN){
		//	R_value = R_element_euc_cache(
		//		fcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
		//else if(const_params::dst_type==DST_MAXIMAL){
		//	R_value = R_element_max_cache(
		//		fcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
	}
	int s_pos = th_y*32 + th_x;
	R_values[s_pos] = R_value;
	__syncthreads();
	

	if(
		th_x > 0 && th_x < (32 - lmin + 1) &&
		th_y > 0 && th_y < (32 - lmin + 1)
	) {
		sum_before += R_value;
	}
	__syncthreads();
	
	
	// ==================> Apply boxcar filter
	int boxcar;
	s_pos = th_y*32 + th_x;
	boxcar = R_values[s_pos];
	// Doing DET
	for(int l=1; l<lmin; l++){
		// Calculating new coordinates for boxcar filter
		// If these overflow it does not matter because of the apron data
		int new_th_y = (th_y + l)&(const_params::width-1);
		int new_th_x = (th_x + l)&(const_params::width-1);
		s_pos = new_th_y*32 + new_th_x;
		boxcar += R_values[s_pos];
	}
	__syncthreads();
	s_pos = th_y*32 + th_x;
	R_values[s_pos] = (int) (boxcar/lmin);
	__syncthreads();
	//======================================<
	
	
	
	// ==================> Apply correction
	int first = 0, second = 0;
	int sum_after = 0, num_lines = 0;
	if(
		th_x > 0 && th_x < (32 - lmin + 1) &&
		th_y > 0 && th_y < (32 - lmin + 1)
	) {
		s_pos = th_y*32 + th_x;
		first  = R_values[(th_y - 1)*32 + (th_x - 1)];
		second = R_values[s_pos];
		if(first == 0 && second == 1) {
			second = second*lmin;
			num_lines++;
		}
		sum_after += second;
		
		R_values[s_pos] = second;
	}
	//======================================<
	
	int g_pos_x = blockIdx.x*(32 - lmin) + th_x;
	int g_pos_y = blockIdx.y*(32 - lmin) + th_y;
	if( g_pos_y >= g_pos_x){
		sum_before = 0;
		sum_after = 0;
		num_lines = 0;
	}
	// ==================> Reductions
	Reduce_WARP(&sum_after);
	Reduce_WARP(&sum_before);
	Reduce_WARP(&num_lines);
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	if(local_id == 0){
		warp_scan_partial_sums[warp_id] = sum_after;
		warp_scan_partial_sums[warp_id + const_params::warp] = sum_before;
		warp_scan_partial_sums[warp_id + 2*const_params::warp] = num_lines;
	}
	__syncthreads();
	if(warp_id==0){
		sum_after = warp_scan_partial_sums[local_id];
		sum_before = warp_scan_partial_sums[const_params::warp + local_id];
		num_lines = warp_scan_partial_sums[2*const_params::warp + local_id];
		Reduce_WARP(&sum_after);
		Reduce_WARP(&sum_before);
		Reduce_WARP(&num_lines);
		if(threadIdx.x==0){
			atomicAdd(&d_S_all[0], sum_before);
			atomicAdd(&d_S_lmin[0], sum_after);
			atomicAdd(&d_N_lmin[0], num_lines);
		}
	}
	//======================================<
}


template<class const_params, typename IOtype>
__global__ void GPU_RQA_LAM_boxcar_square_double(
		unsigned long long int *d_S_all, 
		unsigned long long int *d_S_vmin, 
		unsigned long long int *d_N_vmin, 
		IOtype const* __restrict__ d_input, 
		double threshold, 
		int tau, 
		int emb, 
		int vmin, 
		size_t corrected_size
){
	__shared__ int R_values[const_params::width*const_params::width];
	__shared__ int warp_scan_partial_sums[3*const_params::warp];
	// problem is tau and emb therefore I need larger cache
	extern __shared__ double dlcached[];
	int cache_size = const_params::width + (emb - 1)*tau;
	
	size_t global_pos_x = blockIdx.x*(32 - vmin) + threadIdx.x;
	size_t global_pos_y = blockIdx.y*(32 - vmin) + threadIdx.x;
	
	if(threadIdx.x < cache_size) {
		size_t input_size = corrected_size + (emb - 1)*tau;
		if(global_pos_y < input_size){
			// i-th elements that is row
			dlcached[threadIdx.x] = d_input[global_pos_y];
		}
		if(global_pos_x < input_size){
			// j-th elements that is column
			dlcached[cache_size + threadIdx.x] = d_input[global_pos_x];
		}
	}
	__syncthreads();
	
	// Calculate R values and store them
	int sum_before = 0;
	int th_x = threadIdx.x&31;
	int th_y = threadIdx.x>>5;
	int R_value = 0;
	if(
		th_y + blockIdx.y*(32 - vmin) < corrected_size &&
		th_x + blockIdx.x*(32 - vmin) < corrected_size
	){
		R_value = get_RP_element_cache<const_params>(dlcached, th_y, th_x, threshold, tau, emb, cache_size);
		
		//if(const_params::dst_type==DST_EUCLIDEAN){
		//	R_value = R_element_euc_cache(
		//		dlcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
		//else if(const_params::dst_type==DST_MAXIMAL){
		//	R_value = R_element_max_cache(
		//		dlcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
	}
	int s_pos = th_y*32 + th_x;
	R_values[s_pos] = R_value;
	__syncthreads();
	
	if(
		th_x > 0 && th_x < (32 - vmin + 1) &&
		th_y > 0 && th_y < (32 - vmin + 1)
	) {
		sum_before += R_value;
	}
	__syncthreads();
	
	
	// ==================> Apply boxcar filter
	int boxcar;
	s_pos = th_y*32 + th_x;
	boxcar = R_values[s_pos];
	// Doing DET
	for(int l=1; l<vmin; l++){
		// Calculating new coordinates for boxcar filter
		// If these overflow it does not matter because of the apron data
		int new_th_x = (th_x + l)&(const_params::width-1);
		s_pos = th_y*32 + new_th_x;
		boxcar += R_values[s_pos];
	}
	__syncthreads();
	s_pos = th_y*32 + th_x;
	R_values[s_pos] = (int) (boxcar/vmin);
	__syncthreads();
	//======================================<
	
	
	
	// ==================> Apply correction
	int first = 0, second = 0;
	int sum_after = 0, num_lines = 0;
	if(
		th_x > 0 && th_x < (32 - vmin + 1) &&
		th_y > 0 && th_y < (32 - vmin + 1)
	) {
		s_pos = th_y*32 + th_x;
		first  = R_values[th_y*32 + (th_x - 1)];
		second = R_values[s_pos];
		if(first == 0 && second == 1) {
			second = second*vmin;
			num_lines++;
		}
		sum_after += second;
		
		R_values[s_pos] = second;
	}
	//======================================<
	
	// ==================> Reductions
	Reduce_WARP(&sum_after);
	Reduce_WARP(&sum_before);
	Reduce_WARP(&num_lines);
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	if(local_id == 0){
		warp_scan_partial_sums[warp_id] = sum_after;
		warp_scan_partial_sums[warp_id + const_params::warp] = sum_before;
		warp_scan_partial_sums[warp_id + 2*const_params::warp] = num_lines;
	}
	__syncthreads();
	if(warp_id==0){
		sum_after = warp_scan_partial_sums[local_id];
		sum_before = warp_scan_partial_sums[const_params::warp + local_id];
		num_lines = warp_scan_partial_sums[2*const_params::warp + local_id];
		Reduce_WARP(&sum_after);
		Reduce_WARP(&sum_before);
		Reduce_WARP(&num_lines);
		if(threadIdx.x==0){
			atomicAdd(&d_S_all[0], sum_before);
			atomicAdd(&d_S_vmin[0], sum_after);
			atomicAdd(&d_N_vmin[0], num_lines);
		}
	}
	//======================================<
}


template<class const_params, typename IOtype>
__global__ void GPU_RQA_LAM_boxcar_square_float(
		unsigned long long int *d_S_all, 
		unsigned long long int *d_S_vmin, 
		unsigned long long int *d_N_vmin, 
		IOtype const* __restrict__ d_input, 
		float threshold, 
		int tau, 
		int emb, 
		int vmin, 
		size_t corrected_size
){
	__shared__ int R_values[const_params::width*const_params::width];
	__shared__ int warp_scan_partial_sums[3*const_params::warp];
	// problem is tau and emb therefore I need larger cache
	extern __shared__ float flcached[];
	int cache_size = const_params::width + (emb - 1)*tau;
	
	size_t global_pos_x = blockIdx.x*(32 - vmin) + threadIdx.x;
	size_t global_pos_y = blockIdx.y*(32 - vmin) + threadIdx.x;
	
	if(threadIdx.x < cache_size) {
		size_t input_size = corrected_size + (emb - 1)*tau;
		if(global_pos_y < input_size){
			// i-th elements that is row
			flcached[threadIdx.x] = d_input[global_pos_y];
		}
		if(global_pos_x < input_size){
			// j-th elements that is column
			flcached[cache_size + threadIdx.x] = d_input[global_pos_x];
		}
	}
	__syncthreads();
	
	// Calculate R values and store them
	int sum_before = 0;
	int th_x = threadIdx.x&31;
	int th_y = threadIdx.x>>5;
	int R_value = 0;
	if(
		th_y + blockIdx.y*(32 - vmin) < corrected_size &&
		th_x + blockIdx.x*(32 - vmin) < corrected_size
	){
		R_value = get_RP_element_cache<const_params>(flcached, th_y, th_x, threshold, tau, emb, cache_size);
		
		//if(const_params::dst_type==DST_EUCLIDEAN){
		//	R_value = R_element_euc_cache(
		//		flcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
		//else if(const_params::dst_type==DST_MAXIMAL){
		//	R_value = R_element_max_cache(
		//		flcached,
		//		th_y, // row
		//		th_x, // column
		//		threshold, tau, emb, cache_size
		//	);
		//}
	}
	int s_pos = th_y*32 + th_x;
	R_values[s_pos] = R_value;
	__syncthreads();
	
	if(
		th_x > 0 && th_x < (32 - vmin + 1) &&
		th_y > 0 && th_y < (32 - vmin + 1)
	) {
		sum_before += R_value;
	}
	__syncthreads();
	
	
	// ==================> Apply boxcar filter
	int boxcar;
	s_pos = th_y*32 + th_x;
	boxcar = R_values[s_pos];
	// Doing DET
	for(int l=1; l<vmin; l++){
		// Calculating new coordinates for boxcar filter
		// If these overflow it does not matter because of the apron data
		int new_th_x = (th_x + l)&(const_params::width-1);
		s_pos = th_y*32 + new_th_x;
		boxcar += R_values[s_pos];
	}
	__syncthreads();
	s_pos = th_y*32 + th_x;
	R_values[s_pos] = (int) (boxcar/vmin);
	__syncthreads();
	//======================================<
	
	
	
	// ==================> Apply correction
	int first = 0, second = 0;
	int sum_after = 0, num_lines = 0;
	if(
		th_x > 0 && th_x < (32 - vmin + 1) &&
		th_y > 0 && th_y < (32 - vmin + 1)
	) {
		s_pos = th_y*32 + th_x;
		first  = R_values[th_y*32 + (th_x - 1)];
		second = R_values[s_pos];
		if(first == 0 && second == 1) {
			second = second*vmin;
			num_lines++;
		}
		sum_after += second;
		
		R_values[s_pos] = second;
	}
	//======================================<
	
	// ==================> Reductions
	Reduce_WARP(&sum_after);
	Reduce_WARP(&sum_before);
	Reduce_WARP(&num_lines);
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	if(local_id == 0){
		warp_scan_partial_sums[warp_id] = sum_after;
		warp_scan_partial_sums[warp_id + const_params::warp] = sum_before;
		warp_scan_partial_sums[warp_id + 2*const_params::warp] = num_lines;
	}
	__syncthreads();
	if(warp_id==0){
		sum_after = warp_scan_partial_sums[local_id];
		sum_before = warp_scan_partial_sums[const_params::warp + local_id];
		num_lines = warp_scan_partial_sums[2*const_params::warp + local_id];
		Reduce_WARP(&sum_after);
		Reduce_WARP(&sum_before);
		Reduce_WARP(&num_lines);
		if(threadIdx.x==0){
			atomicAdd(&d_S_all[0], sum_before);
			atomicAdd(&d_S_vmin[0], sum_after);
			atomicAdd(&d_N_vmin[0], num_lines);
		}
	}
	//======================================<
}





//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------


void RQA_HST_init(){
	//---------> Specific nVidia stuff
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
}

//---------------------------- WRAPERS -----------------------------

int GPU_scan_exclusive(unsigned int *d_output, unsigned int *d_input, unsigned int nElements, int nTimeseries){
	//---------> Task specific
	int nThreads = 1024;
	int nBlocks_x = (nElements + nThreads - 1)/nThreads;
	
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize(nBlocks_x, nTimeseries, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG_GPU_HST) ACCRQA_PRINT("\n");
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	unsigned int *d_partial_sums;
	if(nBlocks_x==1) {
		d_partial_sums = NULL;
	}
	else {
		int partial_sum_size = nBlocks_x*nTimeseries*sizeof(int);
		cudaMalloc((void **) &d_partial_sums,  partial_sum_size);
		cudaMemset(d_partial_sums, 0, partial_sum_size);
	}
		
	// ----------------------------------------------->
	// --------> Measured part GPU scan
	//---------> GPU scan
	if(d_partial_sums!=NULL) {
		GPU_scan_warp<Scan_inclusive><<<gridSize,blockSize>>>(d_output, d_input, nElements, d_partial_sums);
		GPU_scan_grid_followup<Scan_exclusive><<<gridSize,blockSize>>>(d_output, d_input, d_partial_sums, nElements);
	}
	else {
		GPU_scan_warp<Scan_exclusive><<<gridSize,blockSize>>>(d_output, d_input, nElements, d_partial_sums);
	}
	// --------> Measured part GPU scan
	// ----------------------------------------------->
	
	if(d_partial_sums!=NULL) cudaFree(d_partial_sums);
	return(0);
}


int GPU_scan_inclusive(unsigned long long int *d_output, unsigned long long int *d_input, size_t nElements, size_t nTimeseries){
	//---------> Task specific
	size_t nThreads = 1024;
	size_t nBlocks_x = (nElements + nThreads - 1)/nThreads;
	
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize(nBlocks_x, nTimeseries, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG_GPU_HST) ACCRQA_PRINT("\n");
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	unsigned long long int *d_partial_sums;
	if(nBlocks_x==1) {
		d_partial_sums = NULL;
	}
	else {
		size_t partial_sum_size = nBlocks_x*nTimeseries*sizeof(unsigned long long int);
		cudaMalloc((void **) &d_partial_sums,  partial_sum_size);
		cudaMemset(d_partial_sums, 0, partial_sum_size);
	}
		
	// ----------------------------------------------->
	// --------> Measured part GPU scan
	//---------> GPU scan
	if(d_partial_sums!=NULL) {
		GPU_scan_warp<Scan_inclusive><<<gridSize,blockSize>>>(d_output, d_input, nElements, d_partial_sums);
		GPU_scan_grid_followup<Scan_inclusive><<<gridSize,blockSize>>>(d_output, d_input, d_partial_sums, nElements);
	}
	else {
		GPU_scan_warp<Scan_inclusive><<<gridSize,blockSize>>>(d_output, d_input, nElements, d_partial_sums);
	}
	// --------> Measured part GPU scan
	// ----------------------------------------------->
	
	if(d_partial_sums!=NULL) cudaFree(d_partial_sums);
	return(0);
}


void RQA_length_histogram_from_timeseries_direct(unsigned long long int *h_length_histogram, int *d_input, unsigned int nSamples){
	int nThreads = 1024;
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Data dimensions: %d;\n", nSamples);
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	//---------------> GPU Allocations
	unsigned long long int *d_histogram;
	size_t histogram_size_in_bytes = (nSamples + 1)*sizeof(unsigned long long int);
	
	cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	//--------------------------------<
	
	//------------> GPU kernel
	GPU_RQA_HST_length_histogram_direct<RQA_ConstParams><<< gridSize , blockSize >>>(d_histogram, d_input, nSamples);
	//--------------------------------<
	
	//------------> Copy data to host
	cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	//--------------------------------<
	
	//----------> Deallocations
	cudaFree(d_histogram);
	//--------------------------------<
}


template<class const_params, typename IOtype>
void RQA_length_histogram_horizontal_wrapper(
	unsigned long long int *h_length_histogram, 
	unsigned long long int *h_scan_histogram, 
	unsigned long long int *h_metric, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	long int input_size,
	int distance_type, 
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	
	//-----------> Kernel configurations
	// CUDA grid and block size for length histogram calculation
	dim3 gridSize(corrected_size, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	// CUDA grid and block size for reverse_array
	dim3 ra_gridSize( (hst_size + nThreads - 1)/nThreads, 1, 1);
	dim3 ra_blockSize(nThreads, 1, 1);
	
	//---------> Checking memory
	
	//---------> GPU Memory allocation
	size_t histogram_size_in_bytes = hst_size*sizeof(unsigned long long int);
	size_t input_size_bytes = input_size*sizeof(IOtype);
	
	IOtype *d_input;
	unsigned long long int *d_histogram;
	unsigned long long int *d_temporary;
	unsigned long long int *d_metric;
	
	cudaError = cudaMalloc((void **) &d_input, input_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_metric = NULL;
	}
	//--------------------------------<
	
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//------------> GPU kernel
	switch(distance_type) {
		case DST_EUCLIDEAN:
			GPU_RQA_HST_length_histogram_horizontal<RQA_euc><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
			break;
		case DST_MAXIMAL:
			GPU_RQA_HST_length_histogram_horizontal<RQA_max><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
			break;
		default :
			*error = ERR_INVALID_ARGUMENT;
			break;
	}
	
	// This part of the code is commented out because h_metric and h_scan_histogram are calculated on the host side
	//------------> Calculation metric
	//reverse_array_and_multiply<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	//GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	//cudaError = cudaMemcpy(h_metric, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	//if(cudaError != cudaSuccess) {
	//	*error = ERR_CUDA;
	//}
	//--------------------------------<
	
	//------------> Calculation scan histogram
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	//GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	//cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	//if(cudaError != cudaSuccess) {
	//	*error = ERR_CUDA;
	//}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_histogram!=NULL) cudaFree(d_histogram);
	if(d_temporary!=NULL) cudaFree(d_temporary);
	if(d_metric!=NULL) cudaFree(d_metric);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
}


template<class const_params, typename IOtype>
void RQA_length_histogram_vertical_wrapper(
	unsigned long long int *h_length_histogram, 
	unsigned long long int *h_scan_histogram, 
	unsigned long long int *h_metric, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	unsigned long long int input_size,
	int distance_type, 
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	
	//-----------> Kernel configurations
	// CUDA grid and block size for length histogram calculation
	dim3 gridSize(corrected_size, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	// CUDA grid and block size for reverse_array
	dim3 ra_gridSize( (hst_size + nThreads - 1)/nThreads, 1, 1);
	dim3 ra_blockSize(nThreads, 1, 1);
	
	//---------> GPU Memory allocation
	size_t histogram_size_in_bytes = hst_size*sizeof(unsigned long long int);
	size_t input_size_bytes = input_size*sizeof(IOtype);
	
	IOtype *d_input;
	unsigned long long int *d_histogram;
	unsigned long long int *d_temporary;
	unsigned long long int *d_metric;
	
	cudaError = cudaMalloc((void **) &d_input, input_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_metric = NULL;
	}
	//--------------------------------<
	
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//------------> GPU kernel
	switch(distance_type) {
		case DST_EUCLIDEAN:
			GPU_RQA_HST_length_histogram_vertical<RQA_euc><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
			break;
		case DST_MAXIMAL:
			GPU_RQA_HST_length_histogram_vertical<RQA_max><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
			break;
		default :
			*error = ERR_INVALID_ARGUMENT;
			break;
	}
	
	// This part of the code is commented out because h_metric and h_scan_histogram are calculated on the host side
	//------------> Calculation metric
	//reverse_array_and_multiply<<< ra_gridSize , ra_blockSize , 0 , NULL >>>(d_temporary, d_histogram, hst_size);
	//GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	//cudaDeviceSynchronize();
	//cudaError = cudaMemcpy(h_metric, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	//if(cudaError != cudaSuccess) {
	//	*error = ERR_CUDA;
	//}
	//--------------------------------<

	//------------> Calculation scan histogram
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	//GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	//cudaDeviceSynchronize();
	//cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	//if(cudaError != cudaSuccess) {
	//	*error = ERR_CUDA;
	//}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_histogram!=NULL) cudaFree(d_histogram);
	if(d_temporary!=NULL) cudaFree(d_temporary);
	if(d_metric!=NULL) cudaFree(d_metric);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
}


template<class const_params, typename IOtype>
void RQA_length_histogram_diagonal_wrapper_mk1(
	unsigned long long int *h_length_histogram, 
	unsigned long long int *h_scan_histogram, 
	unsigned long long int *h_metric, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	long int input_size,
	double *execution_time,
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	GPU_Timer timer;
	
	//-----------> Kernel configurations
	// CUDA grid and block size for length histogram calculation
	dim3 gridSize(corrected_size + corrected_size - 1, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	// CUDA grid and block size for reverse_array
	dim3 ra_gridSize( (hst_size + nThreads - 1)/nThreads, 1, 1);
	dim3 ra_blockSize(nThreads, 1, 1);
	
	//---------> Checking memory
	
	//---------> GPU Memory allocation
	size_t histogram_size_in_bytes = hst_size*sizeof(unsigned long long int);
	size_t input_size_bytes = input_size*sizeof(IOtype);
	
	IOtype *d_input;
	unsigned long long int *d_histogram;
	unsigned long long int *d_temporary;
	unsigned long long int *d_metric;
	
	cudaError = cudaMalloc((void **) &d_input, input_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_metric = NULL;
	}
	
	timer.Start();
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//------------> GPU kernel
	GPU_RQA_HST_length_histogram_diagonal<const_params><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
	
	//------------> Calculation metric
	reverse_array_and_multiply<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaDeviceSynchronize();
	cudaError = cudaMemcpy(h_metric, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//------------> Calculation scan histogram
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaDeviceSynchronize();
	cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	//--------------------------------<
	timer.Stop();
	*execution_time = timer.Elapsed();
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_histogram!=NULL) cudaFree(d_histogram);
	if(d_temporary!=NULL) cudaFree(d_temporary);
	if(d_metric!=NULL) cudaFree(d_metric);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
}


template<class const_params, typename IOtype>
void RQA_length_histogram_diagonal_wrapper_mk2(
	unsigned long long int *h_length_histogram, 
	unsigned long long int *h_scan_histogram, 
	unsigned long long int *h_metric, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	long int input_size,
	int distance_type, 
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	
	//-----------> Kernel configurations
	// CUDA grid and block size for length histogram calculation
	dim3 gridSize(corrected_size - 1, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	// CUDA grid and block size for reverse_array
	dim3 ra_gridSize( (hst_size + nThreads - 1)/nThreads, 1, 1);
	dim3 ra_blockSize(nThreads, 1, 1);
	
	//---------> Checking memory
	
	
	//---------> GPU Memory allocation
	size_t histogram_size_in_bytes = hst_size*sizeof(unsigned long long int);
	size_t input_size_bytes = input_size*sizeof(IOtype);
	
	IOtype *d_input;
	unsigned long long int *d_histogram;
	unsigned long long int *d_temporary;
	unsigned long long int *d_metric;
	
	cudaError = cudaMalloc((void **) &d_input, input_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_metric = NULL;
	}
	
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//------------> GPU kernel
	switch(distance_type) {
		case DST_EUCLIDEAN:
			GPU_RQA_HST_length_histogram_diagonal<RQA_euc><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
			break;
		case DST_MAXIMAL:
			GPU_RQA_HST_length_histogram_diagonal<RQA_max><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
			break;
		default :
			*error = ERR_INVALID_ARGUMENT;
			break;
	}
	
	GPU_RQA_HST_correction_diagonal_half<<< ra_gridSize , ra_blockSize >>>( d_histogram, hst_size );
	cudaDeviceSynchronize();
	
	// This part of the code is commented out because h_metric and h_scan_histogram are calculated on the host side
	//------------> Calculation metric
	//reverse_array_and_multiply<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	//GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	//cudaError = cudaMemcpy(h_metric, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	//if(cudaError != cudaSuccess) {
	//	*error = ERR_CUDA;
	//}
	//--------------------------------<
	
	//------------> Calculation scan histogram
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	//GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	//reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	//cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	//if(cudaError != cudaSuccess) {
	//	*error = ERR_CUDA;
	//}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_histogram!=NULL) cudaFree(d_histogram);
	if(d_temporary!=NULL) cudaFree(d_temporary);
	if(d_metric!=NULL) cudaFree(d_metric);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
	
	cudaDeviceSynchronize();
}


// This version will use atomic operations to add contributions from 
//  individual thread-blocks. Further improvement might be to create
//  an array and reduce later avoiding atomic operations all together
template<class const_params, typename IOtype>
void RQA_length_histogram_diagonal_sum_wrapper(
	IOtype *h_DET, 
	IOtype *h_L, 
	unsigned long long int *h_Lmax, 
	IOtype *h_RR, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	int lmin, 
	size_t input_size,
	int distance_type, 
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Checking that device is present
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	int nThreads = 1024;
	
	//-----------> Kernel configurations
	// CUDA grid and block size for length histogram calculation
	// -1 because diagonal is not used and corrected for later
	dim3 gridSize(corrected_size - 1, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	//---------> GPU Memory allocation
	size_t input_size_bytes = input_size*sizeof(IOtype);
	
	IOtype *d_input;
	unsigned long long int *d_S_all;
	unsigned long long int *d_S_lmin;
	unsigned long long int *d_lmax;
	unsigned long long int *d_N_lmin;
	
	cudaError = cudaMalloc((void **) &d_input, input_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_S_all,  sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_all = NULL;
	}
	cudaError = cudaMalloc((void **) &d_S_lmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_lmin = NULL;
	}
	cudaError = cudaMalloc((void **) &d_lmax,   sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_lmax = NULL;
	}
	cudaError = cudaMalloc((void **) &d_N_lmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_N_lmin = NULL;
	}
	
	//---------> Memory copy and preparation
	cudaMemset(d_S_all,  0, sizeof(unsigned long long int));
	cudaMemset(d_S_lmin, 0, sizeof(unsigned long long int));
	cudaMemset(d_lmax,   0, sizeof(unsigned long long int));
	cudaMemset(d_N_lmin, 0, sizeof(unsigned long long int));
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//------------> GPU kernel
	switch(distance_type) {
		case DST_EUCLIDEAN:
			GPU_RQA_HST_length_histogram_diagonal_sum<RQA_euc><<< gridSize , blockSize >>>(d_S_all, d_S_lmin, d_lmax, d_N_lmin, d_input, threshold, tau, emb, lmin, corrected_size);
			break;
		case DST_MAXIMAL:
			GPU_RQA_HST_length_histogram_diagonal_sum<RQA_max><<< gridSize , blockSize >>>(d_S_all, d_S_lmin, d_lmax, d_N_lmin, d_input, threshold, tau, emb, lmin, corrected_size);
			break;
		default :
			*error = ERR_INVALID_ARGUMENT;
			break;
	}
	
	cudaDeviceSynchronize();
	//--------------------------------<
	
	//------------> Calculation RQA measures
	unsigned long long int h_S_all = 0;
	unsigned long long int h_S_lmin = 0;
	unsigned long long int h_lmax = 0;
	unsigned long long int h_N_lmin = 0;
	
	cudaError = cudaMemcpy(&h_S_all, d_S_all, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_S_lmin, d_S_lmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_lmax, d_lmax, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_N_lmin, d_N_lmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	h_S_all  = 2*h_S_all + corrected_size;
	h_S_lmin = 2*h_S_lmin + corrected_size;
	h_N_lmin = 2*h_N_lmin + 1;
	
	
	*h_DET  = ((double) h_S_lmin)/((double) h_S_all);
	*h_L    = ((double) h_S_lmin)/((double) h_N_lmin);
	*h_Lmax = h_lmax;
	*h_RR   = ((double) h_S_all)/((double) (corrected_size*corrected_size));
	//--------------------------------<
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_S_all!=NULL) cudaFree(d_S_all);
	if(d_S_lmin!=NULL) cudaFree(d_S_lmin);
	if(d_lmax!=NULL) cudaFree(d_lmax);
	if(d_N_lmin!=NULL) cudaFree(d_N_lmin);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
}


// This version is introduces boxcar filtering and which disregards creating 
//  length histogram and applies boxcar filter to eliminate lines of 
//  length < lmin and then correcting for it. It also introduces load balancing
//  coordinate calculation.
template<class const_params, typename IOtype>
void RQA_diagonal_boxcar_wrapper(
	IOtype *h_DET, 
	IOtype *h_L, 
	unsigned long long int *h_Lmax, 
	IOtype *h_RR, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	int lmin, 
	size_t input_size,
	int distance_type, 
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Checking that device is present
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	//----------> Allocation of input
	size_t allocation_size = input_size + 1;
	size_t corrected_size = allocation_size - (emb - 1)*tau;
	IOtype *d_input;
	cudaError = cudaMalloc((void **) &d_input, allocation_size*sizeof(IOtype));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMemcpy(d_input, h_input, input_size*sizeof(IOtype), cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	IOtype extreme_value = std::numeric_limits<IOtype>::max();
	cudaError = cudaMemcpy(&d_input[input_size], &extreme_value, sizeof(IOtype), cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	
	//-----------> Kernel configurations
	// CUDA grid and block size for length histogram calculation
	// -1 because diagonal is not used and corrected for later
	
	//---------> GPU Memory allocation
	unsigned long long int *d_S_all;
	unsigned long long int *d_S_lmin;
	unsigned long long int *d_N_lmin;
	cudaError = cudaMalloc((void **) &d_S_all,  sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_all = NULL;
	}
	cudaError = cudaMalloc((void **) &d_S_lmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_lmin = NULL;
	}
	cudaError = cudaMalloc((void **) &d_N_lmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_N_lmin = NULL;
	}
	
	
	//---------> Memory copy and preparation
	cudaMemset(d_S_all,  0, sizeof(unsigned long long int));
	cudaMemset(d_S_lmin, 0, sizeof(unsigned long long int));
	cudaMemset(d_N_lmin, 0, sizeof(unsigned long long int));

	
	//------------> GPU kernel
	int nThreads = 1024;
	// lmin because it is  - (lmin - 1 + 1) + 1 is for the leading R value
	size_t elements_per_block = DET_FAST_MUL*nThreads - lmin;
	size_t total_elements = ((corrected_size-1)*corrected_size)/2;
	size_t nBlocks = (total_elements + elements_per_block - 1)/elements_per_block;
	dim3 gridSize(nBlocks, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	switch(distance_type) {
		case DST_EUCLIDEAN:
			GPU_RQA_DET_boxcar
				<RQA_euc, IOtype>
				<<< gridSize , blockSize , DET_FAST_MUL*nThreads*sizeof(int)  >>>(
					d_S_all, d_S_lmin, d_N_lmin, 
					d_input, 
					threshold, tau, emb, lmin, corrected_size
				);
			break;
		case DST_MAXIMAL:
			GPU_RQA_DET_boxcar
				<RQA_max, IOtype>
				<<< gridSize , blockSize , DET_FAST_MUL*nThreads*sizeof(int)  >>>(
					d_S_all, d_S_lmin, d_N_lmin, 
					d_input, 
					threshold, tau, emb, lmin, corrected_size
				);
			break;
		default :
			*error = ERR_INVALID_ARGUMENT;
			break;
	}
	
	cudaDeviceSynchronize();
	//--------------------------------<
	
	//------------> Calculation RQA measures
	unsigned long long int h_S_all = 0;
	unsigned long long int h_S_lmin = 0;
	unsigned long long int h_N_lmin = 0;
	
	cudaError = cudaMemcpy(&h_S_all, d_S_all, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_S_lmin, d_S_lmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_N_lmin, d_N_lmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//ACCRQA_PRINT("\n");
	//ACCRQA_PRINT("Before correction: h_S_lmin=%llu; h_S_all=%llu; h_N_lmin=%llu\n", h_S_lmin, h_S_all, h_N_lmin);
	// This is because at the end of the array we have added one element
	corrected_size = corrected_size - 1;
	// Correction because we excluding central diagonal line and lower triangle
	h_S_all  = 2*h_S_all + corrected_size;
	h_S_lmin = 2*h_S_lmin + corrected_size;
	h_N_lmin = 2*h_N_lmin + 1;
	//ACCRQA_PRINT("After correction: h_S_lmin=%llu; h_S_all=%llu; h_N_lmin=%llu\n", h_S_lmin, h_S_all, h_N_lmin);
	
	*h_DET  = ((double) h_S_lmin)/((double) h_S_all);
	*h_L    = ((double) h_S_lmin)/((double) h_N_lmin);
	*h_Lmax = 0;
	*h_RR   = ((double) h_S_all)/((double) ((corrected_size-1)*(corrected_size-1)));
	//--------------------------------<
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_S_all!=NULL) cudaFree(d_S_all);
	if(d_S_lmin!=NULL) cudaFree(d_S_lmin);
	if(d_N_lmin!=NULL) cudaFree(d_N_lmin);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
}



// This version is using the boxcar filtering and related correction but compare 
//  to the boxcar version, this version is using different split of the R matrix
//  that allow simultaneous calculation of DET and LAM.
template<class const_params, typename IOtype>
void RQA_diagonal_boxcar_square_wrapper(
	IOtype *h_DET, 
	IOtype *h_L, 
	unsigned long long int *h_Lmax, 
	IOtype *h_RR, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	int lmin, 
	size_t input_size,
	int input_type,
	int distance_type, 
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Checking that device is present
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	//----------> Allocation of input
	size_t allocation_size = input_size + 1;
	size_t corrected_size = allocation_size - (emb - 1)*tau;
	IOtype *d_input;
	cudaError = cudaMalloc((void **) &d_input, allocation_size*sizeof(IOtype));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMemcpy(d_input + 1, h_input, input_size*sizeof(IOtype), cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	IOtype extreme_value = std::numeric_limits<IOtype>::max();
	cudaError = cudaMemcpy(&d_input[0], &extreme_value, sizeof(IOtype), cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//---------> GPU Memory allocation
	unsigned long long int *d_S_all;
	unsigned long long int *d_S_lmin;
	unsigned long long int *d_N_lmin;
	cudaError = cudaMalloc((void **) &d_S_all,  sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_all = NULL;
	}
	cudaError = cudaMalloc((void **) &d_S_lmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_lmin = NULL;
	}
	cudaError = cudaMalloc((void **) &d_N_lmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_N_lmin = NULL;
	}
	
	
	//---------> Memory copy and preparation
	cudaMemset(d_S_all,  0, sizeof(unsigned long long int));
	cudaMemset(d_S_lmin, 0, sizeof(unsigned long long int));
	cudaMemset(d_N_lmin, 0, sizeof(unsigned long long int));

	
	//------------> GPU kernel
	int nThreads = 1024;
	int width = 32;
	int usefull_part = width - lmin;
	// lmin because it is  - (lmin - 1 + 1) + 1 is for the leading R value
	size_t nBlocks_x = (corrected_size + usefull_part - 1)/usefull_part;
	size_t nBlocks_y = (corrected_size + usefull_part - 1)/usefull_part;

	dim3 gridSize(nBlocks_x, nBlocks_y, 1);
	dim3 blockSize(nThreads, 1, 1);
	int cache_size = 32 + (emb - 1)*tau;
	if(input_type==0 && *error == SUCCESS){
		switch(distance_type) {
			case DST_EUCLIDEAN:
				GPU_RQA_DET_boxcar_square_float
					<RQA_euc>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_lmin, d_N_lmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			case DST_MAXIMAL:
				GPU_RQA_DET_boxcar_square_float
					<RQA_max>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_lmin, d_N_lmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			default :
				*error = ERR_INVALID_ARGUMENT;
				break;
		}
	}
	else if(input_type==1 && *error == SUCCESS){
		switch(distance_type) {
			case DST_EUCLIDEAN:
				GPU_RQA_DET_boxcar_square_double
					<RQA_euc>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_lmin, d_N_lmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			case DST_MAXIMAL:
				GPU_RQA_DET_boxcar_square_double
					<RQA_max>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_lmin, d_N_lmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			default :
				*error = ERR_INVALID_ARGUMENT;
				break;
		}
	}
	else {
		*error = ERR_INVALID_ARGUMENT;
	}
	cudaDeviceSynchronize();
	//--------------------------------<
	
	//------------> Calculation RQA measures
	unsigned long long int h_S_all = 0;
	unsigned long long int h_S_lmin = 0;
	unsigned long long int h_N_lmin = 0;
	
	cudaError = cudaMemcpy(&h_S_all, d_S_all, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_S_lmin, d_S_lmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_N_lmin, d_N_lmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	// This is because at the end of the array we have added one element
	corrected_size = corrected_size - 1;
	// As we process only the upper triangle we must correct for whole RP
	h_S_all  = 2*h_S_all + corrected_size;
	h_S_lmin = 2*h_S_lmin + corrected_size;
	h_N_lmin = 2*h_N_lmin + 1;
	
	*h_DET  = ((double) h_S_lmin)/((double) h_S_all);
	*h_L    = ((double) h_S_lmin)/((double) h_N_lmin);
	*h_Lmax = 0;
	*h_RR   = ((double) h_S_all)/((double) ((corrected_size-1)*(corrected_size-1)));
	//--------------------------------<
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_S_all!=NULL) cudaFree(d_S_all);
	if(d_S_lmin!=NULL) cudaFree(d_S_lmin);
	if(d_N_lmin!=NULL) cudaFree(d_N_lmin);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
}





// This version is using the boxcar filtering and related correction but compare 
//  to the boxcar version, this version is using different split of the R matrix
//  that allow simultaneous calculation of DET and LAM.
template<class const_params, typename IOtype>
void RQA_horizontal_boxcar_square_wrapper(
	IOtype *h_LAM, 
	IOtype *h_TT, 
	unsigned long long int *h_TTmax, 
	IOtype *h_RR, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	int lmin, 
	size_t input_size,
	int input_type,
	int distance_type, 
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Checking that device is present
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	//----------> Allocation of input
	size_t allocation_size = input_size + 1;
	size_t corrected_size = allocation_size - (emb - 1)*tau;
	IOtype *d_input;
	cudaError = cudaMalloc((void **) &d_input, allocation_size*sizeof(IOtype));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMemcpy(d_input + 1, h_input, input_size*sizeof(IOtype), cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	IOtype extreme_value = std::numeric_limits<IOtype>::max();
	cudaError = cudaMemcpy(&d_input[0], &extreme_value, sizeof(IOtype), cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//-----------> Kernel configurations
	// CUDA grid and block size for length histogram calculation
	// -1 because diagonal is not used and corrected for later
	
	//---------> GPU Memory allocation
	unsigned long long int *d_S_all;
	unsigned long long int *d_S_vmin;
	unsigned long long int *d_N_vmin;
	cudaError = cudaMalloc((void **) &d_S_all,  sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_all = NULL;
	}
	cudaError = cudaMalloc((void **) &d_S_vmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_S_vmin = NULL;
	}
	cudaError = cudaMalloc((void **) &d_N_vmin, sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_N_vmin = NULL;
	}
	
	//---------> Memory copy and preparation
	cudaMemset(d_S_all,  0, sizeof(unsigned long long int));
	cudaMemset(d_S_vmin, 0, sizeof(unsigned long long int));
	cudaMemset(d_N_vmin, 0, sizeof(unsigned long long int));

	
	//------------> GPU kernel
	int nThreads = 1024;
	int width = 32;
	int usefull_part = width - lmin;
	// lmin because it is  - (lmin - 1 + 1) + 1 is for the leading R value
	size_t nBlocks_x = (corrected_size + usefull_part - 1)/usefull_part;
	size_t nBlocks_y = (corrected_size + usefull_part - 1)/usefull_part;

	dim3 gridSize(nBlocks_x, nBlocks_y, 1);
	dim3 blockSize(nThreads, 1, 1);
	int cache_size = 32 + (emb - 1)*tau;
	if(input_type==0 && *error == SUCCESS){
		switch(distance_type) {
			case DST_EUCLIDEAN:
				GPU_RQA_LAM_boxcar_square_float
					<RQA_euc>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_vmin, d_N_vmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			case DST_MAXIMAL:
				GPU_RQA_LAM_boxcar_square_float
					<RQA_max>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_vmin, d_N_vmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			default :
				*error = ERR_INVALID_ARGUMENT;
				break;
		}
	}
	else if(input_type==1 && *error == SUCCESS){
		switch(distance_type) {
			case DST_EUCLIDEAN:
				GPU_RQA_LAM_boxcar_square_double
					<RQA_euc>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_vmin, d_N_vmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			case DST_MAXIMAL:
				GPU_RQA_LAM_boxcar_square_double
					<RQA_max>
					<<< gridSize , blockSize , 2*cache_size*sizeof(IOtype) >>>(
						d_S_all, d_S_vmin, d_N_vmin, 
						d_input, 
						threshold, tau, emb, lmin, corrected_size
					);
				break;
			default :
				*error = ERR_INVALID_ARGUMENT;
				break;
		}
	}
	else {
		*error = ERR_INVALID_ARGUMENT;
	}
	cudaDeviceSynchronize();
	//--------------------------------<
	
	//------------> Calculation RQA measures
	unsigned long long int h_S_all = 0;
	unsigned long long int h_S_vmin = 0;
	unsigned long long int h_N_vmin = 0;
	
	cudaError = cudaMemcpy(&h_S_all, d_S_all, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_S_vmin, d_S_vmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(&h_N_vmin, d_N_vmin, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	// This is because at the end of the array we have added one element
	corrected_size = corrected_size - 1;
	h_S_all  = h_S_all;
	h_S_vmin = h_S_vmin;
	h_N_vmin = h_N_vmin;
	
	*h_LAM   = ((double) h_S_vmin)/((double) h_S_all);
	*h_TT    = ((double) h_S_vmin)/((double) h_N_vmin);
	*h_TTmax = 0;
	*h_RR    = ((double) h_S_all)/((double) ((corrected_size-1)*(corrected_size-1)));
	//--------------------------------<
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_S_all!=NULL) cudaFree(d_S_all);
	if(d_S_vmin!=NULL) cudaFree(d_S_vmin);
	if(d_N_vmin!=NULL) cudaFree(d_N_vmin);
	if(d_input!=NULL) cudaFree(d_input);
	//--------------------------------<
}



void test_array_reversal(){
	int *input, *output;
	size_t input_size = 10000;
	cudaMallocManaged(&input, input_size*sizeof(int));
	cudaMallocManaged(&output, input_size*sizeof(int));
	for(size_t f=0; f<input_size; f++){
		input[f] = f;
		output[f] = 0;
	}
	
	int nThreads = 1024;
	dim3 ra_gridSize( (input_size + nThreads - 1)/nThreads, 1, 1);
	dim3 ra_blockSize(nThreads, 1, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(output, input, input_size);
	cudaDeviceSynchronize();
	
	for(size_t f=0; f<input_size; f++){
		ACCRQA_PRINT("[%d; %d]", input[f], output[f]);
	}
	
	cudaFree(input);
	cudaFree(output);
}

template<typename IOtype>
void GPU_RQA_generate_diagonal_R_matrix_wrapper(
	int *h_diagonal_R_matrix, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	size_t input_size, 
	int distance_type, 
	int device, 
	double *execution_time,
	Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(device<devCount) cudaSetDevice(device);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	//---------> Measurements
	*execution_time = 0;
	GPU_Timer timer;
	
	//---------> GPU Memory allocation
	size_t corrected_size = input_size - (emb - 1)*tau;
	size_t matrix_size = corrected_size*(2*corrected_size-1)*sizeof(int);
	size_t input_size_bytes = input_size*sizeof(IOtype);
	IOtype *d_input;
	int *d_diagonal_R_matrix;
	cudaMalloc((void **) &d_input, input_size_bytes);
	cudaMalloc((void **) &d_diagonal_R_matrix, matrix_size);

	//---------> Memory copy
	cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	
	//------------- GPU kernel
	timer.Start();
	
	int nThreads = 1024;
	dim3 gridSize(2*corrected_size - 1, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Input size: %ld; Time step: %d; Embedding: %d; Corrected size: %ld;\n", input_size, tau, emb, corrected_size);
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG_GPU_HST) ACCRQA_PRINT("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	//------------> GPU kernel
	switch(distance_type) {
		case DST_EUCLIDEAN:
			GPU_RQA_generate_diagonal_R_matrix_kernel<RQA_euc><<< gridSize , blockSize >>>(d_diagonal_R_matrix, d_input, threshold, tau, emb, corrected_size);
			break;
		case DST_MAXIMAL:
			GPU_RQA_generate_diagonal_R_matrix_kernel<RQA_max><<< gridSize , blockSize >>>(d_diagonal_R_matrix, d_input, threshold, tau, emb, corrected_size);
			break;
		default :
			*error = ERR_INVALID_ARGUMENT;
			break;
	}
	//--------------------------------<
	
	timer.Stop();
	*execution_time = timer.Elapsed();
	//------------- GPU kernel
	
	cudaMemcpy(h_diagonal_R_matrix, d_diagonal_R_matrix, matrix_size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_input);
	cudaFree(d_diagonal_R_matrix);
}


int GPU_RQA_length_start_end_test(unsigned long long int *h_length_histogram, int *h_input, long int nSamples, int device, int nRuns, double *execution_time){
	//---------> Initial nVidia stuff
	int devCount = 0;
	cudaGetDeviceCount(&devCount);
	if(device<devCount) cudaSetDevice(device);
	else { ACCRQA_PRINT("Wrong device!\n"); return(1); }
	
	//---------> Measurements
	*execution_time = 0;
	GPU_Timer timer;
	
	//---------> GPU Memory allocation
	size_t input_size = nSamples*sizeof(int);
	int *d_input;
	cudaMalloc((void **) &d_input, input_size);

	//---------> Memory allocation
	
	
	cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
	
	//------------- GPU kernel
	timer.Start();
	//RQA_length_histogram_from_timeseries(h_length_histogram, d_input, nSamples);
	RQA_length_histogram_from_timeseries_direct(h_length_histogram, d_input, nSamples);
	timer.Stop();
	*execution_time = timer.Elapsed();
	//------------- GPU kernel
	
	cudaFree(d_input);
	
	return(0);
}


//---------------------------- L1 WRAPERS -----------------------------

void GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_horizontal_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, distance_type, error);
}

void GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_horizontal_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, distance_type, error);
}


void GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_vertical_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, distance_type, error);
}

void GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_vertical_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, distance_type, error);
}


void GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_diagonal_wrapper_mk2<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, distance_type, error);
}

void GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_diagonal_wrapper_mk2<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, distance_type, error);
}


void GPU_RQA_length_histogram_diagonal_sum(double *h_DET, double *h_L, unsigned long long int *h_Lmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_diagonal_sum_wrapper<RQA_ConstParams, double>(h_DET, h_L, h_Lmax, h_RR, h_input, threshold, tau, emb, lmin, input_size, distance_type, error);
}

void GPU_RQA_length_histogram_diagonal_sum(float *h_DET, float *h_L, unsigned long long int *h_Lmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error){
	RQA_length_histogram_diagonal_sum_wrapper<RQA_ConstParams>(h_DET, h_L, h_Lmax, h_RR, h_input, threshold, tau, emb, lmin, input_size, distance_type, error);
}


void GPU_RQA_diagonal_boxcar(double *h_DET, double *h_L, unsigned long long int *h_Lmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error){
	RQA_diagonal_boxcar_wrapper<RQA_ConstParams, double>(h_DET, h_L, h_Lmax, h_RR, h_input, threshold, tau, emb, lmin, input_size, distance_type, error);
}

void GPU_RQA_diagonal_boxcar(float *h_DET, float *h_L, unsigned long long int *h_Lmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error){
	RQA_diagonal_boxcar_wrapper<RQA_ConstParams, float>(h_DET, h_L, h_Lmax, h_RR, h_input, threshold, tau, emb, lmin, input_size, distance_type, error);
}


void GPU_RQA_diagonal_boxcar_square(double *h_DET, double *h_L, unsigned long long int *h_Lmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error){
	// 1 means double. This is because CUDA does not seem to be capable of overloading dynamically allocated shared memory.
	RQA_diagonal_boxcar_square_wrapper<RQA_ConstParams, double>(h_DET, h_L, h_Lmax, h_RR, h_input, threshold, tau, emb, lmin, input_size, 1, distance_type, error);
}

void GPU_RQA_diagonal_boxcar_square(float *h_DET, float *h_L, unsigned long long int *h_Lmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int lmin, size_t input_size, int distance_type, Accrqa_Error *error){
	// 0 means float. This is because CUDA does not seem to be capable of overloading dynamically allocated shared memory.
	RQA_diagonal_boxcar_square_wrapper<RQA_ConstParams, float>(h_DET, h_L, h_Lmax, h_RR, h_input, threshold, tau, emb, lmin, input_size, 0, distance_type, error);
}

void GPU_RQA_horizontal_boxcar_square(double *h_LAM, double *h_TT, unsigned long long int *h_TTmax, double *h_RR, double *h_input, double threshold, int tau, int emb, int vmin, size_t input_size, int distance_type, Accrqa_Error *error){
	// 1 means double. This is because CUDA does not seem to be capable of overloading dynamically allocated shared memory.
	RQA_horizontal_boxcar_square_wrapper<RQA_ConstParams, double>(h_LAM, h_TT, h_TTmax, h_RR, h_input, threshold, tau, emb, vmin, input_size, 1, distance_type, error);
}

void GPU_RQA_horizontal_boxcar_square(float *h_LAM, float *h_TT, unsigned long long int *h_TTmax, float *h_RR, float *h_input, float threshold, int tau, int emb, int vmin, size_t input_size, int distance_type, Accrqa_Error *error){
	// 0 means float. This is because CUDA does not seem to be capable of overloading dynamically allocated shared memory.
	RQA_horizontal_boxcar_square_wrapper<RQA_ConstParams, float>(h_LAM, h_TT, h_TTmax, h_RR, h_input, threshold, tau, emb, vmin, input_size, 0, distance_type, error);
}


void GPU_RQA_generate_diagonal_R_matrix(int *h_diagonal_R_matrix, float *h_input, float threshold, int tau, int emb, size_t input_size, int distance_type, int device, double *execution_time, Accrqa_Error *error
){
	GPU_RQA_generate_diagonal_R_matrix_wrapper(h_diagonal_R_matrix, h_input, threshold, tau, emb, input_size, distance_type, device, execution_time, error);
}

void GPU_RQA_generate_diagonal_R_matrix(int *h_diagonal_R_matrix, double *h_input, double threshold, int tau, int emb, size_t input_size, int distance_type, int device, double *execution_time, Accrqa_Error *error
){
	GPU_RQA_generate_diagonal_R_matrix_wrapper(h_diagonal_R_matrix, h_input, threshold, tau, emb, input_size, distance_type, device, execution_time, error);
}



