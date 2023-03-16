#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip> 
#include <vector>

#include "../include/AccRQA_definitions.hpp"
#include "GPU_scan.cuh"
#include "GPU_reduction.cuh"
#include "AccRQA_metrics.cuh"

using namespace std;


class RQA_ConstParams {
public:
	static const int nRows_per_thread = 1;
	static const int warp = 32;
	static const int shared_memory_size = 512;
};

#define NTHREADS 256
#define LAM_NTHREADS 1024
#define Y_STEPS 4
#define X_STEPS 4
#define HALF_WARP 16
#define NSEEDS 32
#define WARP 32
#define BUFFER 32

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

template<typename IOtype>
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
		(*first) = R_element_max(d_input, blockIdx.x, gl_index, threshold, tau, emb, corrected_size);
	}
	else {
		(*first)= 0;
	}
	
	// -------------------- second ---------------------
	if( (gl_index + 1) >= 0 && (gl_index + 1) < corrected_size ) {
		//(*second) = R_element_cartesian(seed, d_input[gl_index + 1], threshold);
		(*second) = R_element_max(d_input, blockIdx.x, gl_index + 1, threshold, tau, emb, corrected_size);
	}	
	else {
		(*second) = 0;
	}
}

template<typename IOtype>
__inline__ __device__ void GPU_RQA_HST_get_R_matrix_element_vertical(int *first, int *second, IOtype const* __restrict__ d_input, IOtype threshold, int tau, int emb, int block, long int corrected_size){
	long int gl_index = (long int)(block*blockDim.x + threadIdx.x) - 1;
	//float seed = d_input[blockIdx.y];
	
	// loading data like this avoids border case at the beginning
	// -------------------- first ---------------------	
	if( gl_index >= 0 && gl_index < corrected_size ) {
		//(*first) = R_element_cartesian(d_input[gl_index], seed, threshold);
		(*first) = R_element_max(d_input, gl_index, blockIdx.x, threshold, tau, emb, corrected_size);
	}
	else {
		(*first)= 0;
	}
	
	// -------------------- second ---------------------
	if( (gl_index + 1) >= 0 && (gl_index + 1) < corrected_size ) {
		//(*second) = R_element_cartesian(d_input[gl_index + 1], seed, threshold);
		(*second) = R_element_max(d_input, gl_index + 1, blockIdx.x, threshold, tau, emb, corrected_size);
	}	
	else {
		(*second) = 0;
	}
}

template<typename IOtype>
__inline__ __device__ int GPU_RQA_HST_get_R_matrix_element_diagonal(IOtype const* __restrict__ d_input, IOtype threshold, int tau, int emb, int block, long int gl_index, long int corrected_size){
	long int block_y  = (long int) (blockIdx.x) - corrected_size + 1;
	
	// This is stored first as blockIdx.x < size
	// it is stored as blockIdx.x = 0 => row 0 
	if(block_y<0){
		long int row = gl_index;
		long int column = gl_index - block_y; //block_y is negative thus it is like + block_y
		if(row >= 0 && row < corrected_size && column  >= 0 && column < corrected_size) {
			int value = R_element_max(d_input, row, column, threshold, tau, emb, corrected_size);
			return(value);
		}
		else return(0);
	}
	
	if(block_y>0){
		long int row = gl_index + block_y;
		long int column = gl_index;
		if(row >= 0 && row < corrected_size && column  >= 0 && column < corrected_size) {
			int value = R_element_max(d_input, row, column, threshold, tau, emb, corrected_size);
			return(value);
		}
		else return(0);
	}
	
	if(block_y==0){ // diagonal
		long int row = gl_index;
		long int column = gl_index;
		if(row >= 0 && row < corrected_size && column  >= 0 && column < corrected_size) {
			int value = R_element_max(d_input, row, column, threshold, tau, emb, corrected_size);
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
	//if(threadIdx.x==0) printf("bl=%d; start_count=%d; end_count=%d;\n", bl, (*start_count), (*end_count));
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
__global__ void GPU_RQA_HST_diagonal_R_matrix(int *d_diagonal_R_matrix, IOtype const* __restrict__ d_input, IOtype threshold, int tau, int emb, long int corrected_size){
	int nBlocks = (corrected_size/blockDim.x) + 1;
	for(int bl=0; bl<nBlocks; bl++){
		long int gl_index = (long int)(bl*blockDim.x + threadIdx.x) - 1;
		int value = GPU_RQA_HST_get_R_matrix_element_diagonal( d_input, threshold, tau, emb, bl, gl_index, corrected_size);
	
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
//			printf("th:%d; bl=%d; s_start=%d; s_end=%d; s_end[]=%d; s_start[]=%d; length=%d;\n", threadIdx.x, bl, start_count, end_count, s_end[threadIdx.x], s_start[threadIdx.x], s_end[threadIdx.x] - s_start[threadIdx.x]);
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
		GPU_RQA_HST_get_R_matrix_element_horizontal(&first, &second, d_input, threshold, tau, emb, bl, nSamples);	
		
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
		GPU_RQA_HST_get_R_matrix_element_vertical(&first, &second, d_input, threshold, tau, emb, bl, nSamples);	
		
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
		first  = GPU_RQA_HST_get_R_matrix_element_diagonal(d_input, threshold, tau, emb, bl, gl_index, nSamples);
		second = GPU_RQA_HST_get_R_matrix_element_diagonal(d_input, threshold, tau, emb, bl, gl_index + 1, nSamples);
		
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

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------


void RQA_HST_init(){
	//---------> Specific nVidia stuff
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
}

//---------------------------- WRAPERS -----------------------------

int GPU_scan_exclusive(unsigned int *d_output, unsigned int *d_input, unsigned int nElements, int nTimeseries){
	//---------> Task specific
	int nThreads = 1024;
	int nBlocks_x = (nElements + nThreads - 1)/nThreads;
	
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize(nBlocks_x, nTimeseries, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG) printf("\n");
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	unsigned int *d_partial_sums;
	if(nBlocks_x==1) {
		d_partial_sums = NULL;
	}
	else {
		int partial_sum_size = nBlocks_x*nTimeseries*sizeof(int);
		checkCudaErrors(cudaMalloc((void **) &d_partial_sums,  partial_sum_size));
		checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sum_size) );
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


int GPU_scan_inclusive(unsigned long long int *d_output, unsigned long long int *d_input, unsigned int nElements, int nTimeseries){
	//---------> Task specific
	int nThreads = 1024;
	int nBlocks_x = (nElements + nThreads - 1)/nThreads;
	
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize(nBlocks_x, nTimeseries, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG) printf("\n");
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	unsigned long long int *d_partial_sums;
	if(nBlocks_x==1) {
		d_partial_sums = NULL;
	}
	else {
		int partial_sum_size = nBlocks_x*nTimeseries*sizeof(int);
		checkCudaErrors(cudaMalloc((void **) &d_partial_sums,  partial_sum_size));
		checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sum_size) );
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
	
	if(DEBUG) printf("Data dimensions: %d;\n", nSamples);
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	//---------------> GPU Allocations
	unsigned long long int *d_histogram;
	size_t histogram_size_in_bytes = (nSamples + 1)*sizeof(unsigned long long int);
	
	checkCudaErrors(cudaMalloc((void **) &d_histogram, histogram_size_in_bytes) );
	checkCudaErrors(cudaMemset(d_histogram, 0, histogram_size_in_bytes) );
	//--------------------------------<
	
	//------------> GPU kernel
	GPU_RQA_HST_length_histogram_direct<RQA_ConstParams><<< gridSize , blockSize >>>(d_histogram, d_input, nSamples);
	//--------------------------------<
	
	//------------> Copy data to host
	checkCudaErrors(cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost));
	//--------------------------------<
	
	//----------> Deallocations
	checkCudaErrors(cudaFree(d_histogram));
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
	double *execution_time,
	int *error
){
	if(*error != ACCRQA_SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	GpuTimer timer;
	
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
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_metric = NULL;
	}
	//--------------------------------<
	
	timer.Start();
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	
	//------------> GPU kernel
	GPU_RQA_HST_length_histogram_horizontal<const_params><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
	
	//------------> Calculation metric
	reverse_array_and_multiply<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaError = cudaMemcpy(h_metric, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	
	//------------> Calculation scan histogram
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	timer.Stop();
	*execution_time = timer.Elapsed();
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ACCRQA_ERROR_CUDA_KERNEL;
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
	long int input_size,
	double *execution_time,
	int *error
){
	if(*error != ACCRQA_SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	GpuTimer timer;
	
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
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_metric = NULL;
	}
	//--------------------------------<
	
	timer.Start();
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	
	//------------> GPU kernel
	GPU_RQA_HST_length_histogram_vertical<const_params><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);

	//------------> Calculation metric
	reverse_array_and_multiply<<< ra_gridSize , ra_blockSize , 0 , NULL >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaDeviceSynchronize();
	cudaError = cudaMemcpy(h_metric, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<

	//------------> Calculation scan histogram
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaDeviceSynchronize();
	cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	timer.Stop();
	*execution_time = timer.Elapsed();
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ACCRQA_ERROR_CUDA_KERNEL;
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
	int *error
){
	if(*error != ACCRQA_SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	GpuTimer timer;
	
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
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_metric = NULL;
	}
	
	timer.Start();
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
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
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	
	//------------> Calculation scan histogram
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaDeviceSynchronize();
	cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	timer.Stop();
	*execution_time = timer.Elapsed();
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ACCRQA_ERROR_CUDA_KERNEL;
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
	double *execution_time,
	int *error
){
	if(*error != ACCRQA_SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	long int corrected_size = input_size - (emb - 1)*tau;
	unsigned int hst_size = corrected_size + 1;
	int nThreads = 1024;
	GpuTimer timer;
	
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
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_histogram, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_histogram = NULL;
	}
	cudaError = cudaMalloc((void **) &d_temporary, histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_temporary = NULL;
	}
	cudaError = cudaMalloc((void **) &d_metric,    histogram_size_in_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_metric = NULL;
	}
	
	timer.Start();
	//---------> Memory copy and preparation
	cudaMemset(d_histogram, 0, histogram_size_in_bytes);
	cudaMemset(d_temporary, 0, histogram_size_in_bytes);
	cudaMemset(d_metric, 0,    histogram_size_in_bytes);
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	
	//------------> GPU kernel
	GPU_RQA_HST_length_histogram_diagonal<const_params><<< gridSize , blockSize >>>(d_histogram, d_input, threshold, tau, emb, corrected_size);
	GPU_RQA_HST_correction_diagonal_half<<< ra_gridSize , ra_blockSize >>>( d_histogram, hst_size );
	
	//------------> Calculation metric
	reverse_array_and_multiply<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaDeviceSynchronize();
	cudaError = cudaMemcpy(h_metric, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	
	//------------> Calculation scan histogram
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_histogram, hst_size);
	GPU_scan_inclusive(d_metric, d_temporary, hst_size, 1);
	reverse_array<<< ra_gridSize , ra_blockSize >>>(d_temporary, d_metric, hst_size);
	cudaDeviceSynchronize();
	cudaError = cudaMemcpy(h_scan_histogram, d_temporary, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	
	//------------> Copy data to host
	cudaError = cudaMemcpy(h_length_histogram, d_histogram, histogram_size_in_bytes, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	//--------------------------------<
	timer.Stop();
	*execution_time = timer.Elapsed();
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ACCRQA_ERROR_CUDA_KERNEL;
	}
	//--------------------------------<
	
	//----------> Deallocations
	if(d_histogram!=NULL) cudaFree(d_histogram);
	if(d_temporary!=NULL) cudaFree(d_temporary);
	if(d_metric!=NULL) cudaFree(d_metric);
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
		printf("[%d; %d]", input[f], output[f]);
	}
	
	cudaFree(input);
	cudaFree(output);
}


//---------------------------- L2 WRAPERS -----------------------------

/*
template<class const_params, typename IOtype>
int GPU_RQA_length_histogram_horizontal_tp(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, IOtype *h_input, IOtype threshold, int tau, int emb, long int input_size, double *execution_time, int *error){
	if(*error != ACCRQA_SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	//---------> Measurements
	*execution_time = 0;
	GpuTimer timer;
	
	//---------> GPU Memory allocation
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t input_size_bytes = input_size*sizeof(IOtype);
	IOtype *d_input;
	checkCudaErrors( cudaMalloc((void **) &d_input, input_size_bytes) );
	checkCudaErrors( cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice) );
	
	//------------- GPU kernel
	timer.Start();
	RQA_length_histogram_horizontal_wrapper<const_params>(h_length_histogram, h_scan_histogram, h_metric, d_input, threshold, tau, emb, corrected_size);
	timer.Stop();
	*execution_time = timer.Elapsed();
	//------------- GPU kernel
	
	checkCudaErrors(cudaFree(d_input));
	
	return(0);
}


template<class const_params, typename IOtype>
int GPU_RQA_length_histogram_vertical_tp(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, IOtype *h_input, IOtype threshold, int tau, int emb, long int input_size, double *execution_time, int *error){
	if(*error != ACCRQA_SUCCESS) return;
	
	//---------> Initial nVidia stuff
	int devCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&devCount);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_DEVICE_NOT_FOUND;
		return;
	}
	
	//---------> Measurements
	*execution_time = 0;
	GpuTimer timer;
	
	//---------> GPU Memory allocation
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t input_size_bytes = input_size*sizeof(IOtype);
	IOtype *d_input;
	checkCudaErrors( cudaMalloc((void **) &d_input, input_size_bytes) );
	checkCudaErrors( cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice) );
	
	//------------- GPU kernel
	timer.Start();
	RQA_length_histogram_vertical_wrapper<const_params>(h_length_histogram, h_scan_histogram, h_metric, d_input, threshold, tau, emb, corrected_size);
	timer.Stop();
	*execution_time = timer.Elapsed();
	//------------- GPU kernel
	
	checkCudaErrors(cudaFree(d_input));
	
	return(0);
}


template<class const_params, typename IOtype>
int GPU_RQA_length_histogram_diagonal_tp(
	unsigned long long int *h_length_histogram, 
	unsigned long long int *h_scan_histogram, 
	unsigned long long int *h_metric, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	long int input_size, 
	int device, 
	double *execution_time
){
	
	//---------> Measurements
	*execution_time = 0;
	GpuTimer timer;
	
	//---------> GPU Memory allocation
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t input_size_bytes = input_size*sizeof(IOtype);
	IOtype *d_input;
	checkCudaErrors( cudaMalloc((void **) &d_input, input_size_bytes) );
	checkCudaErrors( cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice) );
	
	//------------- GPU kernel
	timer.Start();
	RQA_length_histogram_diagonal_wrapper_mk2<const_params>(h_length_histogram, h_scan_histogram, h_metric, d_input, threshold, tau, emb, corrected_size);
	timer.Stop();
	*execution_time = timer.Elapsed();
	//------------- GPU kernel
	
	checkCudaErrors(cudaFree(d_input));
	
	return(0);
}

*/

template<class const_params, typename IOtype>
int GPU_RQA_diagonal_R_matrix_tp(
	int *h_diagonal_R_matrix, 
	IOtype *h_input, 
	IOtype threshold, 
	int tau, 
	int emb, 
	long int input_size, 
	int device, 
	double *execution_time,
	int *error
){
	//---------> Initial nVidia stuff
	int devCount;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount) checkCudaErrors(cudaSetDevice(device));
	else { printf("Wrong device!\n"); return(1); }
	
	//---------> Measurements
	*execution_time = 0;
	GpuTimer timer;
	
	//---------> GPU Memory allocation
	long int corrected_size = input_size - (emb - 1)*tau;
	long int matrix_size = corrected_size*(2*corrected_size-1)*sizeof(int);
	long int input_size_bytes = input_size*sizeof(IOtype);
	IOtype *d_input;
	int *d_diagonal_R_matrix;
	checkCudaErrors(cudaMalloc((void **) &d_input, input_size_bytes) );
	checkCudaErrors(cudaMalloc((void **) &d_diagonal_R_matrix, matrix_size) );

	//---------> Memory copy
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice) );
	
	//------------- GPU kernel
	timer.Start();
	
	int nThreads = 1024;
	dim3 gridSize(2*corrected_size - 1, 1, 1);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG) printf("Input size: %ld; Time step: %d; Embedding: %d; Corrected size: %ld;\n", input_size, tau, emb, corrected_size);
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	//------------> GPU kernel
	GPU_RQA_HST_diagonal_R_matrix<const_params><<< gridSize , blockSize >>>(d_diagonal_R_matrix, d_input, threshold, tau, emb, corrected_size);
	//--------------------------------<
	
	timer.Stop();
	*execution_time = timer.Elapsed();
	//------------- GPU kernel
	
	checkCudaErrors(cudaMemcpy(h_diagonal_R_matrix, d_diagonal_R_matrix, matrix_size, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_diagonal_R_matrix));
	
	return(0);
}


int GPU_RQA_length_start_end_test(unsigned long long int *h_length_histogram, int *h_input, long int nSamples, int device, int nRuns, double *execution_time){
	//---------> Initial nVidia stuff
	int devCount;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount) checkCudaErrors(cudaSetDevice(device));
	else { printf("Wrong device!\n"); return(1); }
	
	//---------> Measurements
	*execution_time = 0;
	GpuTimer timer;
	
	//---------> GPU Memory allocation
	size_t input_size = nSamples*sizeof(int);
	int *d_input;
	checkCudaErrors(cudaMalloc((void **) &d_input, input_size) );

	//---------> Memory allocation
	
	
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
	
	//------------- GPU kernel
	timer.Start();
	//RQA_length_histogram_from_timeseries(h_length_histogram, d_input, nSamples);
	RQA_length_histogram_from_timeseries_direct(h_length_histogram, d_input, nSamples);
	timer.Stop();
	*execution_time = timer.Elapsed();
	//------------- GPU kernel
	
	checkCudaErrors(cudaFree(d_input));
	
	return(0);
}


//---------------------------- L1 WRAPERS -----------------------------

void GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, double *execution_time, int *error){
	RQA_length_histogram_horizontal_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, execution_time, error);
}

void GPU_RQA_length_histogram_horizontal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, double *execution_time, int *error){
	RQA_length_histogram_horizontal_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, execution_time, error);
}

void GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, double *execution_time, int *error){
	RQA_length_histogram_vertical_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, execution_time, error);
}

void GPU_RQA_length_histogram_vertical(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, double *execution_time, int *error){
	RQA_length_histogram_vertical_wrapper<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, execution_time, error);
}

void GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, float *h_input, float threshold, int tau, int emb, long int input_size, int distance_type, double *execution_time, int *error){
	RQA_length_histogram_diagonal_wrapper_mk2<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, execution_time, error);
}

void GPU_RQA_length_histogram_diagonal(unsigned long long int *h_length_histogram, unsigned long long int *h_scan_histogram, unsigned long long int *h_metric, double *h_input, double threshold, int tau, int emb, long int input_size, int distance_type, double *execution_time, int *error){
	RQA_length_histogram_diagonal_wrapper_mk2<RQA_ConstParams>(h_length_histogram, h_scan_histogram, h_metric, h_input, threshold, tau, emb, input_size, execution_time, error);
}

void GPU_RQA_diagonal_R_matrix(
	int *h_diagonal_R_matrix, 
	float *h_input, 
	float threshold, 
	int tau, 
	int emb, 
	long int input_size, 
	int distance_type, 
	int device, 
	double *execution_time,
	int *error
){
	GPU_RQA_diagonal_R_matrix_tp<RQA_ConstParams>(h_diagonal_R_matrix, h_input, threshold, tau, emb, input_size, device, execution_time, error);
}

void GPU_RQA_diagonal_R_matrix(
	int *h_diagonal_R_matrix, 
	double *h_input, 
	double threshold, 
	int tau, 
	int emb, 
	long int input_size, 
	int distance_type, 
	int device, 
	double *execution_time,
	int *error
){
	GPU_RQA_diagonal_R_matrix_tp<RQA_ConstParams>(h_diagonal_R_matrix, h_input, threshold, tau, emb, input_size, device, execution_time, error);
}

// Functions without histogram
void GPU_RQA_DET_noHST(){}

void GPU_RQA_LAM_noHST(){}
// Things to add:
// Calculate DET and LAM for fixed lmin/vmin without producing the histogram


