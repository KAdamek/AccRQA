#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iomanip> 
#include <vector>

#include "../include/AccRQA_definitions.hpp"
#include "GPU_reduction.cuh"
#include "AccRQA_metrics.cuh"

#define DEBUG_GPU_RR false

using namespace std;


class RQA_ConstParams {
public:
	static const int nRows_per_thread = 1;
	static const int warp = 32;
	static const int shared_memory_size = 513;
};

#define NTHREADS 256
#define LAM_NTHREADS 1024
#define Y_STEPS 4
#define X_STEPS 4
#define HALF_WARP 16
#define NSEEDS 32
#define WARP 32
#define BUFFER 32


template<class const_params, typename IOtype>
__global__ void GPU_RQA_R_kernel(
		int *d_R_matrix, 
		IOtype const* __restrict__ d_input, 
		unsigned long long int size, 
		IOtype threshold, 
		int tau, 
		int emb
	){
	// Each thread will load Y
	unsigned long long int matrix_size = size - (emb - 1)*tau;
	unsigned long long int seed_value_pos = blockIdx.y;
	unsigned long long int pos_x = (blockIdx.x*NTHREADS + threadIdx.x);

	int result = R_element_max(d_input, seed_value_pos, pos_x, threshold, tau, emb, matrix_size);

	size_t matrix_pos_x = (blockIdx.x*NTHREADS + threadIdx.x);
	if(matrix_pos_x<matrix_size){
		d_R_matrix[matrix_pos_x + blockIdx.y*matrix_size] = result;
	}
}

// ***********************************************************************
// ***********************************************************************
// ***********************************************************************

template<class const_params, typename IOtype>
__global__ void GPU_RQA_RR_seedsSM_improved_reduction_kernel(
		unsigned long long int *d_RR_metric_integers, 
		IOtype const* __restrict__ d_input, 
		unsigned long long int size, 
		IOtype *threshold_list, 
		int nThresholds, 
		int tau, 
		int emb
	){
	// Input data
	__shared__ int s_seeds[NSEEDS];
	__shared__ int s_sums[NTHREADS];
	extern __shared__ int s_local_RR[]; //local recurrent rate
	unsigned long long int pos_x, pos_y;
	
	//This checks if the threadblock is in the lower half of the R matrix
	//  (blockIdx.y*NSEEDS) represent beginning of the block within R matrix
	//  ((blockIdx.x+1)*NTHREADS - 1) represent end of the block within R matrix 
	//  if beginning of the block at y is greater then end of the block in x then the block
	//  does not have any points on the diagonal or in the upper half of the R matrix
	if( (blockIdx.y*NSEEDS) > ((blockIdx.x+1)*NTHREADS - 1) ) return;
	
	s_sums[threadIdx.x] = 0;
	if( threadIdx.x < NSEEDS ) s_seeds[threadIdx.x] = 0;
	if( threadIdx.x < nThresholds ) s_local_RR[threadIdx.x] = 0;
	pos_x = blockIdx.x*NTHREADS + threadIdx.x;
	pos_y = blockIdx.y*NSEEDS + threadIdx.x;
	
	// i-th row from the R matrix; each thread iterates through these values
	if( threadIdx.x<NSEEDS && pos_y<size ) {
		s_seeds[threadIdx.x] = pos_y;
	}
	
	for(int t=0; t<nThresholds; t++) {
		// for given threshold each thread processes part of the column from R matrix (going through s_seeds)
		IOtype threshold = threshold_list[t];
		int sum = 0;
		if(pos_x<size){
			for(int f=0; f<NSEEDS; f++){
				pos_y = blockIdx.y*NSEEDS + f;
				// We process only upper triangle of the R matrix which the block may cover partially; 
				// this contribution is added twice since lower triangle is the same
				if( pos_y<pos_x && pos_y<size ) {
					//int result = R_element_cartesian(s_seeds[f], elements, threshold); 
					int result = R_element_max(d_input, s_seeds[f], pos_x, threshold, tau, emb, size);
					//int result = R_element_equ(d_input, s_seeds[f], pos_x, threshold, tau, emb, size);
					sum = sum + 2*result;
				}
				else if( pos_y == pos_x ){ // diagonal
					//int result = R_element_cartesian(d_input[pos_y], d_input[pos_x], threshold);
					int result = R_element_max(d_input, pos_y, pos_x, threshold, tau, emb, size);
					//int result = R_element_equ(d_input, pos_y, pos_x, threshold, tau, emb, size);
					sum = sum + result;
				}
			}
		}
		
		s_sums[threadIdx.x] = sum;
		__syncthreads();
		sum = Reduce_SM(s_sums);
		Reduce_WARP(&sum);
		__syncthreads();
		if(threadIdx.x==0) s_local_RR[t] = sum;
	}
	
	__syncthreads();
	
	if(threadIdx.x<nThresholds) {
		atomicAdd(&d_RR_metric_integers[threadIdx.x], s_local_RR[threadIdx.x] );
	}
}

// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************





void RQA_R_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
}

template<class const_params, typename IOtype>
int RQA_RR_GPU_sharedmemory_metric(
		unsigned long long int *d_RR_metric_integers, 
		IOtype *d_input, 
		unsigned long long int corrected_size, 
		IOtype *threshold_list, 
		int nThresholds, 
		int tau, 
		int emb, 
		double *exec_time
	){
	GpuTimer timer;
	
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(NTHREADS, 1, 1);
	gridSize.x = (corrected_size + NTHREADS - 1)/(NTHREADS);
	gridSize.y = (corrected_size + NSEEDS - 1)/(NSEEDS);
	
	timer.Start();
	RQA_R_init();
	GPU_RQA_RR_seedsSM_improved_reduction_kernel<const_params><<< gridSize , blockSize, nThresholds*sizeof(int)>>>(d_RR_metric_integers, d_input, corrected_size, threshold_list, nThresholds, tau, emb);
	timer.Stop();
	*exec_time += timer.Elapsed();
	
	return(0);
}

template<class const_params, typename IOtype>
int RQA_R_GPU(
		int *d_R_matrix, 
		IOtype *d_input, 
		unsigned long long int size, 
		IOtype threshold, 
		int tau, 
		int emb, 
		double *exec_time
	){
	GpuTimer timer;
	
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(NTHREADS, 1, 1);
	gridSize.x = (size + NTHREADS - 1)/(NTHREADS);
	gridSize.y = (size);
	
	timer.Start();
	RQA_R_init();
	GPU_RQA_R_kernel<const_params><<< gridSize , blockSize >>>(d_R_matrix, d_input, size, threshold, tau, emb);
	timer.Stop();
	*exec_time += timer.Elapsed();

	return(0);
}


int check_memory(size_t total_size, float multiple){
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem,&total_mem);
	double free_memory     = ((double) free_mem);
	double required_memory = multiple*((double) total_size);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", ((float) total_mem)/(1024.0*1024.0), free_memory/(1024.0*1024.0), required_memory/(1024.0*1024.0));
	if(required_memory>free_memory) return(3);
	return(0);
}


template<class const_params, typename IOtype>
void GPU_RQA_R_matrix_tp(
		int *h_R_matrix, 
		IOtype *h_input, 
		unsigned long long int size, 
		IOtype threshold, 
		int tau, 
		int emb, 
		int device, 
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
	
	//---------> Checking memory
	unsigned long long int matrix_size = size - (emb - 1)*tau;
	size_t total_size = matrix_size*matrix_size*sizeof(int) + size*sizeof(float);
	if(check_memory(total_size, 1.0)!=0) {
		*error = ACCRQA_ERROR_CUDA_NOT_ENOUGH_MEMORY;
		return;
	}
	
	//---------> Memory allocation
	size_t input_size = size*sizeof(IOtype);
	size_t output_size = matrix_size*matrix_size*sizeof(int);
	IOtype *d_input;
	int *d_R_matrix;

	cudaError = cudaMalloc((void **) &d_input, input_size);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_R_matrix, output_size);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_R_matrix = NULL;
	}

	//---------> Memory copy and preparation
	cudaError = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
		
	//-----> Compute distance matrix
	if(*error == ACCRQA_SUCCESS) {
		RQA_R_GPU<RQA_ConstParams, IOtype>(d_R_matrix, d_input, size, threshold, tau, emb, execution_time);
	}

	if(DEBUG) printf("RQA R matrix: %fms;\n", *execution_time);
	
	//-----> Copy chunk of output data to host
	cudaError = cudaMemcpy(h_R_matrix, d_R_matrix, output_size, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ACCRQA_ERROR_CUDA_KERNEL;
	}
	
	//---------> Feeing allocated resources
	if(d_input!=NULL) cudaFree(d_input);
	if(d_R_matrix!=NULL) cudaFree(d_R_matrix);
}


template<class const_params, typename IOtype>
void GPU_RQA_RR_metric_tp(
		unsigned long long int *h_RR_metric_integer, 
		IOtype *h_input, 
		long int input_size, 
		IOtype *h_threshold_list, 
		int nThresholds, 
		int tau, 
		int emb, 
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
	
	//---------> Checking memory
	size_t total_size = nThresholds*sizeof(float) + nThresholds*sizeof(unsigned long long int) + input_size*sizeof(float);
	if(check_memory(total_size, 1.0)!=0) {
		*error = ACCRQA_ERROR_CUDA_NOT_ENOUGH_MEMORY;
		return;
	}
	
	//---------> Memory allocation
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t input_size_bytes = input_size*sizeof(IOtype);
	size_t output_size_bytes = nThresholds*sizeof(IOtype);
	IOtype *d_input;
	IOtype *d_threshold_list;
	unsigned long long int *d_RR_metric_integers;

	cudaError = cudaMalloc((void **) &d_input, input_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_threshold_list, output_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_threshold_list = NULL;
	}
	cudaError = cudaMalloc((void **) &d_RR_metric_integers, nThresholds*sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION; 
		d_RR_metric_integers = NULL;
	}
	
	//---------> Memory copy and preparation
	cudaError = cudaMemset(d_RR_metric_integers, 0, nThresholds*sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_ALLOCATION;
	}
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
	cudaError = cudaMemcpy(d_threshold_list, h_threshold_list, output_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
		
	//-----> Compute RR
	if(*error == ACCRQA_SUCCESS){
		RQA_RR_GPU_sharedmemory_metric<RQA_ConstParams>(d_RR_metric_integers, d_input, corrected_size, d_threshold_list, nThresholds, tau, emb, execution_time);
	}
	
	if(DEBUG) printf("RQA recurrent rate: %f;\n", *execution_time);
	
	//-----> Copy chunk of output data to host
	cudaError = cudaMemcpy(h_RR_metric_integer, d_RR_metric_integers, nThresholds*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ACCRQA_ERROR_CUDA_MEMORY_COPY;
	}
		
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ACCRQA_ERROR_CUDA_KERNEL;
	}
	
	//---------> Freeing allocated resources
	if(d_input!=NULL) cudaFree(d_input);
	if(d_threshold_list!=NULL) cudaFree(d_threshold_list);
	if(d_RR_metric_integers!=NULL) cudaFree(d_RR_metric_integers);
}

template<class const_params, typename IOtype>
void RQA_GPU_RR_metric_batch_runner(
		unsigned long long int *h_RR_metric_integer, 
		IOtype *h_input, 
		size_t input_size, 
		IOtype *h_threshold_list, 
		int nThresholds, 
		int tau, 
		int emb, 
		double *total_execution_time,
		int *error
){
	if(*error != ACCRQA_SUCCESS) return;
	
	// separate thresholds into calculable chunks
	int max_nThresholds = 254;
	int nSteps = (nThresholds)/(max_nThresholds);
	int remainder = nThresholds - nSteps*max_nThresholds;
	std::vector<int> th_chunks;
	for(int f=0; f<nSteps; f++){
		th_chunks.push_back(max_nThresholds);
	}
	th_chunks.push_back(remainder);
	
	// send a chunk of threshold to the GPU
	int th_shift = 0;
	unsigned long long int temp_rr_count[256];
	IOtype temp_rr_thresholds[256];
	temp_rr_thresholds[0] = 0.0;
	for(int f=0; f<(int) th_chunks.size(); f++){
		memset(temp_rr_count, 0, 256*sizeof(unsigned long long int));
		memcpy(&temp_rr_thresholds[1], &h_threshold_list[th_shift], th_chunks[f]*sizeof(IOtype));
		
		// calculate RR
		double execution_time = 0;
		*total_execution_time = 0;
		GPU_RQA_RR_metric_tp<const_params,IOtype>(
				temp_rr_count, // 
				h_input, 
				input_size, 
				temp_rr_thresholds, 
				th_chunks[f] + 1, 
				tau, 
				emb, 
				&execution_time,
				error
			);
		(*total_execution_time) = (*total_execution_time) + execution_time;
		
		// copy results to global results
		memcpy(&h_RR_metric_integer[th_shift], &temp_rr_count[1], th_chunks[f]*sizeof(unsigned long long int));
		
		th_shift = th_shift + th_chunks[f];
	}
}

//-------------------------------------------------->
//------------ Wrappers for templating 

void GPU_RQA_R_matrix(int *h_R_matrix, double *h_input, unsigned long long int size, double threshold, int tau, int emb, int device, int distance_type, double *execution_time, int *error){
	GPU_RQA_R_matrix_tp<RQA_ConstParams, double>(h_R_matrix, h_input, size, threshold, tau, emb, device, execution_time, error);
}

void GPU_RQA_R_matrix(int *h_R_matrix, float *h_input, unsigned long long int size, float threshold, int tau, int emb, int device, int distance_type, double *execution_time, int *error){
	GPU_RQA_R_matrix_tp<RQA_ConstParams, float>(h_R_matrix, h_input, size, threshold, tau, emb, device, execution_time, error);
}


void GPU_RQA_RR_metric_integer(unsigned long long int *h_RR_metric_integer, double *h_input, size_t input_size, double *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, double *execution_time, int *error){
	RQA_GPU_RR_metric_batch_runner<RQA_ConstParams, double>(h_RR_metric_integer, h_input, input_size, h_threshold_list, nThresholds, tau, emb, execution_time, error);
}

void GPU_RQA_RR_metric_integer(unsigned long long int *h_RR_metric_integer, float *h_input, size_t input_size, float *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, double *execution_time, int *error){
	RQA_GPU_RR_metric_batch_runner<RQA_ConstParams, float>(h_RR_metric_integer, h_input, input_size, h_threshold_list, nThresholds, tau, emb, execution_time, error);
}

//---------------------------------------------------<