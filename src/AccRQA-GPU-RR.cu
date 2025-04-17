#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "GPU_timer.h"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iomanip> 
#include <vector>

#include "../include/AccRQA_definitions.hpp"
#include "../include/AccRQA_utilities_distance.hpp"
#include "../include/AccRQA_utilities_error.hpp"
#include "GPU_reduction.cuh"
#include "AccRQA_metrics.cuh"

#define DEBUG_GPU_RR false

using namespace std;

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

	//int result = R_element_max(d_input, seed_value_pos, pos_x, threshold, tau, emb, matrix_size);
	int result = get_RP_element<const_params>(d_input, seed_value_pos, pos_x, threshold, tau, emb);
	
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
	if( threadIdx.x < nThresholds ) s_local_RR[threadIdx.x] = 0;
	pos_x = blockIdx.x*NTHREADS + threadIdx.x;
	pos_y = blockIdx.y*NSEEDS + threadIdx.x;
	
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
					int result = get_RP_element<const_params>(d_input, pos_y, pos_x, threshold, tau, emb);
					sum = sum + 2*result;
				}
				else if( pos_y == pos_x && pos_y<size ){ // diagonal
					int result = get_RP_element<const_params>(d_input, pos_y, pos_x, threshold, tau, emb);
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

// ****************************************************************************
// ****************************************************************************
// ****************************************************************************


void RQA_R_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

int check_memory(size_t total_size, float multiple){
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem,&total_mem);
	double free_memory     = ((double) free_mem);
	double required_memory = multiple*((double) total_size);
	if(DEBUG_GPU_RR) printf("\n");
	if(DEBUG_GPU_RR) printf("Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", ((float) total_mem)/(1024.0*1024.0), free_memory/(1024.0*1024.0), required_memory/(1024.0*1024.0));
	if(required_memory>free_memory) return(3);
	return(0);
}


template<typename IOtype>
void GPU_RQA_R_matrix_tp(
		int *h_R_matrix, 
		IOtype *h_input, 
		unsigned long long int size, 
		IOtype threshold, 
		int tau, 
		int emb, 
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
	
	//---------> Checking memory
	unsigned long long int matrix_size = size - (emb - 1)*tau;
	size_t total_size = matrix_size*matrix_size*sizeof(int) + size*sizeof(float);
	if(check_memory(total_size, 1.0)!=0) {
		*error = ERR_CUDA_NOT_ENOUGH_MEMORY;
		return;
	}
	
	//---------> Memory allocation
	size_t input_size = size*sizeof(IOtype);
	size_t output_size = matrix_size*matrix_size*sizeof(int);
	IOtype *d_input;
	int *d_R_matrix;

	cudaError = cudaMalloc((void **) &d_input, input_size);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_R_matrix, output_size);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_R_matrix = NULL;
	}

	//---------> Memory copy and preparation
	cudaError = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
		
	//-----> Compute distance matrix
	if(*error == SUCCESS) {
		dim3 gridSize(1, 1, 1);
		dim3 blockSize(NTHREADS, 1, 1);
		gridSize.x = (size + NTHREADS - 1)/(NTHREADS);
		gridSize.y = (size);
		
		RQA_R_init();
		switch(distance_type) {
			case DST_EUCLIDEAN:
				GPU_RQA_R_kernel<RQA_euc><<< gridSize , blockSize >>>(d_R_matrix, d_input, size, threshold, tau, emb);
				break;
			case DST_MAXIMAL:
				GPU_RQA_R_kernel<RQA_max><<< gridSize , blockSize >>>(d_R_matrix, d_input, size, threshold, tau, emb);
				break;
			default :
				*error = ERR_INVALID_ARGUMENT;
				break;
		}
	}
	
	//-----> Copy chunk of output data to host
	cudaError = cudaMemcpy(h_R_matrix, d_R_matrix, output_size, cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
	}
	
	//---------> Feeing allocated resources
	if(d_input!=NULL) cudaFree(d_input);
	if(d_R_matrix!=NULL) cudaFree(d_R_matrix);
}

template<typename IOtype>
void GPU_RQA_RR_metric_tp(
		unsigned long long int *h_RR_metric_integer, 
		IOtype *h_input, 
		long int input_size, 
		IOtype *h_threshold_list, 
		int nThresholds, 
		int tau, 
		int emb, 
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
	
	//---------> Checking memory
	size_t total_size = nThresholds*sizeof(float) + nThresholds*sizeof(unsigned long long int) + input_size*sizeof(float);
	if(check_memory(total_size, 1.0)!=0) {
		*error = ERR_CUDA_NOT_ENOUGH_MEMORY;
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
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_input = NULL;
	}
	cudaError = cudaMalloc((void **) &d_threshold_list, output_size_bytes);
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_threshold_list = NULL;
	}
	cudaError = cudaMalloc((void **) &d_RR_metric_integers, nThresholds*sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) { 
		*error = ERR_MEM_ALLOC_FAILURE; 
		d_RR_metric_integers = NULL;
	}
	
	//---------> Memory copy and preparation
	cudaError = cudaMemset(d_RR_metric_integers, 0, nThresholds*sizeof(unsigned long long int));
	if(cudaError != cudaSuccess) {
		*error = ERR_MEM_ALLOC_FAILURE;
	}
	cudaError = cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
	cudaError = cudaMemcpy(d_threshold_list, h_threshold_list, output_size_bytes, cudaMemcpyHostToDevice);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
		
	//-----> Compute RR
	if(*error == SUCCESS){
		dim3 gridSize(1, 1, 1);
		dim3 blockSize(NTHREADS, 1, 1);
		gridSize.x = (corrected_size + NTHREADS - 1)/(NTHREADS);
		gridSize.y = (corrected_size + NSEEDS - 1)/(NSEEDS);
		
		RQA_R_init();
		switch(distance_type) {
			case DST_EUCLIDEAN:
				GPU_RQA_RR_seedsSM_improved_reduction_kernel<RQA_euc><<< gridSize , blockSize, nThresholds*sizeof(int)>>>(d_RR_metric_integers, d_input, corrected_size, d_threshold_list, nThresholds, tau, emb);
				break;
			case DST_MAXIMAL:
				GPU_RQA_RR_seedsSM_improved_reduction_kernel<RQA_max><<< gridSize , blockSize, nThresholds*sizeof(int)>>>(d_RR_metric_integers, d_input, corrected_size, d_threshold_list, nThresholds, tau, emb);
				break;
			default :
				*error = ERR_INVALID_ARGUMENT;
				break;
		}
	}
	
	//-----> Copy chunk of output data to host
	cudaError = cudaMemcpy(h_RR_metric_integer, d_RR_metric_integers, nThresholds*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if(cudaError != cudaSuccess) {
		*error = ERR_CUDA;
	}
		
	//---------> error check
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		*error = ERR_CUDA;
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
		int distance_type, 
		Accrqa_Error *error
){
	if(*error != SUCCESS) return;
	
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
		GPU_RQA_RR_metric_tp<IOtype>(
				temp_rr_count, // 
				h_input, 
				input_size, 
				temp_rr_thresholds, 
				th_chunks[f] + 1, 
				tau, 
				emb, 
				distance_type, 
				error
			);
		
		// copy results to global results
		memcpy(&h_RR_metric_integer[th_shift], &temp_rr_count[1], th_chunks[f]*sizeof(unsigned long long int));
		
		th_shift = th_shift + th_chunks[f];
	}
}

//-------------------------------------------------->
//------------ Wrappers for templating 

void GPU_RQA_R_matrix(int *h_R_matrix, double *h_input, unsigned long long int size, double threshold, int tau, int emb, int distance_type, Accrqa_Error *error){
	GPU_RQA_R_matrix_tp<double>(h_R_matrix, h_input, size, threshold, tau, emb, distance_type, error);
}

void GPU_RQA_R_matrix(int *h_R_matrix, float *h_input, unsigned long long int size, float threshold, int tau, int emb, int distance_type, Accrqa_Error *error){
	GPU_RQA_R_matrix_tp<float>(h_R_matrix, h_input, size, threshold, tau, emb, distance_type, error);
}


void GPU_RQA_RR_metric_integer(unsigned long long int *h_RR_metric_integer, double *h_input, size_t input_size, double *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, Accrqa_Error *error){
	RQA_GPU_RR_metric_batch_runner<double>(h_RR_metric_integer, h_input, input_size, h_threshold_list, nThresholds, tau, emb, distance_type, error);
}

void GPU_RQA_RR_metric_integer(unsigned long long int *h_RR_metric_integer, float *h_input, size_t input_size, float *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, Accrqa_Error *error){
	RQA_GPU_RR_metric_batch_runner<float>(h_RR_metric_integer, h_input, input_size, h_threshold_list, nThresholds, tau, emb, distance_type, error);
}

//---------------------------------------------------<