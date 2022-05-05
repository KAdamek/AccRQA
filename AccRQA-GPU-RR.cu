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
	//int result = R_element_equ(d_input, seed_value_pos, pos_x, threshold, tau, emb, matrix_size);

	size_t matrix_pos_x = (blockIdx.x*NTHREADS + threadIdx.x);
	if(matrix_pos_x<matrix_size){
		d_R_matrix[matrix_pos_x + blockIdx.y*matrix_size] = result;
	}
}

// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************

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
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
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
	
	//---------> Task specific
	int nBlocks_x, nBlocks_y;

	nBlocks_x = (corrected_size + NTHREADS - 1)/(NTHREADS);
	nBlocks_y = (corrected_size + NSEEDS - 1)/(NSEEDS);
	
	dim3 gridSize(nBlocks_x, nBlocks_y, 1);
	dim3 blockSize(NTHREADS, 1, 1);
	
	if(DEBUG) printf("Data dimensions: %llu;\n",corrected_size);
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part (Pulse detection FIR)
	timer.Start();
	
	//---------> Pulse detection FIR
	RQA_R_init();
	GPU_RQA_RR_seedsSM_improved_reduction_kernel<const_params><<< gridSize , blockSize, nThresholds*sizeof(int)>>>(d_RR_metric_integers, d_input, corrected_size, threshold_list, nThresholds, tau, emb);
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part (Pulse detection FIR)
	// ----------------------------------------------->
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
	
	//---------> Task specific
	int nBlocks_x, nBlocks_y;

	nBlocks_x = (size + NTHREADS - 1)/(NTHREADS);
	nBlocks_y = (size);
	
	dim3 gridSize(nBlocks_x, nBlocks_y, 1);
	dim3 blockSize(NTHREADS, 1, 1);
	
	if(DEBUG) printf("Data dimensions: %llu;\n",size);
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part (Pulse detection FIR)
	timer.Start();
	
	//---------> Pulse detection FIR
	RQA_R_init();
	GPU_RQA_R_kernel<const_params><<< gridSize , blockSize >>>(d_R_matrix, d_input, size, threshold, tau, emb);
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part (Pulse detection FIR)
	// ----------------------------------------------->
	return(0);
}


int check_memory(size_t total_size, float multiple){
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem,&total_mem);
	double free_memory     = ((double) free_mem);
	double required_memory = multiple*((double) total_size);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", ((float) total_mem)/(1024.0*1024.0), free_memory/(1024.0*1024.0), required_memory/(1024.0*1024.0));
	if(required_memory>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(3);}
	return(0);
}


template<class const_params, typename IOtype>
int GPU_RQA_R_matrix_tp(
		int *h_R_matrix, 
		IOtype *h_input, 
		unsigned long long int size, 
		IOtype threshold, 
		int tau, 
		int emb, 
		int device, 
		double *execution_time
	){
	//---------> Initial nVidia stuff
	int devCount;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount) checkCudaErrors(cudaSetDevice(device));
	else { printf("Wrong device!\n"); exit(1); }
	
	unsigned long long int matrix_size = size - (emb - 1)*tau;
	
	//---------> Checking memory
	size_t total_size = matrix_size*matrix_size*sizeof(int) + size*sizeof(float);
	if(check_memory(total_size, 1.0)!=0) return(1);
	
	//---------> Measurements
	double exec_time = 0;
	GpuTimer timer;

	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	size_t input_size = size*sizeof(IOtype);
	size_t output_size = matrix_size*matrix_size*sizeof(int);
	IOtype *d_input;
	int *d_R_matrix;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input, input_size));
	checkCudaErrors(cudaMalloc((void **) &d_R_matrix, output_size));
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> FFT calculation
		//-----> Copy chunk of input data to a device
		checkCudaErrors(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
		
		//-----> Compute distance matrix
		RQA_R_GPU<RQA_ConstParams, IOtype>(d_R_matrix, d_input, size, threshold, tau, emb, &exec_time);
		
		*execution_time = exec_time;
		if(DEBUG) printf("RQA R matrix: %fms;\n", exec_time);
		
		checkCudaErrors(cudaGetLastError());
		
		//-----> Copy chunk of output data to host
		checkCudaErrors(cudaMemcpy(h_R_matrix, d_R_matrix, output_size, cudaMemcpyDeviceToHost));
	//------------------------------------<
		
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_R_matrix));
	
	return(0);
}


template<class const_params, typename IOtype>
int GPU_RQA_RR_metric_tp(
		unsigned long long int *h_RR_metric_integer, 
		IOtype *h_input, 
		long int input_size, 
		IOtype *h_threshold_list, 
		int nThresholds, 
		int tau, 
		int emb, 
		int device, 
		double *execution_time
	){
	//---------> Initial nVidia stuff
	int devCount;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount) checkCudaErrors(cudaSetDevice(device));
	else { printf("Wrong device!\n"); exit(1); }
	
	//---------> Checking memory
	size_t total_size = nThresholds*sizeof(float) + nThresholds*sizeof(unsigned long long int) + input_size*sizeof(float);
	if(check_memory(total_size, 1.0)!=0) return(1);
	
	//---------> Measurements
	double exec_time = 0;
	GpuTimer timer;

	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t input_size_bytes = input_size*sizeof(IOtype);
	size_t output_size_bytes = nThresholds*sizeof(IOtype);
	IOtype *d_input;
	IOtype *d_threshold_list;
	unsigned long long int *d_RR_metric_integers;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input, input_size_bytes) );
	checkCudaErrors(cudaMalloc((void **) &d_threshold_list, output_size_bytes) );
	checkCudaErrors(cudaMalloc((void **) &d_RR_metric_integers, nThresholds*sizeof(unsigned long long int)) );
	checkCudaErrors(cudaMemset(d_RR_metric_integers, 0, nThresholds*sizeof(unsigned long long int)) );
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> RR calculation
		//-----> Copy chunk of input data to a device
		checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_threshold_list, h_threshold_list, output_size_bytes, cudaMemcpyHostToDevice));
		
		//-----> Compute RR
		// for more thresholds d_RR_metric_integers should be batched
		// d_threshold_list should be batched but remember that 0 th gives bad results.
		// nThreasholds 
		
		RQA_RR_GPU_sharedmemory_metric<RQA_ConstParams>(d_RR_metric_integers, d_input, corrected_size, d_threshold_list, nThresholds, tau, emb, &exec_time);
		
		*execution_time = exec_time;
		if(DEBUG) printf("RQA recurrent rate: %f;\n", exec_time);
		
		checkCudaErrors(cudaGetLastError());
		
		//-----> Copy chunk of output data to host
		checkCudaErrors(cudaMemcpy(h_RR_metric_integer, d_RR_metric_integers, nThresholds*sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
	//------------------------------------<
		
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_threshold_list));
	checkCudaErrors(cudaFree(d_RR_metric_integers));
	
	return(0);
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
		int device, 
		double *total_execution_time
	){
	// separate thresholds into calculable chunks
	int max_nThresholds = 254;
	int nSteps = (nThresholds)/(max_nThresholds);
	int remainder = nThresholds - nSteps*max_nThresholds;
	vector<int> th_chunks;
	for(int f=0; f<nSteps; f++){
		th_chunks.push_back(max_nThresholds);
	}
	th_chunks.push_back(remainder);
	
	// Debugging
	if(DEBUG_GPU_RR){
		printf("DEBUG_GPU_RR: Chunks:\n");
		printf("DEBUG_GPU_RR:   ");
		for(int f=0; f<(int)th_chunks.size(); f++){
			printf("%d  ", th_chunks[f]);
		}
		printf("\n");
	}
	
	// send a chunk of threshold to the GPU
	int th_shift = 0;
	unsigned long long int temp_rr_count[256];
	IOtype temp_rr_thresholds[256];
	temp_rr_thresholds[0] = 0.0;
	for(int f=0; f<(int) th_chunks.size(); f++){
		memset(temp_rr_count, 0, 256*sizeof(unsigned long long int));
		memcpy(&temp_rr_thresholds[1], &h_threshold_list[th_shift], th_chunks[f]*sizeof(IOtype));
		
		if(DEBUG_GPU_RR){
			printf("DEBUG_GPU_RR: Temporary threshold list:\n");
			printf("DEBUG_GPU_RR:   ");
			for(int i=0; i<(int)(th_chunks[f] + 1); i++){
				printf("%e  ", temp_rr_thresholds[i]);
			}
			printf("\n");
			printf("DEBUG_GPU_RR: nThresholds=%d;\n", th_chunks[f] + 1);
		}
		
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
				device, 
				&execution_time
			);
		(*total_execution_time) = (*total_execution_time) + execution_time;
		
		if(DEBUG_GPU_RR){
			long int corrected_size = input_size - (emb - 1)*tau;
			printf("DEBUG_GPU_RR: Temporary output:\n");
			printf("DEBUG_GPU_RR:   ");
			for(int i=0; i<(int)(th_chunks[f] + 1); i++){
				printf("%e  ", ((double) temp_rr_count[i])/((double) (corrected_size*corrected_size)) );
			}
			printf("\n");
		}
		
		// copy results to global results
		memcpy(&h_RR_metric_integer[th_shift], &temp_rr_count[1], th_chunks[f]*sizeof(unsigned long long int));
		
		th_shift = th_shift + th_chunks[f];
	}
}

//-------------------------------------------------->
//------------ Wrappers for templating 

int GPU_RQA_R_matrix(int *h_R_matrix, double *h_input, unsigned long long int size, double threshold, int tau, int emb, int device, int distance_type, double *execution_time){
	int ret;
	ret = GPU_RQA_R_matrix_tp<RQA_ConstParams, double>(h_R_matrix, h_input, size, threshold, tau, emb, device, execution_time);
	return(ret);
}

int GPU_RQA_R_matrix(int *h_R_matrix, float *h_input, unsigned long long int size, float threshold, int tau, int emb, int device, int distance_type, double *execution_time){
	int ret;
	ret = GPU_RQA_R_matrix_tp<RQA_ConstParams, float>(h_R_matrix, h_input, size, threshold, tau, emb, device, execution_time);
	return(ret);
}


int GPU_RQA_RR_metric(unsigned long long int *h_RR_metric_integer, double *h_input, size_t input_size, double *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, int device, double *execution_time){
	RQA_GPU_RR_metric_batch_runner<RQA_ConstParams, double>(h_RR_metric_integer, h_input, input_size, h_threshold_list, nThresholds, tau, emb, device, execution_time);
	return(0);
}

int GPU_RQA_RR_metric(unsigned long long int *h_RR_metric_integer, float *h_input, size_t input_size, float *h_threshold_list, int nThresholds, int tau, int emb, int distance_type, int device, double *execution_time){
	RQA_GPU_RR_metric_batch_runner<RQA_ConstParams, float>(h_RR_metric_integer, h_input, input_size, h_threshold_list, nThresholds, tau, emb, device, execution_time);
	return(0);
}

//---------------------------------------------------<