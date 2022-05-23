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
__global__ void GPU_RQA_RR_kernel(
		unsigned long long int d_RR_metric_integer, 
		IOtype const* __restrict__ d_input, 
		unsigned long long int size, 
		IOtype threshold, 
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
	pos_x = blockIdx.x*NTHREADS + threadIdx.x;
	pos_y = blockIdx.y*NSEEDS + threadIdx.x;
	
	// i-th row from the R matrix; each thread iterates through these values
	if( threadIdx.x<NSEEDS && pos_y<size ) {
		s_seeds[threadIdx.x] = pos_y;
	}
	
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
		
		s_sums[threadIdx.x] = sum;
		__syncthreads();
		sum = Reduce_SM(s_sums);
		Reduce_WARP(&sum);
		__syncthreads();
		if(threadIdx.x==0) s_local_RR[t] = sum;
	}
	
	__syncthreads();
	

}
// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************

template<class const_params, typename IOtype>
int RQA_RR_GPU_sharedmemory_metric(
		unsigned long long int d_RR_metric_integer, 
		IOtype *d_input, 
		unsigned long long int corrected_size, 
		IOtype threshold, 
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
	// --------> Measured part
	timer.Start();
	
	//---------> Kernel execution
	RQA_R_init();
	GPU_RQA_RR_kernel<const_params><<< gridSize , blockSize, 1*sizeof(int)>>>(d_RR_metric_integer, d_input, corrected_size, threshold, tau, emb);
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
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
int GPU_RQA_RR_metric_tp(
		unsigned long long int *h_RR_metric_integer, 
		IOtype *h_input, 
		long int input_size, 
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
	
	//---------> Checking memory
	size_t total_size = input_size*sizeof(IOtype);
	if(check_memory(total_size, 1.0)!=0) return(1);
	
	//---------> Measurements
	double exec_time = 0;
	GpuTimer timer;

	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t input_size_bytes = input_size*sizeof(IOtype);
	IOtype *d_input;
	IOtype d_threshold;
	unsigned long long int d_RR_metric_integer;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input, input_size_bytes) );
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> RR calculation
		//-----> Copy chunk of input data to a device
		checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
		//-----> Compute RR
		
		RQA_RR_GPU_sharedmemory_metric<RQA_ConstParams>(d_RR_metric_integer, d_input, corrected_size, threshold, tau, emb, &exec_time);
		
		*execution_time = exec_time;
		if(DEBUG) printf("RQA recurrent rate: %f;\n", exec_time);
		
		checkCudaErrors(cudaGetLastError());
		
		//-----> Copy chunk of output data to host
		checkCudaErrors(cudaMemcpy(h_RR_metric_integer, d_RR_metric_integer, 1*sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
	//------------------------------------<
		
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	
	return(0);
}

template<class const_params, typename IOtype>
void RQA_GPU_RR_metric_batch_runner(
		unsigned long long int *h_RR_metric_integer, 
		IOtype *h_input, 
		size_t input_size, 
		IOtype threshold,
		int tau, 
		int emb, 
		int device, 
		double *total_execution_time
	){
	// calculate RR
	double execution_time = 0;
	*total_execution_time = 0;
	GPU_RQA_RR_metric_tp<const_params,IOtype>(
			temp_rr_count, // 
			h_input, 
			input_size, 
			threshold,
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


}

//-------------------------------------------------->
//------------ Wrappers for templating 


int GPU_RQA_RR_metric(unsigned long long int *h_RR_metric_integer, double *h_input, size_t input_size, double threshold, int tau, int emb, int distance_type, int device, double *execution_time){
	RQA_GPU_RR_metric_batch_runner<RQA_ConstParams, double>(h_RR_metric_integer, h_input, input_size, threshold, tau, emb, device, execution_time);
	return(0);
}

int GPU_RQA_RR_metric(unsigned long long int *h_RR_metric_integer, float *h_input, size_t input_size, float threshold, int tau, int emb, int distance_type, int device, double *execution_time){
	RQA_GPU_RR_metric_batch_runner<RQA_ConstParams, float>(h_RR_metric_integer, h_input, input_size, threshold, tau, emb, device, execution_time);
	return(0);
}

//---------------------------------------------------<