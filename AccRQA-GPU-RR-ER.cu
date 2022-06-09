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
		unsigned long long int *d_RR_metric_integers, 
		IOtype const* __restrict__ d_input, 
		unsigned long long int size, 
		IOtype threshold, 
		int tau, 
		int emb
	){
	// Input data
	extern __shared__ int s_local_RR[]; //local recurrent rate
	unsigned long long int pos_x, pos_y;
	__shared__ int s_sums[NTHREADS];
	int local_RR[10]; // WIP: Set this equivalent to emb

	s_sums[threadIdx.x] = 0;
	int sum = 0;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x; // To loop ove the threads one grid size at a time.
	while (i < size*(size-1)/2) {
		// Calculate x,y position in the upper triangle of the RQA matrix from the linear index, i
		pos_x = (unsigned long long int) size - 2 - (int) (sqrtf(-8.0 * i + 4.0 * size * (size - 1) - 7.0) / 2.0 - 0.5);
		pos_y = (unsigned long long int) i + pos_x + 1 - size * (size - 1) / 2 + (size - pos_x) * ((size - pos_x) - 1) / 2;
		
		for (int k = 0; k < emb; ++k)
		{	
			double dist = abs(d_input[pos_x + k * tau] - d_input[pos_y + k * tau]);
            if (pos_x + k * tau >= size || pos_y + k * tau >= size) { // Check that this element is included
				;
			}
            else if (abs(d_input[pos_x + k * tau] - d_input[pos_y + k * tau]) < threshold) { // add ones as long as the distance is within the threshold
				local_RR[k] = local_RR[k] + 1;
			}
            else { // If the distance between the vectors is breater than the threshold, increasing k, the length of the vector cannot decrease the distace. Therefore stop.
                break;
			}
		}	
		i += stride;
	}

	// perform the reuduction
	for (int k = 0; k < emb; ++k){
	s_sums[threadIdx.x] = local_RR[k];
	__syncthreads();
	sum = Reduce_SM(s_sums);
	Reduce_WARP(&sum);
	__syncthreads();
	if(threadIdx.x==0) s_local_RR[k] = sum;
	__syncthreads();
	}

	if(threadIdx.x<emb) {
		atomicAdd(&d_RR_metric_integers[threadIdx.x], s_local_RR[threadIdx.x] );
	}
}
// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************

void RQA_R_WIP_init(){
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
		IOtype threshold, 
		int tau, 
		int emb, 
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	
	dim3 gridSize((corrected_size + NTHREADS - 1)/ NTHREADS); // WIP: The optimal number of grids for the input length should be used, cap this
	dim3 blockSize(NTHREADS);
	if(DEBUG) printf("Data dimensions: %llu;\n",corrected_size);
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	//---------> Kernel execution
	RQA_R_WIP_init();
	GPU_RQA_RR_kernel<const_params><<<gridSize, blockSize, emb*sizeof(int)>>>(d_RR_metric_integers, d_input, corrected_size, threshold, tau, emb);
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int check_memory_WIP(size_t total_size, float multiple){
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
		unsigned long long int *h_RR_metric_integers, 
		IOtype *h_input, 
		unsigned long long int input_size, 
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
	size_t total_size = input_size*sizeof(IOtype) + emb*sizeof(unsigned long long int);
	if(check_memory_WIP(total_size, 1.0)!=0) return(1);
	
	//---------> Measurements
	double exec_time = 0;
	GpuTimer timer;

	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	long int corrected_size = input_size - (emb - 1)*tau;
	size_t input_size_bytes = input_size*sizeof(IOtype);
	IOtype *d_input;
	IOtype d_threshold;
	unsigned long long int *d_RR_metric_integers;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input, input_size_bytes) );
	checkCudaErrors(cudaMalloc((void **) &d_RR_metric_integers, emb*sizeof(unsigned long long int)) );
	checkCudaErrors(cudaMemset(d_RR_metric_integers, 0, emb*sizeof(unsigned long long int)) );
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> RR calculation
		//-----> Copy chunk of input data to a device
		checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
		//-----> Compute RR
		
		RQA_RR_GPU_sharedmemory_metric<RQA_ConstParams>(d_RR_metric_integers, d_input, corrected_size, threshold, tau, emb, &exec_time);
		
		*execution_time = exec_time;
		if(DEBUG) printf("RQA recurrent rate: %f;\n", exec_time);
		
		checkCudaErrors(cudaGetLastError());
		
		//-----> Copy chunk of output data to host
		checkCudaErrors(cudaMemcpy(h_RR_metric_integers, d_RR_metric_integers, emb*sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
		for(int i = 0; i < emb; i++) {
			unsigned long long int cont = h_RR_metric_integers[i];
			printf("Index %d, h_RR_metric_integer, %llu \n", i, cont);
		}
	//------------------------------------<
		
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	
	return(0);
}

//-------------------------------------------------->
//------------ Wrappers for templating 


int GPU_RQA_RR_ER_metric(unsigned long long int *h_RR_metric_integers, double *h_input, size_t input_size, double threshold, int tau, int emb, int distance_type, int device, double *execution_time){
	GPU_RQA_RR_metric_tp<RQA_ConstParams, double>(h_RR_metric_integers, h_input, input_size, threshold, tau, emb, device, execution_time);
	return(0);
}

int GPU_RQA_RR_ER_metric(unsigned long long int *h_RR_metric_integers, float *h_input, size_t input_size, float threshold, int tau, int emb, int distance_type, int device, double *execution_time){
	GPU_RQA_RR_metric_tp<RQA_ConstParams, float>(h_RR_metric_integers, h_input, input_size, threshold, tau, emb, device, execution_time);
	return(0);
}

//---------------------------------------------------<