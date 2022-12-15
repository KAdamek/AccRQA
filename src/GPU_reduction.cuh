#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define HALF_WARP 16

#ifndef RQA_REDUCTION
#define RQA_REDUCTION


__device__ __inline__ int Reduce_SM(int *s_data){
	int l_A = s_data[threadIdx.x];
	
	for (int i = ( blockDim.x >> 1 ); i > HALF_WARP; i = i >> 1) {
		if (threadIdx.x < i) {
			l_A = s_data[threadIdx.x] + s_data[i + threadIdx.x];			
			s_data[threadIdx.x] = l_A;
		}
		__syncthreads();
	}
	
	return(l_A);
}

__device__ __inline__ void Reduce_WARP(int *A){
	int l_A;
	
	for (int q = HALF_WARP; q > 0; q = q >> 1) {
		l_A = __shfl_down_sync(0xFFFFFFFF, (*A), q);
		__syncwarp();
		(*A) = (*A) + l_A;
	}
}

#endif